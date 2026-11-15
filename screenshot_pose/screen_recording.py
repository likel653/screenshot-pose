"""Visible-region Windows recording, with fixed-time CFR output and cancellation."""
import ctypes,math,os,threading,time,queue
from pathlib import Path
try:from . import screen_capture
except ImportError:import screen_capture

class RecordingCancelled(Exception):pass

def check_options(box,duration,fps):
 if not math.isfinite(duration) or not .5<=duration<=120:raise ValueError('录制时长应为 0.5 至 120 秒')
 if isinstance(fps,bool) or not isinstance(fps,int) or not 1<=fps<=60:raise ValueError('录制帧率应为 1 至 60 的整数')
 if len(box)!=4 or any(isinstance(v,bool) or not isinstance(v,int) for v in box):raise ValueError('录制区域坐标无效')
 left,top,right,bottom=box;w,h=right-left,bottom-top
 if w<32 or h<32 or w*h>16000000:raise ValueError('录制区域过小或过大，请重新框选视频区域')
 return w,h

def panel_position(box,bounds,width=400,height=155,monitors=None):
 """Place recorder controls entirely outside the recorded region if possible."""
 left,top,right,bottom=bounds
 screens=monitors or [bounds];candidates=[]
 for l,t,r,b in screens:candidates.extend([(l+12,t+12),(r-width-12,t+12),(l+12,b-height-12),(r-width-12,b-height-12)])
 candidates.extend([(box[0]-width-12,box[1]),(box[2]+12,box[1]),(box[0],box[1]-height-12),(box[0],box[3]+12)])
 for x,y in candidates:
  if x<left or y<top or x+width>right or y+height>bottom:continue
  if not any(x>=l and y>=t and x+width<=r and y+height<=b for l,t,r,b in screens):continue
  overlap=max(0,min(x+width,box[2])-max(x,box[0]))*max(0,min(y+height,box[3])-max(y,box[1]))
  if not overlap:return x,y
 return None

class GdiGrabber:
 """Capture only the selected physical pixels, including negative monitor origins."""
 def __init__(self,box):
  if os.name!='nt':raise ValueError('屏幕录制仅支持 Windows')
  from ctypes import wintypes as w
  self.box=box;self.width=box[2]-box[0];self.height=box[3]-box[1]
  self.user=ctypes.WinDLL('user32',use_last_error=True);self.gdi=ctypes.WinDLL('gdi32',use_last_error=True)
  self.dc=self.memory=self.bitmap=self.old=None;self.bits=ctypes.c_void_p()
  specs=[(self.user,'GetDC',[w.HWND],w.HDC),(self.user,'ReleaseDC',[w.HWND,w.HDC],ctypes.c_int),
   (self.gdi,'CreateCompatibleDC',[w.HDC],w.HDC),(self.gdi,'DeleteDC',[w.HDC],w.BOOL),
   (self.gdi,'SelectObject',[w.HDC,w.HANDLE],w.HANDLE),(self.gdi,'DeleteObject',[w.HANDLE],w.BOOL),
   (self.gdi,'BitBlt',[w.HDC,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,w.HDC,ctypes.c_int,ctypes.c_int,w.DWORD],w.BOOL),
   (self.gdi,'CreateDIBSection',[w.HDC,ctypes.c_void_p,w.UINT,ctypes.POINTER(ctypes.c_void_p),w.HANDLE,w.DWORD],w.HANDLE)]
  for lib,name,args,result in specs:fn=getattr(lib,name);fn.argtypes=args;fn.restype=result
  class Header(ctypes.Structure):
   _fields_=[('size',w.DWORD),('width',w.LONG),('height',w.LONG),('planes',w.WORD),('bits',w.WORD),('compression',w.DWORD),('image_size',w.DWORD),('xppm',w.LONG),('yppm',w.LONG),('colors',w.DWORD),('important',w.DWORD)]
  try:
   self.dc=self.user.GetDC(None);self.memory=self.gdi.CreateCompatibleDC(self.dc)
   header=Header(ctypes.sizeof(Header),self.width,-self.height,1,32,0,0,0,0,0,0)
   self.bitmap=self.gdi.CreateDIBSection(self.dc,ctypes.byref(header),0,ctypes.byref(self.bits),None,0)
   if not self.dc or not self.memory or not self.bitmap or not self.bits:raise OSError('无法创建选区录制缓冲区')
   self.old=self.gdi.SelectObject(self.memory,self.bitmap)
  except Exception:self.close();raise
 def grab(self):
  import numpy as np
  if not self.gdi.BitBlt(self.memory,0,0,self.width,self.height,self.dc,self.box[0],self.box[1],0x00CC0020|0x40000000):raise OSError('无法读取屏幕选区')
  buffer=(ctypes.c_ubyte*(self.width*self.height*4)).from_address(self.bits.value)
  return np.frombuffer(buffer,dtype=np.uint8).reshape(self.height,self.width,4)[:,:,:3].copy()
 def close(self):
  if self.old and self.memory:self.gdi.SelectObject(self.memory,self.old)
  if self.bitmap:self.gdi.DeleteObject(self.bitmap)
  if self.memory:self.gdi.DeleteDC(self.memory)
  if self.dc:self.user.ReleaseDC(None,self.dc)
  self.dc=self.memory=self.bitmap=self.old=None

def record(box,output,duration=5.,fps=15,*,stop=None,progress=None,grabber=None,clock=time.perf_counter,wait=None,escape=True):
 """Sample by wall-clock time. Late captures duplicate frames instead of speeding up."""
 import cv2
 width,height=check_options(box,duration,fps);output=Path(output).resolve()
 if output.suffix.lower()!='.avi' or output.exists():raise ValueError('录制输出必须是新的 AVI 文件')
 output.parent.mkdir(parents=True,exist_ok=True);temp=output.with_name(output.stem+'.partial.avi')
 if temp.exists():raise ValueError('录制临时文件已存在，请使用新目录')
 stop=stop or threading.Event();sleep=wait or stop.wait
 factor=min(1.,1600/max(width,height));w=max(2,round(width*factor));h=max(2,round(height*factor));encoded=(w+w%2,h+h%2)
 writer=None;capture=None;index=0;captures=0;duplicates=0;last=None;target=math.ceil(duration*fps)
 def cancelled():
  return stop.is_set() or (escape and os.name=='nt' and bool(ctypes.windll.user32.GetAsyncKeyState(0x1B)&0x8000))
 def emit(frame,duplicate=False):
  nonlocal index,duplicates
  if cancelled():raise RecordingCancelled()
  writer.write(frame);index+=1;duplicates+=int(duplicate)
 try:
  capture=grabber or GdiGrabber(box)
  writer=cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*'MJPG'),fps,encoded)
  if not writer.isOpened():raise ValueError('本地视频编码器无法创建录制文件')
  started=clock()
  while index<target:
   if cancelled():raise RecordingCancelled()
   elapsed=clock()-started
   if elapsed>=duration and last is not None:
    while index<target:emit(last,True)
    break
   due=index/fps
   if elapsed<due:sleep(min(.02,due-elapsed));continue
   frame=capture.grab();captures+=1;stamp=clock()-started
   if frame.shape!=(height,width,3):raise ValueError('屏幕尺寸发生变化，请重新录制')
   if stamp>=duration:
    if last is None:raise ValueError('选区采集速度过慢，请缩小范围后重试')
    while index<target:emit(last,True)
    break
   if (w,h)!=(width,height):frame=cv2.resize(frame,(w,h),interpolation=cv2.INTER_AREA)
   if encoded!=(w,h):frame=cv2.copyMakeBorder(frame,0,encoded[1]-h,0,encoded[0]-w,cv2.BORDER_REPLICATE)
   upto=min(target,int(stamp*fps)+1)
   while index<upto-1:emit(last if last is not None else frame,True)
   emit(frame);last=frame
   if progress:progress(dict(elapsed=min(stamp,duration),duration=duration,frames=index,captures=captures,duplicates=duplicates))
  while clock()-started<duration:
   if cancelled():raise RecordingCancelled()
   sleep(min(.02,duration-(clock()-started)))
  writer.release();writer=None
  if cancelled():raise RecordingCancelled()
  verify=cv2.VideoCapture(str(temp))
  try:
   if not verify.isOpened() or int(verify.get(cv2.CAP_PROP_FRAME_COUNT))!=target:raise ValueError('录制视频验证失败，未保存为完整文件')
  finally:verify.release()
  os.replace(temp,output)
  return dict(video=str(output),bbox=list(box),width=encoded[0],height=encoded[1],fps=fps,frames=index,captures=captures,duplicates=duplicates,
    requested_seconds=duration,video_seconds=index/fps,elapsed_seconds=clock()-started,audio=False)
 finally:
  if writer:writer.release()
  if capture and grabber is None:capture.close()
  temp.unlink(missing_ok=True)

class RecordingWindow:
 def __init__(self,root,box,output,duration,fps,progress=None,show_ui=True,record_fn=record):
  import tkinter as tk
  from tkinter import ttk
  self.root=root;self.box=box;self.output=output;self.duration=duration;self.fps=fps;self.progress=progress;self.result=None;self.error=None
  self.stop=threading.Event();self.events=queue.Queue();self.thread=None;self.progress_warning=None;self.position=panel_position(box,screen_capture.desktop_bounds(),monitors=screen_capture.monitor_bounds())
  root.title('Screenshot Pose · 录制视频区域');root.overrideredirect(True);root.attributes('-topmost',True);root.protocol('WM_DELETE_WINDOW',self.cancel);root.bind('<Escape>',lambda e:self.cancel())
  panel=ttk.Frame(root,padding=14);panel.pack(fill='both',expand=True);self.message=tk.StringVar(value=f'选区已就绪 · 录制 {duration:g} 秒\n播放视频后点击开始；Esc 可取消')
  ttk.Label(panel,text='区域录制 · Screenshot Pose').pack()
  ttk.Label(panel,textvariable=self.message).pack(pady=6)
  buttons=ttk.Frame(panel);buttons.pack();self.start_button=ttk.Button(buttons,text='开始录制',command=self.start);self.start_button.pack(side='left',padx=8)
  ttk.Button(buttons,text='取消',command=self.cancel).pack(side='left',padx=8)
  root.geometry('400x155+100+100');root.update_idletasks()
  if self.position and os.name=='nt':
   user=ctypes.windll.user32;user.GetParent.argtypes=[ctypes.c_void_p];user.GetParent.restype=ctypes.c_void_p
   user.SetWindowPos.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_uint]
   hwnd=user.GetParent(root.winfo_id()) or root.winfo_id();user.SetWindowPos(hwnd,ctypes.c_void_p(-1),*self.position,400,155,0x0040)
  if not show_ui:root.withdraw()
  self.record_fn=record_fn
 def start(self):
  if self.thread:return
  self.start_button.state(['disabled']);self.message.set('录制中… Esc 可取消')
  if self.position is None:self.root.withdraw()
  self.root.update_idletasks()
  def run():
   try:
    time.sleep(.12)
    result=self.record_fn(self.box,self.output,self.duration,self.fps,stop=self.stop,progress=lambda stats:self.events.put(('progress',stats)))
    self.events.put(('done',result))
   except RecordingCancelled:self.events.put(('done',None))
   except Exception as exc:self.events.put(('error',exc))
  self.thread=threading.Thread(target=run,daemon=True);self.thread.start();self.root.after(80,self.poll)
 def poll(self):
  latest=None;terminal=None
  try:
   while True:
    kind,value=self.events.get_nowait()
    if kind=='progress':latest=value
    else:terminal=(kind,value)
  except queue.Empty:pass
  # Always deliver completion, even when queued progress updates could not be
  # published. Coalesce old updates instead of writing every captured frame.
  if terminal:
   kind,value=terminal
   if kind=='error':self.error=value
   else:self.result=value
   self.root.destroy();return
  if latest:
   self.message.set('正在录制 · 剩余 %.1f 秒\nEsc 或取消按钮可取消'%(max(0,self.duration-latest['elapsed'])))
   if self.progress:
    try:
     if self.progress(latest) is False:self.progress_warning='进度文件暂不可写，录制继续'
    except Exception as exc:
     self.progress_warning=str(exc)
  self.root.after(80,self.poll)
 def cancel(self):
  self.stop.set()
  if self.thread:self.message.set('正在取消录制…')
  else:self.root.destroy()

def capture_video(box,output,duration,fps,progress=None):
 import tkinter as tk
 root=tk.Tk();ui=RecordingWindow(root,box,output,duration,fps,progress);root.mainloop()
 if ui.error:raise ui.error
 if ui.result and ui.progress_warning:ui.result['progress_warning']=ui.progress_warning
 return ui.result
