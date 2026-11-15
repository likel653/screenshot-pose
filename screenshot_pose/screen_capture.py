"""User-invoked region capture. Saves only the selected region; no clipboard use."""
import os,ctypes
from pathlib import Path

def selection_box(start,end,width,height,minimum=16):
 x1,x2=sorted((max(0,min(width,int(start[0]))),max(0,min(width,int(end[0])))))
 y1,y2=sorted((max(0,min(height,int(start[1]))),max(0,min(height,int(end[1])))))
 return (x1,y1,x2,y2) if x2-x1>=minimum and y2-y1>=minimum else None

def dpi_awareness():
 if os.name=='nt':
  try:
   ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
  except (OSError,AttributeError):
   try:ctypes.windll.shcore.SetProcessDpiAwareness(2)
   except (OSError,AttributeError):pass

def desktop_bounds():
 if os.name=='nt':
  u=ctypes.windll.user32
  x,y=u.GetSystemMetrics(76),u.GetSystemMetrics(77)
  return x,y,x+u.GetSystemMetrics(78),y+u.GetSystemMetrics(79)
 return 0,0,1920,1080

def monitor_bounds():
 if os.name!='nt':return [desktop_bounds()]
 from ctypes import wintypes as w
 monitors=[]
 callback=ctypes.WINFUNCTYPE(w.BOOL,w.HANDLE,w.HDC,ctypes.POINTER(w.RECT),w.LPARAM)
 def visit(handle,dc,rect,data):
  r=rect.contents;monitors.append((r.left,r.top,r.right,r.bottom));return True
 ctypes.windll.user32.EnumDisplayMonitors(None,None,callback(visit),0)
 return monitors or [desktop_bounds()]

class RegionCapture:
 def __init__(self,root,delay=0,grabber=None,show_ui=True,mode='IMAGE',origin=None):
  import tkinter as tk
  from tkinter import ttk
  self.root=root;self.result=None;self.snapshot=None;self.anchor=None;self.canvas=None;self.show_ui=show_ui;self.error=None
  self.mode=mode;self.origin=origin;self.selected_box=None
  self.remaining=int(delay);self.grabber=grabber;self.finished=False
  root.withdraw();root.title('Screenshot Pose · 截图');root.protocol('WM_DELETE_WINDOW',self.cancel)
  root.bind('<Escape>',self.cancel);root.bind('<Button-3>',self.cancel)
  self.message=tk.StringVar();panel=ttk.Frame(root,padding=22);panel.pack(fill='both',expand=True)
  ttk.Label(panel,textvariable=self.message,font=('',12)).pack(pady=10)
  ttk.Button(panel,text='取消截图',command=self.cancel).pack()
  root.geometry('420x140+100+100')
  if show_ui:root.attributes('-topmost',True);root.deiconify()
  self.tick()
 def tick(self):
  if self.finished:return
  if self.remaining>0:
   self.message.set(f'{self.remaining} 秒后截图，请切换到参考图片\n截图后拖框选择人物；Esc 或右键取消')
   self.remaining-=1;self.root.after(1000,self.tick)
  else:
   self.root.withdraw();self.root.update_idletasks();self.root.after(200,self.begin)
 def begin(self):
  if self.finished:return
  try:self._begin()
  except Exception as exc:
   self.error=exc;self.cancel()
 def _begin(self):
  if self.finished:return
  import tkinter as tk
  from PIL import ImageGrab,ImageTk
  self.snapshot=(self.grabber or (lambda:ImageGrab.grab(all_screens=True)))().convert('RGB')
  for child in self.root.winfo_children():child.destroy()
  w,h=self.snapshot.size;self.width=w;self.height=h
  self.root.overrideredirect(True);self.root.geometry(f'{w}x{h}+0+0')
  self.canvas=tk.Canvas(self.root,width=w,height=h,highlightthickness=0,cursor='crosshair');self.canvas.pack(fill='both',expand=True)
  self.photo=ImageTk.PhotoImage(self.snapshot)
  self.canvas.create_image(0,0,image=self.photo,anchor='nw')
  self.canvas.bind('<ButtonPress-1>',self.press);self.canvas.bind('<B1-Motion>',self.motion);self.canvas.bind('<ButtonRelease-1>',self.release)
  self.root.bind('<Escape>',self.cancel);self.root.bind('<Button-3>',self.cancel)
  if self.show_ui:
   self.root.deiconify();self.root.update_idletasks()
   if os.name=='nt':
    user32=ctypes.windll.user32;user32.GetParent.argtypes=[ctypes.c_void_p];user32.GetParent.restype=ctypes.c_void_p
    user32.SetWindowPos.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_uint]
    hwnd=user32.GetParent(self.root.winfo_id()) or self.root.winfo_id()
    user32.SetWindowPos(hwnd,ctypes.c_void_p(-1),user32.GetSystemMetrics(76),user32.GetSystemMetrics(77),w,h,0x0040)
   self.root.lift();self.root.focus_force()
  self.show_hint('拖框选择录制区域 · 松手后准备录制 · Esc / 右键取消' if self.mode=='VIDEO' else '拖框选择人物 · 松手即识别 · Esc / 右键取消')
 def show_hint(self,text):
  self.canvas.delete('hint');x=min(420,self.width//2)
  self.canvas.create_rectangle(max(0,x-350),18,min(self.width,x+350),64,fill='#18232d',outline='',tags='hint')
  self.canvas.create_text(x,41,text=text,fill='white',font=('',13),tags='hint')
 def press(self,event):
  self.anchor=(event.x,event.y);self.canvas.delete('selection')
 def motion(self,event):
  if self.anchor is None:return
  self.canvas.delete('selection');self.canvas.create_rectangle(*self.anchor,event.x,event.y,outline='#65d8ff',width=3,tags='selection')
 def release(self,event):
  if self.anchor is None:return
  box=selection_box(self.anchor,(event.x,event.y),self.width,self.height)
  self.anchor=None
  if box is None:self.show_hint('选区太小，请重新框选人物 · Esc 取消');return
  origin=self.origin if self.origin is not None else desktop_bounds()[:2]
  self.selected_box=tuple(value+origin[i%2] for i,value in enumerate(box))
  if self.mode!='VIDEO':self.result=self.snapshot.crop(box)
  self.finished=True;self.root.destroy()
 def cancel(self,event=None):
  if not self.finished:self.finished=True;self.root.destroy()

def capture(delay=0):
 if os.name!='nt':raise ValueError('直接截图目前仅在 Windows 上验证，请使用图片文件识别')
 dpi_awareness()
 import tkinter as tk
 root=tk.Tk();selector=RegionCapture(root,delay);root.mainloop()
 if selector.error:raise RuntimeError('无法截取桌面：'+str(selector.error))
 return selector.result

def select_region():
 if os.name!='nt':raise ValueError('屏幕录制目前仅支持 Windows')
 dpi_awareness()
 import tkinter as tk
 root=tk.Tk();selector=RegionCapture(root,0,mode='VIDEO');root.mainloop()
 if selector.error:raise RuntimeError('无法选择录制区域：'+str(selector.error))
 return selector.selected_box
