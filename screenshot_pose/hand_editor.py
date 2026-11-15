"""Local zoomable 21-point hand editor. It performs no inference or uploads."""
import argparse,copy
from pathlib import Path
import tkinter as tk
from tkinter import ttk,messagebox
from PIL import ImageTk,Image
import hand_edit_data as edit

COLORS=('#dca7ff','#58d6ef','#75dfb6','#ffce73','#ff8ea8')
class Editor:
 def __init__(self,root,data,side,output,character_seed=None):
  self.root=root;self.data=data;self.output=Path(output);self.state=edit.initialize(data,side,character_seed);self.original=copy.deepcopy(self.state);self.history=[];self.selected=8;self.drag=False;self.scale=1.;self.offset=(0.,0.);self.image_key=None;self.photo=None;self.solved=None
  self.source=edit.worker.prepare_image(self.state['image']);root.title('Screenshot Pose · '+('左手' if side=='L' else '右手')+' 21 点校正');root.geometry(f'{min(1250,root.winfo_screenwidth()-80)}x{min(880,root.winfo_screenheight()-90)}');root.minsize(min(850,root.winfo_screenwidth()-60),min(650,root.winfo_screenheight()-60))
  style=ttk.Style(root);style.theme_use('clam');style.configure('.',background='#1c2635',foreground='#e7edf5',fieldbackground='#28374c');style.configure('TButton',padding=9);style.map('TButton',background=[('active','#36526c')]);style.configure('TCombobox',fieldbackground='#28374c',foreground='#e7edf5');style.map('TCombobox',fieldbackground=[('readonly','#28374c')],selectbackground=[('readonly','#28374c')],selectforeground=[('readonly','#e7edf5')])
  sidebar=ttk.Frame(root,width=320);sidebar.pack(side='right',fill='y');sidebar.pack_propagate(False)
  scroller=tk.Canvas(sidebar,background='#1c2635',highlightthickness=0);scrollbar=ttk.Scrollbar(sidebar,orient='vertical',command=scroller.yview);scrollbar.pack(side='right',fill='y');scroller.pack(side='left',fill='both',expand=True);scroller.configure(yscrollcommand=scrollbar.set)
  panel=ttk.Frame(scroller,padding=16);window=scroller.create_window((0,0),window=panel,anchor='nw')
  panel.bind('<Configure>',lambda event:scroller.configure(scrollregion=scroller.bbox('all')));scroller.bind('<Configure>',lambda event:scroller.itemconfigure(window,width=event.width))
  def scroll_panel(event):
   widget=event.widget
   while widget is not None:
    if widget in (panel,scroller):scroller.yview_scroll(-1 if event.delta>0 else 1,'units');return
    widget=getattr(widget,'master',None)
  root.bind_all('<MouseWheel>',scroll_panel,add='+')
  ttk.Label(panel,text='手指关节点校正',font=('',14,'bold')).pack(anchor='w')
  ttk.Label(panel,text=Path(self.state['image']).name,wraplength=265,foreground='#96adc8').pack(anchor='w',pady=4)
  ttk.Label(panel,text='拖动指尖、指节；滚轮缩放，中键平移。\n左右指人物自身左右，手腕位置固定。',wraplength=265).pack(anchor='w',pady=12)
  self.choice=tk.StringVar(value=edit.NAMES[self.selected]);combo=ttk.Combobox(panel,textvariable=self.choice,values=edit.NAMES,state='readonly');combo.pack(fill='x');combo.bind('<<ComboboxSelected>>',self.choose)
  self.place=tk.BooleanVar(value=False);ttk.Checkbutton(panel,text='点击放置选中点（重合时使用）',variable=self.place).pack(anchor='w',pady=10)
  ttk.Label(panel,text='选中点前后深度（米，正值远离相机）',wraplength=265).pack(anchor='w',pady=(10,4));self.depth=tk.StringVar();ttk.Entry(panel,textvariable=self.depth).pack(fill='x');ttk.Button(panel,text='更新此点深度',command=self.set_depth).pack(fill='x',pady=5)
  ttk.Button(panel,text='聚焦手部',command=self.focus).pack(fill='x',pady=(12,4));ttk.Button(panel,text='显示完整图片',command=self.full).pack(fill='x')
  ttk.Button(panel,text='撤销上一步 · Ctrl+Z',command=self.undo).pack(fill='x',pady=(12,4));ttk.Button(panel,text='重置为打开时的标记',command=self.reset).pack(fill='x')
  ttk.Button(panel,text='检查骨长与求解残差',command=self.preview_solution).pack(fill='x',pady=5)
  ttk.Button(panel,text='保存并仅更新修改的手指',command=lambda:self.save(True)).pack(fill='x',pady=(18,4));ttk.Button(panel,text='仅保存数据',command=lambda:self.save(False)).pack(fill='x');ttk.Button(panel,text='取消 · Esc',command=root.destroy).pack(fill='x',pady=5)
  self.status=tk.StringVar();ttk.Label(panel,textvariable=self.status,wraplength=265,foreground='#8fdbcc').pack(anchor='w',pady=12)
  self.canvas=tk.Canvas(root,background='#101722',highlightthickness=0);self.canvas.pack(side='left',fill='both',expand=True)
  self.canvas.bind('<Configure>',lambda e:self.focus() if self.photo is None else self.draw());self.canvas.bind('<ButtonPress-1>',self.down);self.canvas.bind('<B1-Motion>',self.motion);self.canvas.bind('<ButtonRelease-1>',lambda e:setattr(self,'drag',False));self.canvas.bind('<MouseWheel>',self.zoom);self.canvas.bind('<ButtonPress-2>',self.pan_start);self.canvas.bind('<B2-Motion>',self.pan)
  root.bind('<Control-z>',lambda e:self.undo());root.bind('<Escape>',lambda e:root.destroy());self.update_depth();root.after(100,self.focus)
 def remember(self):self.history=(self.history+[copy.deepcopy(self.state)])[-60:]
 def update_depth(self):
  self.solved=None
  self.depth.set(f"{self.state['depth'][self.selected]:.4f}");self.status.set(self.state['initialization']+'\n'+self.state['depth_source']+'\n已修改 '+str(len(self.state['changed']))+' 个点；保存时保留指骨长度。')
 def choose(self,event=None):self.selected=edit.NAMES.index(self.choice.get());self.update_depth();self.draw()
 def xy(self,index):
  p=self.state['xy'][index];return (self.offset[0]+p[0]*self.source.width*self.scale,self.offset[1]+p[1]*self.source.height*self.scale)
 def focus(self):
  import numpy as np
  p=np.asarray(self.state['xy'])*self.source.size;lo=p.min(0);hi=p.max(0);size=max(float(max(hi-lo))*1.7,60);center=(lo+hi)/2;w=max(self.canvas.winfo_width(),10);h=max(self.canvas.winfo_height(),10)
  self.scale=min(w,h)/size;self.offset=(w/2-center[0]*self.scale,h/2-center[1]*self.scale);self.draw()
 def full(self):
  w,h=self.canvas.winfo_width(),self.canvas.winfo_height();self.scale=min(w/self.source.width,h/self.source.height);self.offset=((w-self.source.width*self.scale)/2,(h-self.source.height*self.scale)/2);self.draw()
 def draw(self):
  c=self.canvas;w=max(c.winfo_width(),1);h=max(c.winfo_height(),1);key=(w,h,self.scale,self.offset)
  if key!=self.image_key:
   view=self.source.transform((w,h),Image.Transform.AFFINE,(1/self.scale,0,-self.offset[0]/self.scale,0,1/self.scale,-self.offset[1]/self.scale),resample=Image.Resampling.BICUBIC)
   self.photo=ImageTk.PhotoImage(view);self.image_key=key
  c.delete('all');c.create_image(0,0,image=self.photo,anchor='nw')
  for a,b in edit.EDGES:c.create_line(*self.xy(a),*self.xy(b),fill=COLORS[(b-1)//4],width=2)
  for i in range(21):
   x,y=self.xy(i);r=7 if i==self.selected else 4;color='#ffe961' if i==self.selected else COLORS[(i-1)//4] if i else '#f4f6fa'
   c.create_oval(x-r,y-r,x+r,y+r,fill=color,outline='#101722');c.create_text(x+9,y-9,text=str(i),fill=color,anchor='w')
  c.create_text(14,14,text=edit.NAMES[self.selected]+'  |  手腕固定，二维位置与深度分别校正',fill='#ffe961',anchor='nw',font=('',11))
  if self.solved is not None:
   for i in self.state['changed']:
    px,py=self.solved[i];x=self.offset[0]+px*self.source.width*self.scale;y=self.offset[1]+py*self.source.height*self.scale
    c.create_line(x-7,y,x+7,y,fill='white',width=2);c.create_line(x,y-7,x,y+7,fill='white',width=2)
 def down(self,event):
  if not self.place.get():
   closest=min(range(21),key=lambda i:(self.xy(i)[0]-event.x)**2+(self.xy(i)[1]-event.y)**2)
   if (self.xy(closest)[0]-event.x)**2+(self.xy(closest)[1]-event.y)**2>24**2:return
   self.selected=closest;self.choice.set(edit.NAMES[closest]);self.update_depth()
  if self.selected==0:self.status.set('手腕固定；如需移动手腕，请使用身体关节校正。');return
  self.remember();self.drag=True;self.motion(event)
 def motion(self,event):
  if not self.drag:return
  x=(event.x-self.offset[0])/self.source.width/self.scale;y=(event.y-self.offset[1])/self.source.height/self.scale
  edit.move(self.state,self.selected,max(0,min(1,x)),max(0,min(1,y)));self.update_depth();self.draw()
 def zoom(self,event):
  factor=1.15 if event.delta>0 else 1/1.15;new=max(.05,min(50,self.scale*factor));ratio=new/self.scale
  self.offset=(event.x-(event.x-self.offset[0])*ratio,event.y-(event.y-self.offset[1])*ratio);self.scale=new;self.draw()
 def pan_start(self,event):self.pan_origin=(event.x,event.y,self.offset)
 def pan(self,event):
  x,y,offset=self.pan_origin;self.offset=(offset[0]+event.x-x,offset[1]+event.y-y);self.draw()
 def set_depth(self):
  try:
   value=float(self.depth.get());previous=copy.deepcopy(self.state);x,y=self.state['xy'][self.selected];edit.move(self.state,self.selected,x,y,value);self.history=(self.history+[previous])[-60:];self.update_depth();self.draw()
  except ValueError as exc:messagebox.showerror('深度无效',str(exc))
 def undo(self):
  if self.history:self.state=self.history.pop();self.update_depth();self.draw()
 def reset(self):self.remember();self.state=copy.deepcopy(self.original);self.update_depth();self.draw()
 def preview_solution(self):
  try:
   import numpy as np
   result=edit.document(self.data,self.state);p=np.asarray(result['orientation']['hands'][self.state['side']]['world']);p=(p-p[0])@np.asarray(self.state['view_rotation']);seed=np.asarray(self.state['world_seed']);self.solved=np.asarray(self.state['original_xy'])+(p-seed)[:,[0,2]]*np.array([1.,-1.])/self.state['meters_per_pixel']/[self.source.width,self.source.height]
   self.status.set(f"源点求解残差约 {result['hand_edit']['fit_residual_m']*1000:.1f} 毫米。白色十字为保留骨长后的估计位置；二维不能唯一确定深度。");self.draw()
  except ValueError as exc:self.status.set(str(exc))
 def save(self,apply):
  try:edit.save(self.data,self.state,self.output,apply);self.root.destroy()
  except Exception as exc:messagebox.showerror('保存失败',str(exc))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--side',choices=['L','R'],required=True);p.add_argument('--seed');args=p.parse_args()
 if Path(args.input).resolve()==Path(args.output).resolve():raise ValueError('不能覆盖输入 JSON')
 import json
 seed=json.loads(Path(args.seed).read_text(encoding='utf-8')) if args.seed else None
 root=tk.Tk();Editor(root,edit.pose_data.read_pose(args.input),args.side,args.output,seed);root.mainloop()
