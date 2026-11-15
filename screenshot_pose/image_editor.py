"""Standalone local Tk image landmark editor. No browser, network, or bpy."""
import argparse
import copy
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from pose_data import read_pose, diagnostics
from worker import atomic_write, draw_overlay, save_png

NAMES={0:'鼻尖',7:'左耳',8:'右耳',11:'左肩',12:'右肩',13:'左肘',14:'右肘',
       15:'左手腕',16:'右手腕',19:'左手方向',20:'右手方向',23:'左髋',24:'右髋',
       25:'左膝',26:'右膝',27:'左踝',28:'右踝',29:'左脚跟',30:'右脚跟',31:'左脚尖',32:'右脚尖'}
EDGES=[(11,12),(11,13),(13,15),(15,19),(12,14),(14,16),(16,20),
       (11,23),(12,24),(23,24),(23,25),(25,27),(27,31),(27,29),(29,31),(24,26),(26,28),(28,32),(28,30),(30,32),(7,8)]


def move_point(data,index,x,y,carry_hand=True):
    points=data['landmarks']; old=points[index]
    dx,dy=x-old['x'],y-old['y']
    ids=[index]
    if carry_hand and index in (15,16):
        ids=[index,index+2,index+4,index+6]
    for i in ids:
        points[i]['x']+=dx; points[i]['y']+=dy
        points[i]['visibility']=1.;points[i]['presence']=1.
    data['edited_indices']=sorted(set(data.get('edited_indices',[])+ids))
    data['suggested_mode']='HYBRID'
    details=data.get('orientation')
    if details:
        if index in (0,7,8):details.pop('head',None)
        if index in (15,17,19,21):details.get('hands',{}).pop('L',None)
        if index in (16,18,20,22):details.get('hands',{}).pop('R',None)
        if index in (27,29,31):details.get('feet',{}).pop('L',None)
        if index in (28,30,32):details.get('feet',{}).pop('R',None)


def save_document(data,output):
    output=Path(output)
    content=json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)
    # The input JSON remains intact, even if writing or preview generation fails.
    with Image.open(data['image']['path']) as image:
        overlay=draw_overlay(image.convert('RGB'),data['landmarks'],.5,data.get('person_index',0))
        save_png(overlay,output.with_suffix('.png'))
    atomic_write(output,lambda tmp:tmp.write_text(content,encoding='utf-8'))


class Editor:
    def __init__(self,root,data,output):
        self.root,self.data,self.output=root,copy.deepcopy(data),Path(output)
        self.original=copy.deepcopy(data);self.history=[];self.selected=15;self.drag=False
        self.source=Image.open(data['image']['path']).convert('RGB')
        self.scale=1.;self.offset=(0,0);self.tkimage=None
        root.title('Screenshot Pose · 图片关节点校正');root.geometry('1120x900')
        panel=ttk.Frame(root,padding=12);panel.pack(side='right',fill='y')
        ttk.Label(panel,text='先修正轮廓，再调整深度',font=('',12,'bold')).pack(anchor='w')
        ttk.Label(panel,text='左 / 右指人物自身左右\n拖动关节；重合时用下拉框选择\n选择后也可在图上直接点击新位置').pack(anchor='w',pady=12)
        self.choice=tk.StringVar(value=NAMES[self.selected])
        combo=ttk.Combobox(panel,textvariable=self.choice,values=list(NAMES.values()),state='readonly',width=24)
        combo.pack(fill='x');combo.bind('<<ComboboxSelected>>',self.select_name)
        self.place=tk.BooleanVar(value=False)
        ttk.Checkbutton(panel,text='点击放置选中关节（重合时推荐）',variable=self.place).pack(anchor='w',pady=10)
        self.carry=tk.BooleanVar(value=True)
        ttk.Checkbutton(panel,text='移动手腕时带动手部方向点',variable=self.carry).pack(anchor='w')
        ttk.Label(panel,text='前后深度（估计米，小值靠前）').pack(anchor='w',pady=(18,3))
        self.depth=tk.StringVar();ttk.Entry(panel,textvariable=self.depth).pack(fill='x')
        ttk.Button(panel,text='更新选中关节的深度',command=self.set_depth).pack(fill='x',pady=5)
        ttk.Button(panel,text='撤销上一步',command=self.undo).pack(fill='x',pady=(20,5))
        ttk.Button(panel,text='重置为打开时的结果',command=self.reset).pack(fill='x')
        ttk.Button(panel,text='保存并返回 Blender',command=self.save).pack(fill='x',pady=(25,5))
        ttk.Button(panel,text='取消（不保存）',command=root.destroy).pack(fill='x')
        self.status=tk.StringVar(value='\n'.join(diagnostics(data)) or '请核对肩、肘、手腕、膝、踝位置')
        ttk.Label(panel,textvariable=self.status,wraplength=260,foreground='#b26919').pack(fill='x',pady=18)
        self.canvas=tk.Canvas(root,background='#232830',highlightthickness=0)
        self.canvas.pack(side='left',fill='both',expand=True)
        self.canvas.bind('<Configure>',self.resize)
        self.canvas.bind('<ButtonPress-1>',self.down)
        self.canvas.bind('<B1-Motion>',self.motion)
        self.canvas.bind('<ButtonRelease-1>',lambda event:setattr(self,'drag',False))
        root.bind('<Control-z>',lambda event:self.undo())
        self.update_depth()

    def remember(self):
        self.history.append(copy.deepcopy(self.data));self.history=self.history[-60:]

    def update_depth(self):
        self.depth.set(f"{self.data['world_landmarks'][self.selected]['z']:.4f}")

    def select_name(self,event=None):
        self.selected=next(i for i,n in NAMES.items() if n==self.choice.get())
        self.update_depth();self.draw()

    def resize(self,event=None):
        w,h=max(self.canvas.winfo_width(),1),max(self.canvas.winfo_height(),1)
        self.scale=min(w/self.source.width,h/self.source.height)
        size=(max(1,int(self.source.width*self.scale)),max(1,int(self.source.height*self.scale)))
        self.offset=((w-size[0])/2,(h-size[1])/2)
        self.tkimage=ImageTk.PhotoImage(self.source.resize(size,Image.Resampling.LANCZOS))
        self.draw()

    def xy(self,i):
        p=self.data['landmarks'][i]
        return (self.offset[0]+p['x']*self.source.width*self.scale,
                self.offset[1]+p['y']*self.source.height*self.scale)

    def draw(self):
        c=self.canvas;c.delete('all')
        if self.tkimage:c.create_image(*self.offset,image=self.tkimage,anchor='nw')
        for a,b in EDGES:
            c.create_line(*self.xy(a),*self.xy(b),fill='#33bdff' if a%2 else '#ff6389',width=2)
        for i,name in NAMES.items():
            x,y=self.xy(i);r=7 if i==self.selected else 4
            color='#ffee53' if i==self.selected else '#33bdff' if i%2 else '#ff6389'
            c.create_oval(x-r,y-r,x+r,y+r,fill=color,outline='black',width=1)
            if i==self.selected:c.create_text(x+12,y-12,text=name,fill='#ffee53',anchor='w')

    def down(self,event):
        if not self.place.get():
            closest=min(NAMES,key=lambda i:(self.xy(i)[0]-event.x)**2+(self.xy(i)[1]-event.y)**2)
            if (self.xy(closest)[0]-event.x)**2+(self.xy(closest)[1]-event.y)**2>24**2:return
            self.selected=closest;self.choice.set(NAMES[closest]);self.update_depth()
        self.remember();self.drag=True;self.motion(event)

    def motion(self,event):
        if not self.drag:return
        x=(event.x-self.offset[0])/(self.source.width*self.scale)
        y=(event.y-self.offset[1])/(self.source.height*self.scale)
        move_point(self.data,self.selected,max(0,min(1,x)),max(0,min(1,y)),self.carry.get())
        self.draw()

    def set_depth(self):
        try:
            value=float(self.depth.get())
            if not -5<=value<=5:raise ValueError()
            self.remember();self.data['world_landmarks'][self.selected]['z']=value
            self.data['suggested_mode']='HYBRID'
        except ValueError:messagebox.showerror('深度无效','请输入 -5 到 5 之间的数值。')

    def undo(self):
        if self.history:self.data=self.history.pop();self.update_depth();self.draw()

    def reset(self):
        self.remember();self.data=copy.deepcopy(self.original);self.update_depth();self.draw()

    def save(self):
        try:
            self.data['suggested_mode']='HYBRID';save_document(self.data,self.output)
            self.root.destroy()
        except Exception as exc:messagebox.showerror('保存失败',str(exc))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    root=tk.Tk();Editor(root,read_pose(args.input),args.output);root.mainloop()
