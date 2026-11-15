"""Blender UI for a role-independent, portable image + prediction library."""
import importlib,json,math,uuid
from pathlib import Path
import bpy,bpy.utils.previews
from bpy.types import Operator,PropertyGroup
from bpy.props import StringProperty
from . import library_store
OPTIONS=('coordinate_mode','depth','mirror','yaw','full_source_rotation','auto_full_quality','auto_depth_select','fit_projection','fit_model','fit_torso','whole_body','whole_body_root_limit','whole_body_rotation_limit','head_consensus','foot_pose','orient_head','orient_hand_l','orient_hand_r','head_yaw','head_pitch','head_roll','palm_flip_l','palm_flip_r','palm_roll_l','palm_roll_r','hand_left','hand_right','hand_grip_l','hand_grip_r','finger_joints','auto_finger_curl')
OPTIONS=('preserve_source_pose',)+OPTIONS
ICONS=None
def addon():return importlib.import_module(__package__)
def root():
 prefs=addon().preferences();configured=prefs.pose_library_dir if prefs else ''
 return Path(bpy.path.abspath(configured)).resolve() if configured else Path.home()/'Documents'/'ScreenshotPoseLibrary'
def clear_icons():
 global ICONS
 if ICONS is not None:bpy.utils.previews.remove(ICONS);ICONS=None
def icon(path):
 global ICONS
 if ICONS is None:ICONS=bpy.utils.previews.new()
 path=Path(path);key=str(path.resolve())+str(path.stat().st_mtime_ns)
 if key not in ICONS:ICONS.load(key,str(path),'IMAGE')
 return ICONS[key].icon_id
def selected(s):
 if not 0<=s.library_index<len(s.library_items):raise ValueError('请先刷新并选择一个姿势')
 key=s.library_items[s.library_index].entry_id
 return key,library_store.metadata(library_store.entry(root(),key))
def refresh(s):
 rows,bad=library_store.list_entries(root(),s.library_search);previous=s.library_items[s.library_index].entry_id if 0<=s.library_index<len(s.library_items) else None
 s.library_items.clear()
 for meta in rows:
  item=s.library_items.add();item.name=meta['name'];item.entry_id=meta['id'];item.tags=meta.get('tags','')
 s.library_index=next((i for i,r in enumerate(rows) if r['id']==previous),0)
 s.library_status=f'{len(rows)} 个姿势'+(f'；{bad} 个条目损坏或缺素材，已跳过' if bad else '')
 clear_icons()

def save(s,name,tags='',notes=''):
 a=addon()
 if a._JOB or s.candidate_state:raise ValueError('请先完成识别或确认当前预览，再保存姿势')
 data=a.pose_tools.effective_document(s,a.pose_data.read_pose(bpy.path.abspath(s.json_path)))
 image=Path(data['image'].get('path',''))
 if not image.is_absolute():image=Path(bpy.path.abspath(s.json_path)).parent/image
 guides={o['sp_joint']:list(o.location) for o in a.preview.marker_objects(s)}
 options={n:getattr(s,n) for n in OPTIONS}
 if s.chosen_candidate and s.chosen_signature==a.pose_tools.signature(s):
  options.update(full_source_rotation=json.loads(s.chosen_candidate)['kind']=='FULL',auto_full_quality=False,auto_depth_select=False)
 meta=library_store.save(root(),data,image,name,tags,notes,options,guides,bpy.path.abspath(s.overlay_path) if s.overlay_path else None)
 s.library_search='';refresh(s);s.library_index=next(i for i,r in enumerate(s.library_items) if r.entry_id==meta['id'])
 s.status='已加入本地姿势库：'+meta['name'];return meta

def load(context,s,key):
 a=addon()
 if a._JOB or s.candidate_state:raise ValueError('请先完成任务或结束候选预览')
 # Validate all saved settings and local guides before changing the scene.
 meta=library_store.metadata(library_store.entry(root(),key));options={k:v for k,v in meta.get('options',{}).items() if k in OPTIONS}
 for name,value in options.items():
  prop=s.bl_rna.properties[name]
  if prop.type=='ENUM' and value not in prop.enum_items.keys():raise ValueError('条目包含不兼容的选项：'+name)
  if prop.type=='BOOLEAN' and not isinstance(value,bool):raise ValueError('条目开关类型无效')
  if prop.type=='FLOAT' and (isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value)):raise ValueError('条目数值无效')
 for point in meta.get('guides',{}).values():
  if not isinstance(point,list) or len(point)!=3 or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in point):raise ValueError('条目关节点坐标无效')
 destination=a.runtime_dir()/'library_work'/uuid.uuid4().hex
 meta,data=library_store.materialize(root(),key,destination)
 a.frame_tools.accept(s)
 for name,value in options.items():setattr(s,name,value)
 s.ground_feet=False;s.contact_enabled=False
 s.json_path=str(destination/'pose.json');s.image_path=data['image']['path'];s.overlay_path=str(destination/'overlay.png') if (destination/'overlay.png').is_file() else ''
 a.preview.make_preview(context,s,data)
 for obj in a.preview.marker_objects(s):
  if obj['sp_joint'] in meta.get('guides',{}):obj.location=meta['guides'][obj['sp_joint']]
 a.preview.refresh_lines(context,s)
 s.status='已载入：'+meta['name']+'；请选择目标骨架并应用。接地 / 场景接触需重新设置'
 return data

def detected(scene):
 s=scene.screenshot_pose
 s.status+='；可加入本地姿势库'
 if s.library_prompt and scene==bpy.context.scene:
  try:bpy.ops.screenshot_pose.library_save('INVOKE_DEFAULT')
  except RuntimeError:s.library_status='可点击“加入本地姿势库”保存识别结果'

class SPLibraryItem(PropertyGroup):
 entry_id:StringProperty();tags:StringProperty()

class SP_OT_library_save(Operator):
 bl_idname='screenshot_pose.library_save';bl_label='加入本地姿势库'
 title:StringProperty(name='名称',default='未命名姿势',maxlen=100)
 tags:StringProperty(name='标签',maxlen=300)
 notes:StringProperty(name='备注',maxlen=2000)
 def invoke(self,context,event):
  try:
   data=addon().pose_data.read_pose(bpy.path.abspath(context.scene.screenshot_pose.json_path));self.title=Path(data['image'].get('original_path','姿势')).stem[:100]
  except Exception:self.title='未命名姿势'
  return context.window_manager.invoke_props_dialog(self,width=460)
 def draw(self,context):
  self.layout.prop(self,'title');self.layout.prop(self,'tags');self.layout.prop(self,'notes')
  self.layout.label(text='保存截图、通用姿势数据和插件内关节点校正')
  self.layout.label(text='不保存角色骨骼手动编辑或场景接触物体')
 def execute(self,context):
  try:save(context.scene.screenshot_pose,self.title,self.tags,self.notes)
  except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}
  return {'FINISHED'}

class SP_OT_library_action(Operator):
 bl_idname='screenshot_pose.library_action';bl_label='本地姿势库'
 action:StringProperty(default='REFRESH')
 title:StringProperty(name='名称',maxlen=100);tags:StringProperty(name='标签',maxlen=300);notes:StringProperty(name='备注',maxlen=2000)
 def invoke(self,context,event):
  try:
   if self.action=='RENAME':
    _,meta=selected(context.scene.screenshot_pose);self.title=meta['name'];self.tags=meta.get('tags','');self.notes=meta.get('notes','')
    return context.window_manager.invoke_props_dialog(self,width=460)
  except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}
  return self.execute(context)
 def draw(self,context):
  for name in ('title','tags','notes'):self.layout.prop(self,name)
 def execute(self,context):
  s=context.scene.screenshot_pose
  try:
   if self.action=='REFRESH':refresh(s)
   elif self.action=='OPEN':root().mkdir(parents=True,exist_ok=True);bpy.ops.wm.path_open(filepath=str(root()))
   else:
    key,meta=selected(s)
    if self.action=='LOAD':load(context,s,key)
    elif self.action=='RENAME':library_store.rename(root(),key,self.title,self.tags,self.notes);refresh(s)
    elif self.action=='TRASH':library_store.trash(root(),key);refresh(s);s.status='已移入库内 .trash 回收目录，可手动恢复文件夹'
    else:raise ValueError('未知姿势库操作')
  except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}
  return {'FINISHED'}
OPERATORS=(SP_OT_library_save,SP_OT_library_action)
