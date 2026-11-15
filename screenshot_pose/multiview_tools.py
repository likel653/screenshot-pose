"""Optional multi-view jobs. Never apply a generated hypothesis automatically."""
import importlib,json,uuid
from pathlib import Path
import bpy
from bpy.props import StringProperty,BoolProperty,IntProperty,CollectionProperty
from bpy.types import Operator,PropertyGroup
from bpy_extras.io_utils import ImportHelper

def addon():return importlib.import_module(__package__)

class SPReferenceImage(PropertyGroup):
 file_path:StringProperty(name='辅助图片',subtype='FILE_PATH')
 enabled:BoolProperty(name='使用',default=True)
 person_index:IntProperty(name='人物序号',default=1,min=1,max=4)

class SP_OT_reference_add(Operator,ImportHelper):
 bl_idname='screenshot_pose.reference_add';bl_label='添加辅助视角'
 filter_glob:StringProperty(default='*.png;*.jpg;*.jpeg;*.webp;*.bmp',options={'HIDDEN'})
 files:CollectionProperty(type=bpy.types.OperatorFileListElement)
 directory:StringProperty(subtype='DIR_PATH')
 def execute(self,context):
  a=addon();s=context.scene.screenshot_pose
  if a._JOB or s.candidate_state:return {'CANCELLED'}
  paths=[str(Path(self.directory)/f.name) for f in self.files] or [self.filepath]
  existing={str(Path(bpy.path.abspath(r.file_path)).resolve()).casefold() for r in s.reference_images if r.file_path}
  for path in paths:
   if len(s.reference_images)>=3:break
   if str(Path(path).resolve()).casefold() in existing:continue
   row=s.reference_images.add();row.file_path=path;existing.add(str(Path(path).resolve()).casefold())
  return {'FINISHED'}

class SP_OT_reference_remove(Operator):
 bl_idname='screenshot_pose.reference_remove';bl_label='移除辅助图片'
 index:IntProperty(default=0)
 def execute(self,context):
  s=context.scene.screenshot_pose
  if addon()._JOB or s.candidate_state:return {'CANCELLED'}
  if 0<=self.index<len(s.reference_images):s.reference_images.remove(self.index)
  return {'FINISHED'}

def complete(scene,source,output,overlay):
 a=addon();s=scene.screenshot_pose
 data=a.pose_data.read_pose(output);report=data['multiview']
 s.multiview_report=str(Path(output).with_suffix('.report.json'));s.multiview_output=str(output)
 if not report['accepted']:
  s.status='辅助图未通过一致性筛选，保留主图；请查看多图报告';return
 if str(Path(bpy.path.abspath(s.json_path)).resolve())!=str(Path(source).resolve()):
  s.status='多图候选已保存；当前主图已改变，因此未切换数据';return
 # Existing guides, character channels and confirmed primary choices remain intact.
 old=s.chosen_candidate if s.chosen_signature==a.pose_tools.signature(s) else ''
 s.json_path=str(output);s.overlay_path=str(overlay)
 s.pose_candidates.clear();s.candidate_signature=''
 if not old:s.chosen_candidate='';s.chosen_signature=''
 if old:
  try:
   choice=json.loads(old)
   if choice.get('source','').startswith('sam3d_multiview'):s.chosen_candidate='';s.chosen_signature=''
   else:s.chosen_signature=a.pose_tools.signature(s)
  except (ValueError,KeyError):s.chosen_candidate='';s.chosen_signature=''
 s.status=f"已生成多图候选（采用 {report['accepted']} 张辅助图）；到姿势候选预览比较，尚未应用"

class SP_OT_multiview(Operator):
 bl_idname='screenshot_pose.multiview';bl_label='生成多图辅助候选'
 def execute(self,context):
  try:
   a=addon();s=context.scene.screenshot_pose
   if not s.multiview_enabled:raise ValueError('请先启用多图辅助')
   if a._JOB or s.candidate_state:raise ValueError('请先结束任务或候选预览')
   source=Path(bpy.path.abspath(s.json_path)).resolve();data=a.pose_data.read_pose(source)
   full=data.get('full_skeleton',{})
   if full.get('source')!='sam3d_mhr':raise ValueError('请先用 SAM 三维识别主图')
   entries=[dict(path=str(Path(bpy.path.abspath(r.file_path)).resolve()),person_index=r.person_index) for r in s.reference_images if r.enabled]
   if not 1<=len(entries)<=3 or any(not Path(e['path']).is_file() for e in entries):raise ValueError('请选择 1 至 3 张存在的辅助图片')
   folder=a.runtime_dir()/'multiview'/uuid.uuid4().hex;folder.mkdir(parents=True)
   request=folder/'references.json';request.write_text(json.dumps(entries,ensure_ascii=False),encoding='utf-8')
   output=folder/'pose.json';overlay=folder/'overlay.png'
   command=[a.runtime_python('SAM3D'),a.HERE/'worker_multiview.py','--input',source,'--references',request,'--model',a.runtime_dir()/'pose_landmarker_full.task','--output',output,'--overlay',overlay,'--strength',str(s.multiview_strength),'--confidence',str(s.confidence)]+a.sam_root_args()
   a.start_job(context.scene,command,'分析辅助视角',lambda scene:complete(scene,source,output,overlay))
  except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}
  return {'FINISHED'}

class SP_OT_multiview_report(Operator):
 bl_idname='screenshot_pose.multiview_report';bl_label='多图辅助报告'
 text:StringProperty(options={'HIDDEN'})
 def invoke(self,context,event):
  try:
   path=Path(bpy.path.abspath(context.scene.screenshot_pose.multiview_report))
   if path.stat().st_size>262144:raise ValueError('报告过大')
   self.text=path.read_text(encoding='utf-8')
  except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}
  return context.window_manager.invoke_props_dialog(self,width=560)
 def draw(self,context):
  from .ui import wrap
  report=json.loads(self.text);l=self.layout
  wrap(l,context,f"采用 {report['accepted']} 张，筛除 {len(report['rejected'])} 张；缓存命中 {report.get('cache_hits',0)} 张")
  for row in report.get('references',[]):wrap(l,context,Path(row['image']).name+f"：已采用，估计视角差 {row['view_angle_deg']:.0f}°")
  for row in report['rejected']:wrap(l,context,Path(row['image']).name+'：'+row['reason'],'INFO')
  wrap(l,context,'结果为未标定视角的估计辅助候选，不保证优于主图。头部与手掌局部识别沿用主图。')
 def execute(self,context):return {'FINISHED'}

OPERATORS=(SP_OT_reference_add,SP_OT_reference_remove,SP_OT_multiview,SP_OT_multiview_report)
