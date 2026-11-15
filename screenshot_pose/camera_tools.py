"""Optional screenshot calibration jobs; observation adoption never moves the rig."""
import importlib,json,uuid
from pathlib import Path
import bpy
from bpy.types import Operator

def addon():return importlib.import_module(__package__)

class SP_OT_setup_camera(Operator):
 bl_idname='screenshot_pose.setup_camera';bl_label='准备相机标定模型（首次联网）'
 def execute(self,context):
  try:
   a=addon();root=Path(a.LOCAL_RUNTIME.get('sam_root',str(a.HERE.parent)))
   a.start_job(context.scene,[a.runtime_python('SAM3D'),a.HERE/'setup_camera.py','--root',root,'--runtime',a.runtime_dir()],
       '准备 AnyCalib 相机模型',lambda scene:setattr(scene.screenshot_pose,'status','相机标定模型已准备；可标定当前截图'))
  except Exception as e:self.report({'ERROR'},str(e));return {'CANCELLED'}
  return {'FINISHED'}

class SP_OT_calibrate_camera(Operator):
 bl_idname='screenshot_pose.calibrate_camera';bl_label='标定相机用于透视拟合'
 bl_description='估计截图内参；默认保留身体与手部观测，可选按新内参重新识别身体。完成后需手动应用'
 def execute(self,context):
  try:
   a=addon();s=context.scene.screenshot_pose
   if a._JOB or s.candidate_state:raise ValueError('请先结束当前任务或预览')
   source=Path(bpy.path.abspath(s.json_path)).resolve();data=a.pose_data.read_pose(source);signature=a.pose_tools.signature(s)
   if not (a.runtime_dir()/'camera/config.json').is_file():raise ValueError('请先准备相机标定模型')
   if data.get('edited_indices') or (a.preview.root_for(s) and a.pose_tools.edited(s,data)):raise ValueError('当前有人工关节校正，请先保存到姿势库，再从原图标定')
   folder=a.runtime_dir()/'camera_jobs'/uuid.uuid4().hex;folder.mkdir(parents=True);out=folder/'pose.json'
   def complete(scene):
    state=scene.screenshot_pose
    if str(Path(bpy.path.abspath(state.json_path)).resolve())!=str(source) or a.pose_tools.signature(state)!=signature:
     state.status='标定结果已保存；当前参考已变化，未切换数据';return
    result=a.pose_data.read_pose(out);state.json_path=str(out);state.chosen_candidate='';state.chosen_signature='';state.pose_candidates.clear()
    state.coordinate_mode='HYBRID';state.fit_projection=True;state.fit_model='PERSPECTIVE';state.last_result=''
    if bpy.context.scene==scene:a.preview.make_preview(bpy.context,state,result)
    state.status='已标定相机'+('并重新识别身体' if result['camera_refinement']['body_reinferred_with_calibration'] else '，保留原身体观测')+'；点击应用查看透视拟合（试验）'
   command=[a.runtime_python('SAM3D'),a.HERE/'worker_camera_job.py','--input',source,
        '--model',a.runtime_dir()/'pose_landmarker_full.task','--output',out]+a.sam_root_args()
   if not s.camera_refresh_body:command.append('--no-refresh-body')
   a.start_job(context.scene,command,'标定相机'+('并重新识别身体' if s.camera_refresh_body else '（保留身体观测）'),complete)
  except Exception as e:self.report({'ERROR'},str(e));return {'CANCELLED'}
  return {'FINISHED'}

class SP_OT_camera_before(Operator):
 bl_idname='screenshot_pose.camera_before';bl_label='切回标定前观测'
 def execute(self,context):
  try:
   import hashlib
   a=addon();s=context.scene.screenshot_pose
   if a._JOB or s.candidate_state:raise ValueError('请先结束任务或预览')
   data=a.pose_data.read_pose(bpy.path.abspath(s.json_path));previous=data.get('camera_refinement',{})
   source=Path(previous.get('primary_json',''))
   if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest()!=previous.get('primary_sha256'):
    raise ValueError('标定前的观测文件不存在或已被修改')
   old=a.pose_data.read_pose(source);s.json_path=str(source);s.fit_model='ORTHO';s.chosen_candidate='';s.chosen_signature='';s.last_result=''
   s.pose_candidates.clear();a.preview.make_preview(context,s,old);s.status='已切回标定前观测；点击应用查看原有近似'
  except Exception as e:self.report({'ERROR'},str(e));return {'CANCELLED'}
  return {'FINISHED'}

class SP_OT_setup_head3d(Operator):
 bl_idname='screenshot_pose.setup_head3d';bl_label='准备独立头向 GPU 模型（可选，约95MB）'
 def execute(self,context):
  try:
   a=addon()
   def done(scene):scene.screenshot_pose.head_gpu=True;scene.screenshot_pose.status='独立头向模型已准备；点击只重新识别头部候选，完成后手动应用'
   a.start_job(context.scene,[a.runtime_python(),a.HERE/'setup_head3d.py','--runtime',a.runtime_dir()],'准备 6DRepNet360 候选模型',done)
  except Exception as e:self.report({'ERROR'},str(e));return {'CANCELLED'}
  return {'FINISHED'}

OPERATORS=(SP_OT_setup_camera,SP_OT_calibrate_camera,SP_OT_camera_before,SP_OT_setup_head3d)
