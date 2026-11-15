"""Independent hand reference selection and non-destructive refinement jobs."""
import importlib,json,uuid
from pathlib import Path
import bpy
from bpy.props import EnumProperty,StringProperty
from bpy.types import Operator

def addon():return importlib.import_module(__package__)
def screenshot_status(s,data,path):
 report=data.get('automatic_hands')
 if not report:return
 adopted=sum(bool(v.get('adopted')) for v in report.get('hands',{}).values())
 for side,info in report.get('hands',{}).items():
  if info.get('adopted'):setattr(s,'orient_hand_'+side.lower(),True)
 if report['status']=='off':return
 s.status+='；'+str(report.get('backend','GPU'))+' 手部采用 '+str(adopted)+'/2'
 if report['status']=='unavailable':s.status+='（模型未就绪，保留已有观测）'
 elif adopted<2:s.status+='；未采用的手保留已有观测，请复核'
 file=Path(path).with_suffix('.report.json')
 if file.is_file():s.hand_refine_report=str(file)
_COMPARISONS={}
def comparison_choices(s):
 try:
  path=Path(bpy.path.abspath(s.hand_refine_report));st=path.stat();key=(str(path.resolve()),st.st_mtime_ns,st.st_size)
  if key not in _COMPARISONS:
   if len(_COMPARISONS)>8:_COMPARISONS.clear()
   if st.st_size>1048576:return []
   report=json.loads(path.read_text(encoding='utf8'));_COMPARISONS[key]=report.get('candidates',{})
  return [b for b,p in _COMPARISONS[key].items() if b in ('WILOR','HAMER','HAMER_TEXTURE') and Path(p).is_file()]
 except (OSError,ValueError,TypeError):return []

def complete(scene,source,signature,output):
 a=addon();s=scene.screenshot_pose;d=a.pose_data.read_pose(output);report=d['hand_refinement'];s.hand_refine_report=str(Path(output).with_suffix('.report.json'))
 if hasattr(s,'hand3d_comparison'):s.hand3d_comparison=report.get('comparison_html','')
 if report.get('backend') in ('COMPARE','COMPARE_TEXTURE'):
  counts='；'.join(b+' 采用 '+str(sum(bool(v.get('adopted')) for v in rows.values()))+'/'+str(len(rows)) for b,rows in report.get('candidate_reports',{}).items())
  s.status='双模型对比已保存，尚未应用；'+counts
  return
 count=sum(v['adopted'] for v in report['hands'].values())
 if not count:s.status='未获得新的可靠手部观测，保留主图；仍可指定手势约束并局部应用';return
 if str(Path(bpy.path.abspath(s.json_path)).resolve())!=str(Path(source).resolve()) or a.pose_tools.signature(s)!=signature:
  s.status='手部结果已保存，当前主图已变化，因此未切换数据';return
 choice=s.chosen_candidate if s.chosen_signature==signature else ''
 s.json_path=str(output);s.pose_candidates.clear();s.candidate_signature='';s.chosen_candidate=choice;s.chosen_signature=a.pose_tools.signature(s) if choice else ''
 for side,info in report['hands'].items():
  if info['adopted']:setattr(s,'orient_hand_'+side.lower(),True)
 s.status=f'已更新 {count} 只手的观测，尚未应用；请查看报告，再仅更新对应手掌'

class SP_OT_hands_refine(Operator):
 bl_idname='screenshot_pose.hands_refine';bl_label='精修手部参考'
 side:EnumProperty(items=[('L','左手',''),('R','右手',''),('BOTH','双手','')],default='BOTH')
 def execute(self,context):
  try:
   a=addon();s=context.scene.screenshot_pose
   if a._JOB or s.candidate_state:raise ValueError('请先结束任务或候选预览')
   source=Path(bpy.path.abspath(s.json_path)).resolve();a.pose_data.read_pose(source);signature=a.pose_tools.signature(s)
   if s.hand3d_backend in ('HAMER_TEXTURE','COMPARE_TEXTURE'):
    from .hand3d import readiness
    state=readiness(a.runtime_dir(),'HAMER_TEXTURE')
    if not state['ready']:raise ValueError('增强版权重未准备：请先下载作者 .ckpt 并点击核验增强权重')
   selections={}
   for side in ('L','R') if self.side=='BOTH' else (self.side,):
    path=getattr(s,'hand_reference_'+side.lower());path=str(Path(bpy.path.abspath(path)).resolve()) if path else ''
    if path and not Path(path).is_file():raise ValueError('手部参考图不存在')
    selections[side]=dict(image=path,person_index=getattr(s,'hand_person_'+side.lower()),gesture=getattr(s,'hand_left' if side=='L' else 'hand_right'),gpu_2d=s.hand_gpu_2d,backend=s.hand3d_backend,precision=s.hand3d_precision)
   folder=a.runtime_dir()/'hands'/uuid.uuid4().hex;folder.mkdir(parents=True)
   request=folder/'request.json';request.write_text(json.dumps(selections,ensure_ascii=False),encoding='utf-8');output=folder/'pose.json'
   python=a.runtime_python('SAM3D')
   if not python.is_file():python=a.runtime_python()
   command=[python,a.HERE/'worker_hands.py','--input',source,'--request',request,'--model',a.runtime_dir()/'pose_landmarker_full.task','--output',output]+a.sam_root_args()
   a.start_job(context.scene,command,'精修手部参考',lambda scene:complete(scene,source,signature,output))
  except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}
  return {'FINISHED'}

class SP_OT_hands_report(Operator):
 bl_idname='screenshot_pose.hands_report';bl_label='手部识别与约束报告'
 payload:StringProperty(options={'HIDDEN'})
 def invoke(self,context,event):
  try:
   path=Path(bpy.path.abspath(context.scene.screenshot_pose.hand_refine_report))
   if path.stat().st_size>262144:raise ValueError('报告过大')
   self.payload=path.read_text(encoding='utf-8')
  except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}
  return context.window_manager.invoke_props_dialog(self,width=560)
 def draw(self,context):
  from .ui import wrap
  report=json.loads(self.payload)
  for side,info in report['hands'].items():
   box=self.layout.box();box.label(text='左手' if side=='L' else '右手')
   wrap(box,context,'参考：'+Path(info['reference']).name)
   if info['adopted']:
    if info.get('native_rotations'):wrap(box,context,'已保留 MANO 原生指骨旋转，可还原指骨扭转。')
    sampling=info.get('input_sampling')
    if sampling:wrap(box,context,'手部输入：%d × %d；%s'%(*sampling['used_size'],'已核验原图' if sampling['original_used'] else '缓存图 / 原图无额外像素'))
    crops=info.get('crop_selection')
    if crops:
     selected=crops['candidates'][crops['chosen']]
     wrap(box,context,'已比较 %d 个裁剪，采用 %.1f 倍裁剪；仅在独立二维指节证据充分时更换。'%(len(crops['candidates']),selected['expansion']))
    if info.get('backend'):wrap(box,context,'三维手部模型：'+info['backend'])
    wrap(box,context,'已采用 '+('完整 21 点' if info['has_21_points'] else '仅掌向，无完整手指点'))
    if info.get('fallback'):wrap(box,context,'局部模型未检出，沿用该视角已有观测；不是新的手指识别。','INFO')
    fit=info.get('fit')
    if fit:wrap(box,context,'二维微调：'+('已采用' if fit['accepted'] else '未改善，保留初始点'))
   else:wrap(box,context,'保留主图：'+info['reason'],'INFO')
   if info.get('gpu_2d_warning'):wrap(box,context,'GPU 二维：'+info['gpu_2d_warning'],'INFO')
   elif info.get('gpu_2d'):wrap(box,context,'GPU 已采用二维定位；它不提供三维深度。')
   wrap(box,context,'指定约束：'+{'NONE':'自动 / 保持','V':'V 手势','GRIP':'握持近似','FIST':'握拳','OPEN':'张开'}.get(info['gesture'],'未知'))
  wrap(self.layout,context,'手势约束是人为指定或几何规则建议，不等于模型识别成功。握持不模拟手机接触。')
 def execute(self,context):return {'FINISHED'}

class SP_OT_setup_hand_gpu(Operator):
 bl_idname='screenshot_pose.setup_hand_gpu';bl_label='准备手部 GPU 二维模型（首次联网）'
 def execute(self,context):
  try:
   a=addon();python=a.runtime_python('SAM3D')
   if not python.is_file():raise ValueError('请先准备 GPU 识别环境')
   a.start_job(context.scene,[python,a.HERE/'setup_hand_gpu.py','--runtime',a.runtime_dir()],'准备可选手部 GPU 模型',lambda scene:setattr(scene.screenshot_pose,'status','GPU 手部二维模型已准备；可选启用，三维深度仍需其他来源'))
  except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}
  return {'FINISHED'}

class SP_OT_hand3d_setup(Operator):
 bl_idname='screenshot_pose.hand3d_setup';bl_label='准备 WiLoR 与 HaMeR（首次联网）'
 def execute(self,context):
  try:
   a=addon();s=context.scene.screenshot_pose
   root=Path(a.LOCAL_RUNTIME.get('sam_root',str(a.HERE.parent)))
   prefs=a.preferences()
   if prefs and prefs.sam_root:root=Path(bpy.path.abspath(prefs.sam_root))
   command=[a.setup_python(),a.HERE/'setup_hand3d.py','--root',root,'--runtime',a.runtime_dir()]
   if s.hand3d_mano:command+=['--mano',bpy.path.abspath(s.hand3d_mano)]
   def done(scene):
    from .hand3d import readiness
    states=[readiness(a.runtime_dir(),b) for b in ('WILOR','HAMER')]
    scene.screenshot_pose.status='WiLoR 与 HaMeR 文件准备完成；'+('可进行真实三维推理' if all(v['ready'] for v in states) else '仍缺少：'+ '；'.join(v['backend']+' '+','.join(v['missing']) for v in states if not v['ready']))
   a.start_job(context.scene,command,'准备两套三维手部模型',done)
  except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}
  return {'FINISHED'}

class SP_OT_hand3d_comparison(Operator):
 bl_idname='screenshot_pose.hand3d_comparison';bl_label='查看三维手部模型对比'
 def execute(self,context):
  path=Path(bpy.path.abspath(context.scene.screenshot_pose.hand3d_comparison))
  if not path.is_file():self.report({'ERROR'},'尚无对比报告');return {'CANCELLED'}
  bpy.ops.wm.url_open(url=path.resolve().as_uri());return {'FINISHED'}

class SP_OT_hand3d_adopt(Operator):
 bl_idname='screenshot_pose.hand3d_adopt';bl_label='选择三维手部结果'
 backend:EnumProperty(items=[('WILOR','WiLoR',''),('HAMER','HaMeR',''),('HAMER_TEXTURE','纹理监督增强 HaMeR','')])
 side:EnumProperty(items=[('CURRENT','界面所选手',''),('BOTH','双手',''),('L','左手',''),('R','右手','')],default='CURRENT')
 component:EnumProperty(items=[('CURRENT','界面所选范围',''),('ALL','整手',''),('PALM','仅掌向',''),('FINGERS','仅手指','')],default='CURRENT')
 def execute(self,context):
  try:
   from .worker_hand3d import digest
   from . import hand_selection
   a=addon();s=context.scene.screenshot_pose
   if a._JOB or s.candidate_state:raise ValueError('请先结束任务或候选预览')
   report=json.loads(Path(bpy.path.abspath(s.hand_refine_report)).read_text(encoding='utf8'))
   source=Path(report['primary_json']).resolve();current=Path(bpy.path.abspath(s.json_path)).resolve()
   allowed={source}|{Path(v).resolve() for v in report['candidates'].values()}
   current_data=a.pose_data.read_pose(current);lineage=current_data.get('hand_component_lineage',{})
   lineage_ok=lineage.get('primary_json')==str(source) and lineage.get('primary_sha256')==report['primary_sha256']
   if (current not in allowed and not lineage_ok) or digest(source)!=report['primary_sha256']:raise ValueError('主图数据已变化，请重新生成对比')
   # Even a composed document must retain the exact body/reference data.
   base=a.pose_data.read_pose(source)
   if any(current_data.get(k)!=base.get(k) for k in ('image','landmarks','world_landmarks','full_skeleton','body_candidates')):raise ValueError('当前身体或参考图已改变，请重新生成对比')
   data=a.pose_data.read_pose(Path(report['candidates'][self.backend]))
   side=s.hand_adopt_side if self.side=='CURRENT' else self.side;component=s.hand_adopt_component if self.component=='CURRENT' else self.component
   merged,rows=hand_selection.select(current_data,data,('L','R') if side=='BOTH' else (side,),component,self.backend)
   merged['hand_refinement']=dict(backend='COMPONENT_SELECTION',hands=rows,body_preserved=True)
   merged['hand_component_lineage']=dict(primary_json=str(source),primary_sha256=report['primary_sha256'])
   folder=a.runtime_dir()/'hands'/('selection_'+uuid.uuid4().hex);folder.mkdir(parents=True)
   output=folder/'pose.json'
   output.write_text(json.dumps(merged,ensure_ascii=False,indent=2),encoding='utf8')
   comparison=s.hand3d_comparison;report_path=s.hand_refine_report
   a.hand_tools.complete(context.scene,current,a.pose_tools.signature(s),output)
   s.hand3d_comparison=comparison;s.hand_refine_report=report_path
  except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}
  return {'FINISHED'}

class SP_OT_setup_hand_texture(Operator):
 bl_idname='screenshot_pose.setup_hand_texture';bl_label='核验增强权重'
 def execute(self,context):
  try:
   a=addon();s=context.scene.screenshot_pose
   checkpoint=Path(bpy.path.abspath(s.hand_texture_checkpoint))
   if not s.hand_texture_checkpoint or not checkpoint.is_file():raise ValueError('请先选择作者提供的 texture-supervised HaMeR .ckpt 文件')
   config=json.loads((a.runtime_dir()/'hand3d/config.json').read_text(encoding='utf8'))
   a.start_job(context.scene,[config['python'],a.HERE/'setup_hand_texture.py','--runtime',a.runtime_dir(),'--checkpoint',checkpoint],
       '核验纹理监督 HaMeR 权重',lambda scene:setattr(scene.screenshot_pose,'status','增强权重架构已核验；可独立识别或与原版对比'))
  except Exception as e:self.report({'ERROR'},str(e));return {'CANCELLED'}
  return {'FINISHED'}

OPERATORS=(SP_OT_hands_refine,SP_OT_hands_report,SP_OT_setup_hand_gpu,SP_OT_hand3d_setup,SP_OT_hand3d_comparison,SP_OT_hand3d_adopt,SP_OT_setup_hand_texture)
