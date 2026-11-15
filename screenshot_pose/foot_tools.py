"""Foot-only, cancellable orientation auditions and local inference jobs."""
import copy,importlib,json,uuid
from pathlib import Path
import bpy
from bpy.props import EnumProperty
from bpy.types import Operator
from mathutils import Matrix,Vector
from . import foot_data
def addon():return importlib.import_module(__package__)

def restore(s,state,animated=False):
 a=addon();snap=copy.deepcopy(state['snapshot'])
 if not animated:snap['action']=None
 a.retarget.restore_pose(s.target,snap)

def clean():
 for obj in list(bpy.data.objects):
  if obj.get('sp_foot_diagnostic'):
   data=obj.data;bpy.data.objects.remove(obj,do_unlink=True)
   if data and data.users==0:bpy.data.curves.remove(data)

def diagram(s):
 a=addon();rig,mapping,data=a.pose_tools.check(s);clean();side=s.foot_side
 foot=rig.pose.bones[mapping['foot.'+side]];cal=a.foot_pose.sole_calibration(rig,mapping,s.shoe_mesh.name if s.shoe_mesh else '')
 if side not in cal:raise ValueError('缺少角色鞋底参考')
 fit=a.pose_tools.last_result(s).get('fit');C=Matrix(fit['camera']['rotation']).transposed() if fit else Matrix.Identity(3)
 fd=data.get('orientation',{}).get('feet',{}).get(side)
 if not fd:raise ValueError('当前数据没有识别脚向')
 frame,source=a.foot_pose.foot_source.frame_for(data,side,fd)
 if source=='sam3d_native_sole_rotation':
  fd=dict(fd,frame=frame.tolist());fd.pop('points',None)
 actual=rig.matrix_world.to_quaternion().to_matrix()@foot.matrix.to_3x3().normalized()@foot.bone.matrix_local.to_3x3().normalized().transposed()@Matrix(cal[side]['frame'])
 origin=rig.matrix_world@foot.head
 for name,frame,color in [('识别脚向',C@Matrix(fd['frame']),(.1,.5,1,1)),('角色鞋底',actual,(1,.45,.05,1))]:
  curve=bpy.data.curves.new(name,'CURVE');curve.dimensions='3D';curve.bevel_depth=.002
  for axis in (1,2):
   spline=curve.splines.new('POLY');spline.points.add(1)
   for p,v in zip(spline.points,(origin,origin+frame.col[axis]*.18)):p.co=(*v,1)
  obj=bpy.data.objects.new(name,curve);bpy.context.scene.collection.objects.link(obj);obj.color=color;obj.show_in_front=True;obj.hide_render=True;obj['sp_foot_diagnostic']=True
  mat=bpy.data.materials.get('SP Foot '+name) or bpy.data.materials.new('SP Foot '+name);mat.diffuse_color=color;curve.materials.append(mat)
  if name=='识别脚向' and fd.get('points'):
   points=fd['points'];wrist=Vector(points['ankle'])
   spline=curve.splines.new('POLY');spline.points.add(4)
   for point,key in zip(spline.points,('ankle','heel','big_toe','small_toe','heel')):point.co=(*(origin+C@(Vector(points[key])-wrist)),1)
  elif name=='角色鞋底':
   spline=curve.splines.new('POLY');spline.points.add(1)
   for point,key in zip(spline.points,('heel','forefoot')):point.co=(*(rig.matrix_world@foot.matrix@Vector(cal[side][key])),1)
 s.status='脚向诊断快照：蓝色为识别，橙色为角色鞋底；长轴为鞋尖 / 足底法向，修改姿势后请刷新'

def session(s):
 a=addon()
 if s.candidate_state:
  state=json.loads(s.candidate_state)
  if state.get('kind')!='FOOT_DIRECTION':raise ValueError('请先结束其他预览')
  return state
 rig,mapping,data=a.pose_tools.check(s);fd=data.get('orientation',{}).get('feet',{}).get(s.foot_side)
 if not fd:raise ValueError('没有当前脚的识别朝向，请先识别或局部复核')
 effective,_=a.foot_pose.foot_source.frame_for(data,s.foot_side,fd);fd=dict(fd,frame=effective.tolist())
 if not a.orientation._safe(rig.pose.bones.get(mapping.get('foot.'+s.foot_side,'')),rig):raise ValueError('脚部 FK 映射不可写')
 state=dict(kind='FOOT_DIRECTION',target=rig.name,side=s.foot_side,frame=bpy.context.scene.frame_current,signature=a.pose_tools.signature(s),snapshot=a.retarget.snapshot_pose(rig),backup=rig.get('screenshot_pose_backup'),last_result=s.last_result,guides={o.name:list(o.location) for o in a.preview.marker_objects(s)},options=foot_data.candidates(fd['frame']),index=-1,nla={t.name:t.mute for t in rig.animation_data.nla_tracks} if rig.animation_data else {})
 fit=a.pose_tools.last_result(s).get('fit');camera=Matrix(fit['camera']['rotation']) if fit else Matrix.Identity(3)
 world=camera.transposed()@Matrix(fd['frame']);forward=world.col[1].copy();forward.z=0
 if forward.length>1e-6:
  forward.normalize();level=Matrix((forward.cross(Vector((0,0,1))),forward,Vector((0,0,1)))).transposed()
  state['options'].insert(1,dict(name='水平鞋底（人工假设）',frame=[list(row) for row in camera@level]))
 if rig.animation_data:
  if rig.animation_data.action:rig.animation_data.action.use_fake_user=True
  rig.animation_data.action=None
  for t in rig.animation_data.nla_tracks:t.mute=True
 restore(s,state);s.candidate_state=json.dumps(state,ensure_ascii=False);return state

def apply_option(s,state,index):
 a=addon();a.pose_tools.validate_session(s,state)
 if state['side']!=s.foot_side:raise ValueError('左右脚已切换，请取消后重新预览')
 restore(s,state);data=a.pose_data.read_pose(bpy.path.abspath(s.json_path));data=copy.deepcopy(data)
 spec=state['options'][index];side=state['side'];data['orientation']['feet'][side].update(frame=spec['frame'],confidence=1.,source='user_foot_direction_choice',confidence_method='user_choice_not_model_probability')
 mapping=a.pose_tools.mapping_for(s);proxy=a.pose_tools.SettingsProxy(s,ground_feet=False,contact_enabled=False,strength=1.)
 report=a.foot_pose.apply(s.target,mapping,proxy,data,a.pose_tools.last_result(s).get('fit'),sides=(side,))
 if not report['touched']:raise ValueError('没有找到可用的鞋底参考，未修改角色')
 if s.contact_enabled and getattr(s,'support_'+side.lower())!='FREE':
  contact=a.contact_targets.apply(s.target,mapping,s,data,'FOOT_'+side,report['calibration']);report['touched']+=contact['touched'];report['warnings']+=contact['warnings']
 elif s.ground_feet and side in a.contact_solver.grounded_sides(s):
  contact=a.pose_tools.local_contact(s.target,mapping,s,side,report);report['touched']+=contact['touched'];report['warnings']+=contact['warnings']
 state['index']=index;state['report']=report;s.candidate_state=json.dumps(state,ensure_ascii=False)
 s.status='临时脚向：'+spec['name']+'；这是假设候选，不是新增模型预测'
 return data,report

class SP_OT_foot_preview(Operator):
 bl_idname='screenshot_pose.foot_preview';bl_label='脚向候选';bl_options={'REGISTER','UNDO'}
 action:EnumProperty(items=[(k,k,'') for k in ('PREVIEW','NEXT','CONFIRM','CANCEL','DIAGRAM','HIDE','AUTO')],default='PREVIEW')
 def execute(self,context):
  a=addon();s=context.scene.screenshot_pose;out=None;created=None;state=None
  try:
   if a._JOB:raise ValueError('请先等待当前识别任务结束')
   if self.action=='AUTO':
    if s.candidate_state:raise ValueError('请先结束当前预览')
    rig,mapping,data=a.pose_tools.check(s);data=copy.deepcopy(data);side=s.foot_side
    native=a.foot_pose.foot_source.native_frame(data,side)
    if native is None:raise ValueError('当前文件没有 SAM 原生脚骨旋转，请重新使用 SAM 识别')
    data.setdefault('orientation',{}).setdefault('feet',{}).setdefault(side,{}).update(frame=native.tolist(),source='sam3d_native_sole_rotation',confidence=.65,confidence_method='uncalibrated_native_rotation')
    signature=a.pose_tools.signature(s);out=a.runtime_dir()/'feet'/('auto_'+uuid.uuid4().hex+'.json');out.parent.mkdir(parents=True,exist_ok=True)
    from .worker import write_json
    write_json(out,data);a.hand_preview.adopt_path(s,out,signature);s.status='已恢复此脚 SAM 自动脚向；点击应用到角色生效'
    return {'FINISHED'}
   if self.action=='DIAGRAM':diagram(s);return {'FINISHED'}
   if self.action=='HIDE':clean();return {'FINISHED'}
   if self.action=='CANCEL':
    if not s.candidate_state or json.loads(s.candidate_state).get('kind')!='FOOT_DIRECTION':raise ValueError('没有脚向预览')
    a.pose_tools.cancel(s);return {'FINISHED'}
   state=session(s)
   if self.action=='CONFIRM' and state['index']<0:raise ValueError('请先预览脚向')
   index=(state['index']+1)%len(state['options']) if self.action=='NEXT' else max(state['index'],0)
   data,report=apply_option(s,state,index)
   if self.action=='CONFIRM':
    out=a.runtime_dir()/'feet'/('choice_'+uuid.uuid4().hex+'.json');out.parent.mkdir(parents=True,exist_ok=True)
    from .worker import write_json
    write_json(out,data);solved=a.retarget.snapshot_pose(s.target)
    a.retarget.restore_pose(s.target,state['snapshot'])
    if s.keyframe:created=a.pose_tools.copy_action(s.target,state['snapshot'])
    anim=s.target.animation_data;action=anim.action if anim else None;slot=getattr(anim,'action_slot',None)
    solved.update(action=action.name if action else None,action_slot_handle=slot.handle if slot else None,action_fake_user=bool(action.use_fake_user) if action else False)
    a.retarget.restore_pose(s.target,solved,update=not s.keyframe)
    if s.keyframe:a.contact_solver.key_touched(s.target,report['touched'],context.scene.frame_current);a.retarget._update(s.target)
    for t in anim.nla_tracks if anim else ():
     if t.name in state['nla']:t.mute=state['nla'][t.name]
    a.hand_preview.adopt_path(s,out,state['signature']);s.candidate_state='';s.status='已确认脚向；保留源文件，后续应用沿用选择'
   return {'FINISHED'}
  except Exception as exc:
   if state and s.candidate_state:a.pose_tools.cancel(s,True)
   if created:
    created.use_fake_user=False
    if created.users==0:bpy.data.actions.remove(created)
   if out and out.exists():out.unlink()
   self.report({'ERROR'},str(exc));return {'CANCELLED'}

def complete(scene,source,signature,out):
 a=addon();s=scene.screenshot_pose;data=a.pose_data.read_pose(out)
 s.foot_refine_report=json.dumps(data['foot_refinement'],ensure_ascii=False)
 if a.pose_tools.signature(s)!=signature or Path(bpy.path.abspath(s.json_path)).resolve()!=Path(source).resolve():s.status='主图已变化，脚部结果已保存但未切换';return
 count=sum(x['adopted'] for x in data['foot_refinement'].values())
 if count:a.hand_preview.adopt_path(s,out,signature)
 s.status=f'脚部局部复核通过 {count} 只脚；应用时检查角色鞋向'

class SP_OT_foot_refine(Operator):
 bl_idname='screenshot_pose.foot_refine';bl_label='局部复核双脚'
 def execute(self,context):
  try:
   a=addon();s=context.scene.screenshot_pose
   if a._JOB or s.candidate_state:raise ValueError('请先结束任务或预览')
   source=Path(bpy.path.abspath(s.json_path)).resolve();a.pose_data.read_pose(source);signature=a.pose_tools.signature(s)
   out=a.runtime_dir()/'feet'/uuid.uuid4().hex/'pose.json';out.parent.mkdir(parents=True)
   command=[a.runtime_python('SAM3D'),a.HERE/'worker_feet.py','--source',source,'--model',a.runtime_dir()/'rtmw_pose.onnx','--output',out]
   a.start_job(context.scene,command,'两尺度脚部局部复核',lambda scene:complete(scene,source,signature,out))
  except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}
  return {'FINISHED'}
OPERATORS=(SP_OT_foot_preview,SP_OT_foot_refine)
