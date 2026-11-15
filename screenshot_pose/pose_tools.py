"""Scoped pose updates and reversible, key-free candidate audition."""
import hashlib,importlib,json,math
from pathlib import Path
from types import SimpleNamespace
import bpy
from mathutils import Matrix,Vector
from bpy.props import StringProperty,EnumProperty,IntProperty
from bpy.types import Operator,PropertyGroup
from . import candidate_data

def addon():return importlib.import_module(__package__)
SCOPES=[('HEAD','头部','更新头部与颈部'),('HAND_L','左手掌与手指','固定前臂，只更新手掌与手指'),('HAND_R','右手掌与手指','固定前臂，只更新手掌与手指'),('ARM_L','左臂（含手）','更新上臂、前臂、手掌与手指'),('ARM_R','右臂（含手）','更新上臂、前臂、手掌与手指'),('LEG_L','左腿（含脚）','更新一侧腿脚，骨盆与另一条腿不动'),('LEG_R','右腿（含脚）','更新一侧腿脚，骨盆与另一条腿不动')]
class SettingsProxy:
 def __init__(self,base,**values):self.base=base;self.values=values
 def __getattr__(self,name):return self.values[name] if name in self.values else getattr(self.base,name)
def mapping_for(s):return {r.role:r.bone for r in s.mapping if r.enabled and r.bone}
def check(s):
 a=addon();rig=s.target
 a.video_review.guard()
 if a._JOB:raise ValueError('请先等待或取消识别任务')
 if not rig or rig.type!='ARMATURE':raise ValueError('请先选择角色骨架')
 if rig.mode=='EDIT':raise ValueError('请先退出骨架编辑模式')
 if rig.library and not rig.override_library:raise ValueError('链接骨架只读，请先创建 Library Override')
 root=a.preview.root_for(s)
 if not root or root.get('sp_target','')!=rig.name:raise ValueError('请为当前角色重建关节点')
 if not s.mapping:a.set_mapping(s)
 mapping=mapping_for(s)
 if len(mapping.values())!=len(set(mapping.values())):raise ValueError('骨骼映射存在重复，请先修正')
 return rig,mapping,a.pose_data.read_pose(bpy.path.abspath(s.json_path))
def last_result(s):
 try:return json.loads(s.last_result) if s.last_result else {}
 except ValueError:return {}
def copy_action(rig,snapshot):
 a=addon();old=rig.animation_data.action if rig.animation_data else None
 a.animation_preview.prepare_copy(rig)
 if old:old.use_fake_user=True
 action=old.copy() if old else bpy.data.actions.new(rig.name+' Local Pose')
 action.name=(old.name if old else rig.name)+' · 局部姿势'
 a.retarget._assign_action(rig,action,snapshot.get('action_slot_handle'))
 return action
def restore_backup(rig,value):
 if value is None:
  if 'screenshot_pose_backup' in rig:del rig['screenshot_pose_backup']
 else:rig['screenshot_pose_backup']=value
def local_contact(rig,mapping,s,side,foot_state):
 a=addon();bones=[rig.pose.bones[mapping[r+'.'+side]] for r in ('thigh','shin','foot')]
 upper,lower,foot=bones;world=lambda b:rig.matrix_world@b.head
 hip,knee,ankle=map(world,bones);lengths=((knee-hip).length,(ankle-knee).length)
 target=ankle.copy();anchors=foot_state.get('sole_anchors',{}).get(side)
 target.z=s.floor_height-min((rig.matrix_world.to_3x3()@foot.matrix.to_3x3()@Vector(p)).z for p in anchors) if anchors else s.floor_height+sum(lengths)*s.ankle_clearance
 rotation=foot.matrix.to_quaternion().copy()
 for _ in range(3):
  bend,end,_=a.contact_solver.solve_two_bone(world(upper),target,knee,*lengths)
  a.contact_solver._aim(rig,upper,bend-world(upper));a.contact_solver._aim(rig,lower,end-world(lower))
  a.retarget._set_rotation(foot,a.retarget._to_local_rotation(foot,rotation));a.retarget._update(rig)
 error=(world(foot)-target).length
 return dict(error=error,touched=[b.name for b in bones],warnings=['脚部目标在固定骨盆下不可达，请调整关节或使用全身接地'] if error>.002 else [])

def apply_local(context,s,*,scope=None,fingers=None,update_palm=True,data_override=None,preview_session=None,keyframe=None,force_palm=False,palm_reset=False):
 a=addon()
 if s.candidate_state and preview_session is None:raise ValueError('请先确认或取消候选预览')
 if preview_session is not None:validate_session(s,preview_session)
 rig,mapping,data=check(s);data=data_override if data_override is not None else data;scope=scope or s.local_scope;side=scope[-1] if scope!='HEAD' else None
 if scope not in dict((r[0],r[1]) for r in SCOPES):raise ValueError('局部更新范围无效')
 use_keyframe=s.keyframe if keyframe is None else keyframe
 roles={'neck','head'} if scope=='HEAD' else {r+'.'+side for r in (('upper_arm','forearm','hand') if scope.startswith('ARM') else ('thigh','shin','foot') if scope.startswith('LEG') else ('hand',))}
 if any(r not in mapping for r in roles):raise ValueError('当前局部缺少完整骨骼映射')
 if s.strength==0:s.status='应用强度为零，未改变角色';return dict(scope=scope,touched=[],warnings=[])
 before=a.retarget.snapshot_pose(rig);backup=rig.get('screenshot_pose_backup');previous_result=s.last_result
 report=dict(scope=scope,touched=[],properties=[],warnings=[],dependent_bones=[],global_solve=False);created=None
 fit=last_result(s).get('fit')
 try:
  if not backup:rig['screenshot_pose_backup']=json.dumps(before,ensure_ascii=False)
  # Use the last fitted camera, but never re-optimize the whole character.
  camera=Matrix(fit['camera']['rotation']).transposed() if fit else Matrix.Identity(3)
  transform=rig.matrix_world.inverted_safe().to_3x3()@camera
  joints={o['sp_joint']:transform@o.location for o in a.preview.marker_objects(s)}
  detail_map=dict(mapping)
  if not scope.startswith('HAND'):
   body=a.retarget.apply_pose(rig,mapping,joints,strength=s.strength,rigify_fk=s.switch_fk,roles=roles)
   report['touched'].extend(mapping[r] for r in body['applied']);report['warnings'].extend(body['warnings'])
   report['touched'].extend(body.get('touched',[]))
   if s.switch_fk:
    report['properties'].extend((b.name,p) for b,p,v in a.retarget._fk_switches(rig,{r:mapping[r] for r in roles}))
  proxy=SettingsProxy(s,hand_side=side,finger_scope=fingers,forearm_share=0. if scope.startswith('HAND') else s.forearm_share,orient_head=scope=='HEAD' and s.orient_head,
                      orient_hand_l=side=='L' and update_palm and not scope.startswith('LEG') and (s.orient_hand_l or force_palm),
                      orient_hand_r=side=='R' and update_palm and not scope.startswith('LEG') and (s.orient_hand_r or force_palm))
  if palm_reset and side:proxy.values.update({'palm_flip_'+side.lower():False,'palm_roll_'+side.lower():0.,'hand_candidate_'+side.lower():'1'})
  if not scope.startswith('LEG'):
   detail=a.orientation.apply(rig,detail_map,proxy,data,fit)
   for key in ('touched','properties','warnings','dependent_bones'):report[key].extend(detail.get(key,[]))
   if side and ((s.finger_joints and s.auto_finger_curl) or a.hand_presets.manual_fingers(data,side)):
    fingers=a.finger_pose.apply(rig,mapping,proxy,data,a.orientation.calibration(rig,mapping,s))
    for key in ('touched','properties','warnings','dependent_bones'):report[key].extend(fingers.get(key,[]))
   if side:
    preset=a.hand_presets.selected(s,data,side)
    if preset!='NONE':
     exclude=a.hand_presets.manual_fingers(data,side)|(set(a.finger_pose.FINGERS)-set(fingers) if fingers is not None else set())
     hand=a.hand_presets.apply(rig,side,preset,s.strength,s.hand_spread,getattr(s,'hand_grip_'+side.lower(),.75),exclude)
     for key in ('touched','properties','warnings','dependent_bones'):report[key].extend(hand.get(key,[]))
  else:
   foot=a.foot_pose.apply(rig,mapping,s,data,fit,sides=(side,)) if s.foot_pose else {}
   for key in ('touched','warnings'):report[key].extend(foot.get(key,[]))
   if s.ground_feet and side in a.contact_solver.grounded_sides(s):
    contact=local_contact(rig,mapping,s,side,foot);report['contact']=contact
    report['touched'].extend(contact['touched']);report['warnings'].extend(contact['warnings'])
   if s.contact_enabled:
    contact=a.contact_targets.apply(rig,mapping,s,data,'FOOT_'+side)
    report['contact']=contact
    for key in ('touched','properties','warnings'):report[key].extend(contact[key])
  report['touched']=sorted(set(report['touched']));report['properties']=list(dict.fromkeys(tuple(p) for p in report['properties']))
  allowed=set(report['touched'])|{n for n,p in report['properties']}|set(report['dependent_bones'])
  after=a.retarget.snapshot_pose(rig)
  # Enforce the promise before writing keys: untouched controls retain channels.
  changed=[]
  for n,state in before['bones'].items():
   now=after['bones'][n]
   if n not in allowed and (state['properties']!=now['properties'] or max(abs(state['matrix_basis'][i][j]-now['matrix_basis'][i][j]) for i in range(4) for j in range(4))>2e-6):changed.append(n)
  if changed:raise ValueError('绑定联动修改了其他控制骨，已撤回局部更新：'+', '.join(changed[:3]))
  if use_keyframe and allowed:
   created=copy_action(rig,before);a.contact_solver.key_touched(rig,report['touched'],context.scene.frame_current,report['properties'])
   switch_paths=a.retarget.location_switch_paths(rig,{r:mapping[r] for r in roles})
   for curve in a.animation_preview.curves(created):
    if curve.data_path in switch_paths:
     for point in curve.keyframe_points:point.interpolation='CONSTANT'
  result=last_result(s);result['local_update']=report;result['observation_signature']=signature(s);result['review_context']=a.review_tools.context_stamp(s);s.last_result=json.dumps(result,ensure_ascii=False)
  a.preview.refresh_lines(context,s);s.status='已仅更新'+dict((x[0],x[1]) for x in SCOPES)[scope]+('并打关键帧' if use_keyframe else '')
  if report['warnings']:s.status+='；请查看结果中的局部更新提示'
  return report
 except Exception:
  a.retarget.restore_pose(rig,before);restore_backup(rig,backup);s.last_result=previous_result
  if created and created.users==0:bpy.data.actions.remove(created)
  raise

def signature(s):return hashlib.sha256(Path(bpy.path.abspath(s.json_path)).read_bytes()).hexdigest()
def effective_document(s,data):
 if s.chosen_candidate and s.chosen_signature==signature(s):return candidate_data.document(data,json.loads(s.chosen_candidate))
 if s.chosen_candidate:s.chosen_candidate='';s.chosen_signature=''
 return data
def apply_chosen(context,s,data):
 a=addon();spec=json.loads(s.chosen_candidate);document=candidate_data.document(data,spec)
 flags={n:getattr(s,n) for n in ('full_source_rotation','auto_full_quality','auto_depth_select')}
 try:
  s.full_source_rotation=spec['kind']=='FULL';s.auto_full_quality=False;s.auto_depth_select=False
  result=a._apply_current_once(context,s,data_override=document)
 finally:
  for n,v in flags.items():setattr(s,n,v)
 result['candidate_choice']=dict(id=spec['id'],name=spec['name'],source=spec['source'],preview=False)
 s.last_result=json.dumps(result,ensure_ascii=False);return result
def edited(s,data):
 a=addon();canonical,_=a.pose_data.to_joints(data,s.coordinate_mode,s.depth,s.mirror,s.yaw,s.threshold)
 return any((o.location-Vector(canonical[o['sp_joint']])).length>1e-5 for o in a.preview.marker_objects(s))
def collect(s):
 rig,mapping,data=check(s)
 if s.candidate_state:raise ValueError('请先确认或取消当前预览')
 specs=candidate_data.available(data,edited(s,effective_document(s,data)) or s.mirror or abs(s.yaw)>1e-6);s.pose_candidates.clear()
 for spec in specs:
  item=s.pose_candidates.add();item.name=spec['name'];item.spec=json.dumps(spec);item.score='尚未预览'
 s.candidate_index=0;s.candidate_signature=signature(s)
 s.status=f'已准备 {len(specs)} 个真实预测候选；点击预览，不会写入关键帧'
 return specs
def state(s):return json.loads(s.candidate_state) if s.candidate_state else None
def cancel(s,quiet=False):
 session=state(s)
 if not session:return
 a=addon();rig=bpy.data.objects.get(session['target'])
 for scene in bpy.data.scenes:
  if hasattr(scene,'screenshot_pose') and scene.screenshot_pose==s and scene.frame_current!=session['frame']:scene.frame_set(session['frame'])
 if rig:
  if session.get('kind')=='FOOT_DIRECTION' and rig.animation_data:
   for track in rig.animation_data.nla_tracks:
    if track.name in session['nla']:track.mute=session['nla'][track.name]
  if session.get('kind')=='DIRECT_EDIT':a.direct_tools.restore_tracks(rig,session)
  a.retarget.restore_pose(rig,session['snapshot']);restore_backup(rig,session['backup'])
 for o in a.preview.marker_objects(s):
  if o.name in session['guides']:o.location=session['guides'][o.name]
 s.last_result=session['last_result'];s.candidate_state='';s.candidate_active_index=-1
 if hasattr(s,'hand_palm_active_index'):s.hand_palm_active_index=-1
 if not quiet:s.status='已取消候选预览，恢复预览前的姿势与关节点'
 if bpy.context.scene and bpy.context.scene.screenshot_pose==s:a.preview.refresh_lines(bpy.context,s)
def cancel_all():
 for scene in bpy.data.scenes:
  if hasattr(scene,'screenshot_pose') and scene.screenshot_pose.candidate_state:cancel(scene.screenshot_pose,True)
def validate_session(s,session):
 if s.target.name!=session['target'] or bpy.context.scene.frame_current!=session['frame']:raise ValueError('角色或帧已改变，请取消预览后重新开始')
 if signature(s)!=session['signature']:raise ValueError('姿势数据已改变，请取消预览并重新准备候选')
def run_candidate(context,s,session,index,keyframe):
 if session.get('kind')=='DIRECT_EDIT':raise ValueError('请先确认或取消局部拖动')
 if session.get('kind')=='BEFORE_AFTER':raise ValueError('请先返回当前姿势，结束前后对比')
 if session.get('kind')=='HAND_PALM':raise ValueError('请在掌向预览中确认或取消')
 if session.get('kind')=='FOOT_DIRECTION':raise ValueError('请在脚向预览中确认或取消')
 a=addon();rig,mapping,data=check(s);validate_session(s,session)
 if not 0<=index<len(s.pose_candidates):raise ValueError('候选序号无效')
 spec=json.loads(s.pose_candidates[index].spec);document=candidate_data.document(data,spec)
 a.retarget.restore_pose(rig,session['snapshot']);restore_backup(rig,session['backup'])
 canonical=session['canonical']
 points,_=a.pose_data.to_joints(document,s.coordinate_mode,s.depth,s.mirror,s.yaw,s.threshold)
 for o in a.preview.marker_objects(s):o.location=Vector(points[o['sp_joint']])+Vector(session['guides'][o.name])-Vector(canonical[o['sp_joint']])
 flags={n:getattr(s,n) for n in ('keyframe','full_source_rotation','auto_full_quality','auto_depth_select')}
 try:
  s.keyframe=keyframe;s.full_source_rotation=spec['kind']=='FULL';s.auto_full_quality=False;s.auto_depth_select=False
  result=a._apply_current_once(context,s,data_override=document)
 finally:
  for n,v in flags.items():setattr(s,n,v)
 result['candidate_choice']=dict(id=spec['id'],name=spec['name'],source=spec['source'],preview=not keyframe)
 return result
def audition(context,s,index=None):
 a=addon();rig,mapping,data=check(s)
 if not s.pose_candidates:collect(s)
 if not s.candidate_state:
  if signature(s)!=s.candidate_signature:raise ValueError('姿势数据已改变，请刷新候选列表')
  canonical,_=a.pose_data.to_joints(effective_document(s,data),s.coordinate_mode,s.depth,s.mirror,s.yaw,s.threshold)
  session=dict(target=rig.name,snapshot=a.retarget.snapshot_pose(rig),backup=rig.get('screenshot_pose_backup'),last_result=s.last_result,frame=context.scene.frame_current,signature=signature(s),index=-1,canonical=canonical,
               guides={o.name:list(o.location) for o in a.preview.marker_objects(s)})
  s.candidate_state=json.dumps(session,ensure_ascii=False)
 else:session=state(s)
 chosen=s.candidate_index if index is None else index
 try:
  result=run_candidate(context,s,session,chosen,False)
  restore_backup(rig,session['backup']);session['index']=chosen;s.candidate_index=chosen;s.candidate_active_index=chosen;s.candidate_state=json.dumps(session,ensure_ascii=False)
  score=result.get('fit',{}).get('after_px');s.pose_candidates[chosen].score=f'{score:.1f} px' if score is not None else '未启用二维贴合'
  result['candidate_choice']['preview']=True;s.last_result=json.dumps(result,ensure_ascii=False)
  s.status='临时预览：'+s.pose_candidates[chosen].name+'；确认后保留，取消可恢复'
  return result
 except Exception:
  cancel(s,True);raise
def confirm(context,s):
 session=state(s)
 if not session or session['index']<0:raise ValueError('请先预览一个候选')
 if session.get('kind')=='HAND_PALM':raise ValueError('请在掌向预览中确认或取消')
 try:
  result=run_candidate(context,s,session,session['index'],s.keyframe)
  s.chosen_candidate=s.pose_candidates[session['index']].spec;s.chosen_signature=signature(s)
  result['candidate_choice']['preview']=False;s.last_result=json.dumps(result,ensure_ascii=False);s.candidate_state='';s.candidate_active_index=-1
  s.status='已确认候选：'+s.pose_candidates[session['index']].name+('，已打关键帧' if s.keyframe else '')
  return result
 except Exception:
  cancel(s,True);raise

class SPCandidateItem(PropertyGroup):
 spec:StringProperty();score:StringProperty()
class SP_OT_local_apply(Operator):
 bl_idname='screenshot_pose.apply_local';bl_label='仅更新选定部位';bl_options={'REGISTER','UNDO'}
 scope:EnumProperty(items=[('CURRENT','当前所选部位','')]+SCOPES,default='CURRENT')
 def execute(self,context):
  try:
   if self.scope!='CURRENT':context.scene.screenshot_pose.local_scope=self.scope
   apply_local(context,context.scene.screenshot_pose)
  except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}
  return {'FINISHED'}
class SP_OT_candidates(Operator):
 bl_idname='screenshot_pose.candidates';bl_label='姿势候选';bl_options={'REGISTER','UNDO'}
 action:EnumProperty(items=[('COLLECT','准备候选',''),('PREVIEW','预览',''),('PREVIOUS','上一个',''),('NEXT','下一个',''),('CONFIRM','确认',''),('CANCEL','取消',''),('AUTO','恢复自动比较','')],default='COLLECT')
 def execute(self,context):
  s=context.scene.screenshot_pose
  try:
   if self.action=='COLLECT':collect(s)
   elif self.action=='AUTO':
    if s.candidate_state:raise ValueError('请先确认或取消当前预览')
    s.chosen_candidate='';s.chosen_signature='';s.status='已恢复自动比较；下次全身应用时生效'
   elif self.action=='CANCEL':cancel(s)
   elif self.action=='CONFIRM':confirm(context,s)
   elif self.action in ('NEXT','PREVIOUS'):
    if not s.pose_candidates:collect(s)
    current=state(s);index=current['index'] if current else s.candidate_index
    audition(context,s,(index+(1 if self.action=='NEXT' else -1))%len(s.pose_candidates))
   else:audition(context,s)
  except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}
  return {'FINISHED'}
OPERATORS=(SP_OT_local_apply,SP_OT_candidates)
