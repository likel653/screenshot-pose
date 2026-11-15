"""Independent full-pose solves, with animation restored before a single bake."""
import hashlib,importlib,json
from pathlib import Path
import bpy
from mathutils import Vector
from . import pose_data
RUNNING=False
def addon():return importlib.import_module(__package__)

def preserve_lower(s,data):
 if not (s.preview_name and not s.ground_feet and not s.contact_enabled and pose_data.lower_body_cropped(data)):
  return False
 canonical,_=pose_data.to_joints(data,s.coordinate_mode,s.depth,s.mirror,s.yaw,s.threshold)
 return not any((marker.location-Vector(canonical[marker['sp_joint']])).length>1e-5
                for marker in addon().preview.marker_objects(s)
                if marker['sp_joint'].startswith(('hip.','knee.','ankle.','toe.')))

def upper_mapping(mapping):
 return {role:name for role,name in mapping.items() if role.split('.')[0] not in ('hips','thigh','shin','foot')}
def controls(s):
 a=addon();rig=s.target;names={r.bone for r in s.mapping if r.enabled and r.bone in rig.pose.bones}
 root=s.root_bone or next((n for n in ('c_root_master.x','torso') if n in rig.pose.bones),'')
 if root:names.add(root)
 if getattr(s,'fit_torso',False) or getattr(s,'whole_body',False):
  names.update(n for n in ('c_shoulder.l','c_shoulder.r') if n in rig.pose.bones and a.orientation._safe(rig.pose.bones[n],rig))
 names.update(b.name for b,index,value in a.retarget.location_fk_switches(rig,a.pose_tools.mapping_for(s)))
 for side in ('L','R'):
  for finger in a.finger_pose.FINGERS:
   for i in (1,2,3):
    b=a.finger_pose.control(rig,finger,i,side)
    if a.finger_pose.writable(b,rig,finger,side):names.add(b.name)
   n=f'c_{finger}1_base.{side.lower()}'
   if n in rig.pose.bones:names.add(n)
 return names
def source_key(data):
 image=data['image'];path=Path(bpy.path.abspath(image.get('path','')))
 return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else json.dumps(image,sort_keys=True)
def accept(s):
 """Clear per-image choices only when explicitly adopting a new prediction."""
 a=addon()
 if hasattr(s,'fit_model'):s.fit_model='ORTHO'
 if s.candidate_state:a.pose_tools.cancel(s,True)
 a.direct_tools.hide_overlay(s)
 for name in ('last_result','chosen_candidate','chosen_signature','candidate_signature','direct_undo_state','diagnosis','hand_refine_report','hand3d_comparison'):setattr(s,name,'')
 s.pose_candidates.clear();s.hand_palm_options.clear();s.hand_palm_active_index=-1
 for name in ('head_yaw','head_pitch','head_roll','palm_roll_l','palm_roll_r'):setattr(s,name,0.)
 s.palm_flip_l=False;s.palm_flip_r=False;s.hand_candidate_l='0';s.hand_candidate_r='0';s.hand_left='NONE';s.hand_right='NONE'
def apply(context,s,solve):
 global RUNNING
 a=addon()
 if RUNNING or not s.target or s.strength==0:return solve(context,s)
 if s.candidate_state:raise ValueError('请先确认或取消当前预览')
 a.animation_preview.end(s.target)
 rig=s.target;data=a.pose_data.read_pose(bpy.path.abspath(s.json_path));original=a.retarget.snapshot_pose(rig)
 backup=rig.get('screenshot_pose_backup');last=s.last_result;old_init=rig.get('sp_frame_context');keyframe=s.keyframe
 if not s.mapping:a.set_mapping(s)
 mapping=a.pose_tools.mapping_for(s);names=controls(s);finger_names=set()
 if preserve_lower(s,data):
  upper=upper_mapping(mapping)
  names.difference_update(set(mapping.values())-set(upper.values()))
  names.difference_update((s.root_bone,'c_root_master.x','torso'))
  names.difference_update(b.name for b,index,value in a.retarget.location_fk_switches(rig,mapping))
  mapping=upper
  names.update(b.name for b,index,value in a.retarget.location_fk_switches(rig,mapping))
 names.update(b.name for b,k,v in a.retarget._fk_switches(rig,mapping))
 for side in ('L','R'):
  for f in a.finger_pose.FINGERS:
   for i in (1,2,3):
    b=a.finger_pose.control(rig,f,i,side)
    if b and b.name in names:finger_names.add(b.name)
 signature=a.character_tools.fingerprint(rig);baseline=json.loads(rig.get('sp_pose_baseline','null'))
 if not baseline or baseline.get('signature')!=signature:
  seed=json.loads(backup) if backup else original
  baseline=dict(signature=signature,bones={n:seed['bones'].get(n,original['bones'][n]) for n in names})
  rig['sp_pose_baseline']=json.dumps(baseline)
 elif any(n not in baseline['bones'] for n in names):
  seed=json.loads(backup) if backup else original
  for n in names:baseline['bones'].setdefault(n,seed['bones'].get(n,original['bones'][n]))
  rig['sp_pose_baseline']=json.dumps(baseline)
 token=dict(frame=context.scene.frame_current,source=source_key(data));fresh=old_init!=json.dumps(token,sort_keys=True)
 initialized=[];created=None;nla={t.name:t.mute for t in rig.animation_data.nla_tracks} if rig.animation_data else {}
 RUNNING=True
 try:
  if rig.animation_data:
   if rig.animation_data.action:rig.animation_data.action.use_fake_user=True
   rig.animation_data.action=None
   for t in rig.animation_data.nla_tracks:t.mute=True
  if fresh and s.fresh_pose_start:
   for name in names:
    state=baseline['bones'][name]
    b=rig.pose.bones[name];b.location=state['location'];b.scale=state['scale']
    # A fresh image/frame must start from the same mapped control rotations.
    # Resetting only fingers left previous body/foot rotations in local fit,
    # producing a different solve in a sequence than in an independent frame.
    b.rotation_mode=state['rotation_mode']
    for channel in ('rotation_euler','rotation_quaternion','rotation_axis_angle'):setattr(b,channel,state[channel])
    for key,value in state['properties'].items():
     if key in ('bend_all','fingers_grasp') or '_base.' in name:b[key]=value
    initialized.append(name)
   a.retarget._update(rig)
  if not backup:rig['screenshot_pose_backup']=json.dumps(original)
  s.keyframe=False;result=solve(context,s);solved=a.retarget.snapshot_pose(rig)
  # Return original animation ownership, then bake the final result once. No
  # intermediate candidate or unfinished body solve is written to the Action.
  a.retarget.restore_pose(rig,original)
  if keyframe:created=a.pose_tools.copy_action(rig,original)
  action=rig.animation_data.action if rig.animation_data else None
  solved['action']=action.name if action else None
  slot=getattr(rig.animation_data,'action_slot',None) if rig.animation_data else None
  solved['action_slot_handle']=slot.handle if slot else None
  solved['action_fake_user']=bool(action.use_fake_user) if action else False
  # Do not evaluate the previous Action between restoring solved channels and
  # inserting the new keys: its old curves would replace the new solution.
  if keyframe:a.retarget.restore_pose(rig,solved,update=False)
  touched={n for n in initialized if '_base.' not in n};properties=[]
  touched.update(result.get('touched',[]))
  touched.update(r.bone for r in s.mapping if r.enabled and r.role in result.get('applied',[]))
  for key in ('full_rig','contact','feet','orientation','fingers','contacts','fit'):
   section=result.get(key,{})
   touched.update(section.get('touched',[]));properties.extend(section.get('properties',[]))
  for name in names:
   now=solved['bones'][name];before=original['bones'][name]
   if now['matrix_basis']!=before['matrix_basis']:touched.add(name)
   for key,value in now['properties'].items():
    if key in before['properties'] and (keyframe or before['properties'][key]!=value):properties.append((name,key))
  if keyframe:
   a.contact_solver.key_touched(rig,touched,context.scene.frame_current,list(set(tuple(p) for p in properties)))
   for key in solved.get('object_properties',{}):rig.keyframe_insert(data_path=a.retarget.object_property_path(key),frame=context.scene.frame_current,group='MHX FK switches')
   object_paths={a.retarget.object_property_path(k) for k in solved.get('object_properties',{})}
   object_paths.update(a.retarget.location_switch_paths(rig,a.pose_tools.mapping_for(s)))
   for curve in a.animation_preview.curves(rig.animation_data.action):
    if curve.data_path in object_paths:
     for point in curve.keyframe_points:point.interpolation='CONSTANT'
   a.retarget._update(rig)
   for t in rig.animation_data.nla_tracks if rig.animation_data else ():
    if t.name in nla:t.mute=nla[t.name]
  else:
   result['animation_preview']=a.animation_preview.begin(rig,original,solved,names|touched|{n for n,k in properties},nla,context.scene.frame_current)
  rig['sp_frame_context']=json.dumps(token,sort_keys=True)
  result['frame_isolation']=dict(frame=token['frame'],fresh=fresh,initialized=initialized,final_pose_baked_once=keyframe)
  s.last_result=json.dumps(result,ensure_ascii=False,default=str)
  s.status=f'已应用第 {token["frame"]} 帧'+('并打关键帧' if keyframe else '（临时姿势；换帧恢复原动画）')
  return result
 except Exception:
  for t in rig.animation_data.nla_tracks if rig.animation_data else ():
   if t.name in nla:t.mute=nla[t.name]
  a.retarget.restore_pose(rig,original);a.pose_tools.restore_backup(rig,backup);s.last_result=last
  if old_init is None:
   if 'sp_frame_context' in rig:del rig['sp_frame_context']
  else:rig['sp_frame_context']=old_init
  if created:
   created.use_fake_user=False
   if created.users==0:bpy.data.actions.remove(created)
  raise
 finally:s.keyframe=keyframe;RUNNING=False
