"""Full head/palm frames, rest calibration, and FK forearm twist distribution."""
import json,math
import numpy as np
import bpy
from mathutils import Matrix,Quaternion,Vector,Euler
from . import retarget,orientation_math as om,hand_presets,head_math

def _safe(bone,rig):
 return bone and not retarget._active_rotation_constraints(bone) and not retarget._driver_controls_rotation(rig,bone)

def _absolute(rig,bone,q):
 retarget._set_rotation(bone,retarget._to_local_rotation(bone,q));retarget._update(rig)

def _find_finger(rig,finger,side,override=''):
 if override:return rig.pose.bones.get(override)
 side=side.lower();long='Left' if side=='l' else 'Right'
 names=[f'FK-Finger_{finger.title()}1.{side.upper()}',f'c_{finger}1.{side}',f'c_{finger}1_rot.{side}',f'{finger}1.{side}',f'mixamorig:{long}Hand{finger.title()}1',f'{long}Hand{finger.title()}1',f'f_{finger}.01.{side.upper()}',f'{finger}.01.{side.upper()}']
 if retarget.slider_fk_profile(rig) and finger in ('index','pinky'):
  names.append('arm finger '+('2' if finger=='index' else '5')+'a '+long.lower())
 return next((rig.pose.bones[n] for n in names if n in rig.pose.bones),None)

def calibration(rig,mapping,settings):
 forward=Vector({'-Y':(0,-1,0),'Y':(0,1,0),'X':(1,0,0),'-X':(-1,0,0)}[settings.head_forward])
 up=Vector((0,0,1));right=up.cross(forward);head=Matrix((right,up,forward)).transposed()
 result={'schema':'screenshot_pose/calibration_v1','target':rig.name,'head_frame':[list(r) for r in head],'hands':{}}
 for side in ('L','R'):
  hand=rig.pose.bones.get(mapping.get('hand.'+side,''))
  if not hand:continue
  index=_find_finger(rig,'index',side,getattr(settings,'index_'+side.lower()))
  pinky=_find_finger(rig,'pinky',side,getattr(settings,'pinky_'+side.lower()))
  if not index or not pinky:continue
  wrist=hand.bone.head_local;i=index.bone.head_local;p=pinky.bone.head_local
  try:frame=om.frame(i-p,(i+p)*.5-wrist)
  except ValueError:continue
  if getattr(settings,'rest_palm_flip_'+side.lower(),False):frame=frame@np.diag([-1,1,-1])
  result['hands'][side]={'frame':frame.tolist(),'index':index.name,'pinky':pinky.name,'hand':hand.name}
 return result

def store_calibration(rig,mapping,settings):
 result=calibration(rig,mapping,settings);rig['sp_orientation_calibration']=json.dumps(result,ensure_ascii=False);return result

def _twist_angle(q,axis):
 q=q.copy();q.normalize()
 if q.w<0:q.negate()
 dot=Vector((q.x,q.y,q.z)).dot(axis)
 return 2*math.atan2(dot,q.w)

def _clamp_wrist(q,neutral,axis,limit):
 d=q@neutral.inverted();angle=_twist_angle(d,axis);twist=Quaternion(axis,angle);swing=d@twist.inverted()
 if swing.w<0:swing.negate()
 if swing.angle<=limit:return q,False
 clamped=Quaternion((1,0,0,0)).slerp(swing,limit/max(swing.angle,1e-8))
 return (clamped@twist@neutral).normalized(),True

def apply(rig,mapping,settings,data,fit=None):
 report={'head':None,'hands':{},'warnings':[],'touched':[],'properties':[],'dependent_bones':[]}
 details=data.get('orientation',{})
 if details and details.get('schema')!='screenshot_pose/orientation_v1':raise ValueError('局部朝向数据格式不兼容')
 if details and details.get('camera_axes')!='X_right_Y_away_Z_up':raise ValueError('局部朝向相机坐标系不兼容')
 if details and (settings.mirror or abs(settings.yaw)>1e-6):raise ValueError('头手局部朝向需要关闭镜像/朝向修正后重建；镜像图片可直接重新识别')
 cal=calibration(rig,mapping,settings)
 R=Matrix(fit['camera']['rotation']).transposed() if fit else Matrix.Identity(3)
 world_to_rig=rig.matrix_world.to_quaternion().inverted().to_matrix()
 camera_to_rig=world_to_rig@R
 def desired(frame,rest,bone):
  frame=Matrix(om.proper_rotation(frame).tolist());rest=Matrix(rest)
  return (camera_to_rig@frame@rest.inverted()@bone.bone.matrix_local.to_3x3()).to_quaternion()
 head_data=details.get('head')
 if settings.orient_head and getattr(settings,'head_consensus',False):
  head=rig.pose.bones.get(mapping.get('head',''));chest=rig.pose.bones.get(mapping.get('chest',''))
  ref=head_math.reference(data)
  if _safe(head,rig) and ref is not None:
   physical=head.matrix.to_3x3().normalized()@head.bone.matrix_local.to_3x3().normalized().transposed()@Matrix(cal['head_frame'])
   body=np.asarray(camera_to_rig.transposed()@physical)
   neck_frame=None
   if chest:
    neck_frame=np.asarray(camera_to_rig.transposed()@chest.matrix.to_3x3().normalized()@chest.bone.matrix_local.to_3x3().normalized().transposed()@Matrix(cal['head_frame']))
   candidates=list(details.get('head_candidates',[]))
   if head_data and not any(c.get('frame')==head_data.get('frame') for c in candidates):candidates.append(head_data)
   K,camera_info=head_math.camera(data,ref)
   head_data,selection=head_math.select(body,candidates,ref,K,settings.orientation_threshold,neck_frame,details.get('head_before_refinement',head_data))
   selection['camera']=camera_info;report['head_selection']=selection
  else:
   head_data=None;report['head_selection']=dict(chosen='body_head',reason='insufficient_independent_face_evidence')
 if settings.orient_head and head_data and head_data.get('confidence',0)>=settings.orientation_threshold:
  head=rig.pose.bones.get(mapping.get('head',''));neck=rig.pose.bones.get(mapping.get('neck',''))
  if _safe(head,rig):
   offset=Euler((settings.head_pitch,settings.head_roll,settings.head_yaw),'XYZ').to_matrix()
   q=desired((Matrix(head_data['frame'])@offset),cal['head_frame'],head)
   target=head.matrix.to_quaternion().slerp(q,settings.strength*settings.head_strength)
   neck_used=False
   if _safe(neck,rig) and settings.neck_share>0:
    # Neck takes a fraction of the requested change; head is set ABSOLUTELY after it.
    delta=target@head.matrix.to_quaternion().inverted()
    neck_target=Quaternion((1,0,0,0)).slerp(delta,settings.neck_share)@neck.matrix.to_quaternion()
    _absolute(rig,neck,neck_target);report['touched'].append(neck.name);neck_used=True
   _absolute(rig,head,target)
   for _ in range(2):_absolute(rig,head,target)
   error=head.matrix.to_quaternion().rotation_difference(target).angle
   report['head']={'source':head_data['source'],'confidence':head_data['confidence'],'rotation_error_deg':math.degrees(error),'neck_shared':neck_used,'target_quaternion':list(target),'target_frame':head_data['frame']}
   report['touched'].append(head.name)
  else:report['warnings'].append('头骨受约束或未映射，未覆盖其旋转')
 elif settings.orient_head:
  head=rig.pose.bones.get(mapping.get('head',''))
  if _safe(head,rig) and any((settings.head_pitch,settings.head_roll,settings.head_yaw)):
   current=head.matrix.to_quaternion();physical=current.to_matrix()@head.bone.matrix_local.to_3x3().inverted()@Matrix(cal['head_frame'])
   offset=Euler((settings.head_pitch,settings.head_roll,settings.head_yaw),'XYZ').to_matrix()
   q=(physical@offset@physical.inverted()).to_quaternion()@current
   _absolute(rig,head,current.slerp(q,settings.strength*settings.head_strength));report['touched'].append(head.name)
  report['warnings'].append('没有可靠的局部头部结果，保留原姿势并仅应用角度微调')
 for side in ('L','R'):
  if not getattr(settings,'orient_hand_'+side.lower()):continue
  hand=rig.pose.bones.get(mapping.get('hand.'+side,''));forearm=rig.pose.bones.get(mapping.get('forearm.'+side,''))
  if not _safe(hand,rig):report['warnings'].append(side+'手映射受约束或缺失');continue
  hd=details.get('hands',{}).get(side);roll=getattr(settings,'palm_roll_'+side.lower());flip=getattr(settings,'palm_flip_'+side.lower())
  if not hd or hd.get('confidence',0)<settings.orientation_threshold:
   if roll or flip:
    angle=(roll+(math.pi if flip else 0))*settings.strength*settings.hand_strength
    physical=hand.matrix.to_3x3()@hand.bone.matrix_local.to_3x3().inverted()@Matrix(cal['hands'][side]['frame']) if side in cal['hands'] else hand.matrix.to_3x3()
    axis=physical.col[1].normalized();q=Quaternion(axis,angle)@hand.matrix.to_quaternion();_absolute(rig,hand,q);report['touched'].append(hand.name)
   report['warnings'].append(side+'手缺少可靠局部三维结果，保留手腕；可在图片中校正或微调')
   continue
  if side not in cal['hands']:
   report['warnings'].append(side+'手缺少食指/小指参考骨，请在角色方向校准中指定');continue
  frames=[hd['frame']]+hd.get('alternatives',[])
  current=hand.matrix.to_quaternion();neutral=current
  if forearm:
   neutral=forearm.matrix.to_quaternion()@forearm.bone.matrix_local.to_quaternion().inverted()@hand.bone.matrix_local.to_quaternion()
  requested=int(getattr(settings,'hand_candidate_'+side.lower()))
  if requested>len(frames):report['warnings'].append(side+'手当前只有 '+str(len(frames))+' 个有效候选，已使用最后一个')
  candidates=[desired(Matrix(f)@Quaternion((0,1,0),roll+(math.pi if flip else 0)).to_matrix(),cal['hands'][side]['frame'],hand) for f in frames]
  choice=min(range(len(candidates)),key=lambda i:candidates[i].rotation_difference(neutral).angle) if requested==0 else min(requested-1,len(candidates)-1)
  target=current.slerp(candidates[choice],settings.strength*settings.hand_strength)
  wrist_before=hand.head.copy();forearm_before=forearm.matrix.to_quaternion().copy() if forearm else None
  distributed=0.
  if _safe(forearm,rig) and settings.forearm_share>0:
   axis=(hand.head-forearm.head).normalized()
   angle=_twist_angle(target@neutral.inverted(),axis)
   distributed=max(-math.radians(130),min(math.radians(130),angle*settings.forearm_share))
   _absolute(rig,forearm,Quaternion(axis,distributed)@forearm.matrix.to_quaternion())
   if (hand.head-wrist_before).length>.001:
    _absolute(rig,forearm,forearm_before);distributed=0.;report['warnings'].append(side+'手绑定在扭转前臂时移动手腕，已回退前臂分配')
   else:report['touched'].append(forearm.name)
  limited=False
  if forearm:
   axis=(hand.head-forearm.head).normalized()
   neutral=forearm.matrix.to_quaternion()@forearm.bone.matrix_local.to_quaternion().inverted()@hand.bone.matrix_local.to_quaternion()
   target,limited=_clamp_wrist(target,neutral,axis,settings.wrist_limit)
  _absolute(rig,hand,target)
  for _ in range(2):_absolute(rig,hand,target)
  if limited:report['warnings'].append(side+'手腕弯曲超出限制，已保留较自然的朝向；可调高手腕弯曲上限')
  error=hand.matrix.to_quaternion().rotation_difference(target).angle
  report['hands'][side]={'source':hd['source'],'confidence':hd['confidence'],'candidate':choice+1,'forearm_twist_deg':math.degrees(distributed),'wrist_limited':limited,'rotation_error_deg':math.degrees(error),'target_quaternion':list(target)}
  report['touched'].append(hand.name)
  preset=hand_presets.selected(settings,data,side)
  if settings.auto_finger_curl and 'curls' in hd and preset=='NONE' and not (getattr(settings,'finger_joints',False) and 'world' in hd):
   report['dependent_bones'].extend(hand_presets.dependent_bones(rig,side))
   for finger,curl in hd['curls'].items():
    bone=rig.pose.bones.get(f'c_{finger}1_base.{side.lower()}')
    if bone and 'bend_all' in bone:
     bone['bend_all']=float(curl)*settings.strength
     report['properties'].append((bone.name,'bend_all'))
     if 'ik_fk_switch' in bone:bone['ik_fk_switch']=1.;report['properties'].append((bone.name,'ik_fk_switch'))
   if 'fingers_grasp' in hand:hand['fingers_grasp']=0.;report['properties'].append((hand.name,'fingers_grasp'))
 report['warnings'].extend(details.get('warnings',[]));retarget._update(rig)
 return report
