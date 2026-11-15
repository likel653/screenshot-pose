"""Individual FK phalanges with palm-relative roll calibration and spread."""
import math
import numpy as np
from mathutils import Matrix,Quaternion,Vector
from . import retarget,orientation_math as om,hand_presets

FINGERS=('thumb','index','middle','ring','pinky')
FINGER_ALONG=Vector((0,1,0))

def control(rig,finger,segment,side):
 s=side.lower();long='Left' if side=='L' else 'Right'
 names=[f'FK-Finger_{finger.title()}{segment}.{side}',f'c_{finger}{segment}_rot.{s}',f'{finger}.{segment:02d}.{side}',
        f'f_{finger}.{segment:02d}.{side}',f'mixamorig:{long}Hand{finger.title()}{segment}',f'{long}Hand{finger.title()}{segment}']
 return next((rig.pose.bones[n] for n in names if n in rig.pose.bones),None)

def writable(bone,rig,finger,side):
 if bone is None or retarget._driver_controls_rotation(rig,bone):return False
 for c in retarget._active_rotation_constraints(bone):
  # ARP adds bend_all in local X. With bend_all reset to zero, individual
  # rotation channels remain writable; do not mute/delete these constraints.
  if not(c.type=='COPY_ROTATION' and c.mix_mode in ('OFFSET','ADD') and
         c.subtarget==f'{finger}_bend_all.{side.lower()}' and c.owner_space=='LOCAL'):return False
 return True


def hinge_axis(bone,palm_rest,side,finger,has_bend_driver):
 """Return a flexion-positive axis in the child's rest frame."""
 if has_bend_driver:
  # ARP bend_all curls around negative local X on both hands.
  return Vector((-1,0,0))
 rest=bone.bone.matrix_local.to_quaternion()
 inward=palm_rest.col[2]*(1 if side=='R' else -1)
 if finger=='thumb':
  inward-=palm_rest.col[0]
 axis=rest.inverted()@((rest@FINGER_ALONG).cross(inward))
 return axis.normalized() if axis.length>=1e-6 else Vector((1,0,0))


def hinge_angle(rotation,axis):
 """Measure signed curl from the segment direction in its flexion plane."""
 along=rotation@FINGER_ALONG
 across,forward=along.dot(axis.cross(FINGER_ALONG)),along.dot(FINGER_ALONG)
 return math.atan2(across,forward) if math.hypot(across,forward)>1e-6 else 0.


def limit_targets(targets,initial,hinges,strength):
 raw=targets.copy()
 # Finger targets are inserted from the base to the tip. Each bounded joint
 # inherits the corrected orientation of the preceding phalanx.
 for name in targets:
  if name not in hinges:
   targets[name]=initial[name].slerp(raw[name],strength)
   continue
  parent,offset,axis,(lo,hi)=hinges[name]
  def bounded(pose):
   relative=(pose[parent]@offset).inverted()@pose[name]
   return max(lo,min(hi,hinge_angle(relative,axis)))
  start,end=bounded(initial),bounded(raw)
  angle=start+(end-start)*strength
  targets[name]=targets[parent]@offset@Quaternion(axis,angle)

def apply(rig,mapping,settings,data,calibration):
 report={'touched':[],'properties':[],'warnings':[],'hands':{},'dependent_bones':[]}
 strength=settings.strength*settings.hand_strength
 if strength<=0:return report
 for side in ('L','R'):
  if getattr(settings,'hand_side',None) not in (None,side):continue
  manual=hand_presets.manual_fingers(data,side);requested=getattr(settings,'finger_scope',None)
  if not manual and not (getattr(settings,'auto_finger_curl',True) and getattr(settings,'finger_joints',True)):continue
  if requested is not None and not requested:continue
  if not getattr(settings,'orient_hand_'+side.lower()) and not manual:continue
  if hand_presets.selected(settings,data,side)!='NONE' and not manual:continue
  hd=data.get('orientation',{}).get('hands',{}).get(side,{})
  if (hd.get('confidence',0)<settings.orientation_threshold and not manual) or 'world' not in hd or side not in calibration['hands']:
   report['warnings'].append(side+'手缺少可靠的 21 个三维点，未更新逐节手指；可使用手势约束')
   report['hands'][side]={'joints':0,'status':'missing_reliable_21_points'}
   continue
  points=np.asarray(hd['world'],float)
  if points.shape!=(21,3) or not np.isfinite(points).all():raise ValueError('逐节手指需要有效的 21 个三维点')
  source_palm=Matrix(om.proper_rotation(hd['frame']).tolist())
  hand=rig.pose.bones.get(mapping.get('hand.'+side,''))
  if not hand:continue
  selected=set(FINGERS if requested is None else requested)
  if hd.get('source')=='manual_hand_landmarks' or hand_presets.selected(settings,data,side)!='NONE':selected&=manual
  if len(selected)<5 and abs(float(hand.get('fingers_grasp',0)))>1e-6:raise ValueError('当前角色启用了整手 fingers_grasp 联动；请先将其归零，再单独更新手指')
  palm_rest=Matrix(calibration['hands'][side]['frame'])
  # Respect the evaluated wrist (including manual roll and wrist limits).
  palm_actual=hand.matrix.to_3x3().normalized()@hand.bone.matrix_local.to_3x3().normalized().transposed()@palm_rest
  targets={};initial={};sources={};hinges={};offset_controls=set()
  native=hd.get('bone_frames',{});skeleton=data.get('full_skeleton',{}).get('bones',{})
  mano_native=bool(hd.get('native_source')=='MANO' and getattr(settings,'hand_native_roll',True) and not manual)
  if hd.get('native_source')=='MANO' and not mano_native:native={}
  source_rest_palm=None
  if native and mano_native:
   source_rest_palm=om.proper_rotation(hd['source_rest_palm'])
  elif native and 'wrist.'+side in skeleton:
   restpoints=[np.asarray(native[f'{f}:1']['rest_head']) for f in ('index','middle','pinky')]
   source_rest_palm=om.frame(restpoints[0]-restpoints[2],np.mean(restpoints,axis=0)-np.asarray(skeleton['wrist.'+side]['rest_head']))
  used=[]
  for fi,finger in enumerate(FINGERS):
   if requested is not None and finger not in requested:continue
   if (hd.get('source')=='manual_hand_landmarks' or hand_presets.selected(settings,data,side)!='NONE') and finger not in manual:continue
   chain=[control(rig,finger,n,side) for n in (1,2,3)]
   if not all(writable(b,rig,finger,side) for b in chain):
    report['warnings'].append(side+'手'+finger+'缺少可写的逐节 FK 控制链，保持原手指');continue
   used.append(finger)
   base=rig.pose.bones.get(f'c_{finger}1_base.{side.lower()}')
   if base:
    for key,value in [('bend_all',0.),('ik_fk_switch',1.)]:
     if key in base:base[key]=value;report['properties'].append((base.name,key))
   for n,bone in enumerate(chain,1):
    # ARP's bend driver can still add an offset through phalanx scale even
    # when both bend_all and fingers_grasp are zero.
    if retarget._active_rotation_constraints(bone):offset_controls.add(bone.name)
    # Thumb starts at point 1; each finger has MCP/PIP/DIP/tip.
    start=1+fi*4+n-1;along=points[start+1]-points[start]
    if np.linalg.norm(along)<1e-6:raise ValueError('逐节手指关节点重合')
    item=native.get(f'{finger}:{n}')
    target_along=bone.bone.tail_local-bone.bone.head_local;hint=0
    if item and mano_native:
     rest_along=np.asarray(item['rest_tail'])-np.asarray(item['rest_head'])
     for h in (0,2,1):
      if np.linalg.norm(np.cross(source_rest_palm[:,h],rest_along/np.linalg.norm(rest_along)))>.05 and np.linalg.norm(np.cross(np.asarray(palm_rest)[:,h],np.asarray(target_along.normalized())))>.05:hint=h;break
    if not item:
     for h in (0,2,1):
      if np.linalg.norm(np.cross(np.asarray(source_palm)[:,h],along/np.linalg.norm(along)))>.05 and np.linalg.norm(np.cross(np.asarray(palm_rest)[:,h],np.asarray(target_along.normalized())))>.05:hint=h;break
    if item and source_rest_palm is not None:
     rest_frame=om.frame(source_rest_palm[:,hint],np.asarray(item['rest_tail'])-np.asarray(item['rest_head']))
     posed=om.proper_rotation(item['pose_rotation'])@om.proper_rotation(item['rest_rotation']).T@rest_frame
     source_frame=Matrix((om.frame(posed[:,0],along) if mano_native else posed).tolist())
     mode='mano_native_roll_with_3d_segments' if mano_native else 'native_joint_rotation'
    else:
     source_frame=Matrix(om.frame(np.asarray(source_palm)[:,hint],along).tolist());mode='individual_3d_segments'
    target_frame=Matrix(om.frame(np.asarray(palm_rest)[:,hint],target_along).tolist())
    desired=palm_actual@source_palm.transposed()@source_frame@target_frame.transposed()@bone.bone.matrix_local.to_3x3()
    targets[bone.name]=desired.to_quaternion();initial[bone.name]=bone.matrix.to_quaternion();sources[bone.name]=mode
    # Fingers retain spread at the base. The thumb keeps opposition and
    # only its tip is restricted to flexion. Explicit edits stay as authored.
    if finger not in manual and (n==3 or (n==2 and finger!='thumb')):
     parent=chain[n-2]
     offset=parent.bone.matrix_local.to_quaternion().inverted()@bone.bone.matrix_local.to_quaternion()
     axis=hinge_axis(bone,palm_rest,side,finger,bone.name in offset_controls)
     limits=(-10,115) if n==2 else (-15,90)
     hinges[bone.name]=(parent.name,offset,axis,tuple(map(math.radians,limits)))
  if len(used)==5 and 'fingers_grasp' in hand:hand['fingers_grasp']=0.;report['properties'].append((hand.name,'fingers_grasp'))
  retarget._update(rig)
  limit_targets(targets,initial,hinges,strength)
  ordered=retarget.absolute_targets(rig,targets,batch=getattr(settings,'video_batch_updates',False),offset_controls=offset_controls)
  errors={}
  for name in ordered:
   err=rig.pose.bones[name].matrix.to_quaternion().rotation_difference(targets[name]).angle
   errors[name]=math.degrees(min(err,abs(2*math.pi-err)))
  report['hands'][side]={'joints':len(ordered),'sources':sources,'errors_deg':errors,'hinge_joints':list(hinges)}
  report['touched'].extend(ordered)
  report['dependent_bones'].extend(hand_presets.dependent_bones(rig,side,used))
  if errors and max(errors.values())>3:report['warnings'].append(side+'手部分指骨未达到目标旋转，请检查角色控制链')
 return report
