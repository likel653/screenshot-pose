"""Rest-calibrated full source rotations and optional torso/shoulder chains."""
import math
import bpy
from mathutils import Matrix,Vector
from . import retarget,orientation_math as om

SOURCE={'hips':'root','spine':'spine04','chest':'spine01','neck':'neck02','head':'head'}
for side in ('L','R'):
 for role,bone in [('upper_arm','upperarm01'),('forearm','lowerarm01'),('hand','wrist'),('thigh','upperleg01'),('shin','lowerleg01'),('foot','foot')]:
  SOURCE[role+'.'+side]=bone+'.'+side

def enabled(data,settings):
 return bool(getattr(settings,'full_source_rotation',False) and data.get('full_skeleton') and not data.get('edited_indices') and not settings.mirror and abs(settings.yaw)<1e-6)

def safe(bone,rig):
 return bone and not retarget._active_rotation_constraints(bone) and not retarget._driver_controls_rotation(rig,bone)

def apply(rig,mapping,settings,data,*,roles_to_apply=None):
 report={'touched':[],'warnings':[],'source_bones':0,'errors_deg':{},'mode':'full_source_rotation'}
 source=data['full_skeleton']
 if source.get('schema')!='screenshot_pose/skeleton_v1' or source.get('pose_axes')!='X_right_Y_away_Z_up':raise ValueError('完整源骨架格式或坐标系不兼容')
 bones=source['bones'];report['source_bones']=len(bones)
 roles=source.get('role_map',SOURCE)
 source_name=lambda name:roles.get({'upperleg01':'thigh','upperarm01':'upper_arm','lowerarm01':'forearm','lowerleg01':'shin','wrist':'hand','foot':'foot'}.get(name.split('.')[0],name.split('.')[0])+'.'+name.split('.')[-1],name)
 def mapped(name):
  if source.get('source')=='sam3d_mhr' and name.startswith('toe1-1.'):return ('l_' if name.endswith('.L') else 'r_')+'ball'
  return source_name(name)
 # Validate every frame before mutating any controller.
 rotations={name:(Matrix(om.proper_rotation(item['pose_rotation']).tolist()),Matrix(om.proper_rotation(item['rest_rotation']).tolist())) for name,item in bones.items()}
 heads=lambda name:Vector(bones[mapped(name)]['rest_head'])
 pelvis=(heads('upperleg01.L')+heads('upperleg01.R'))/2
 chest=(heads('upperarm01.L')+heads('upperarm01.R'))/2
 src_body=retarget._frame(heads('upperleg01.L')-heads('upperleg01.R'),chest-pelvis).to_matrix()
 target_body,_=retarget._body_frames(retarget._rest_joints(rig,mapping))
 if target_body is None:raise ValueError('缺少目标骨架静置躯干参考')
 alignment=src_body@target_body.to_matrix().transposed()
 to_rig=rig.matrix_world.to_quaternion().inverted().to_matrix()
 targets={}
 def add(name,src,segment=None,reference_axis=0):
  bone=rig.pose.bones.get(name)
  if not safe(bone,rig) or src not in rotations:return
  posed,rest=rotations[src]
  if segment:
   start,end=segment
   along=Vector(bones[end]['rest_head'])-Vector(bones[start]['rest_head'])
   source_frame=Matrix(om.frame(src_body.col[reference_axis],along).tolist())
   target_frame=Matrix(om.frame(target_body.to_matrix().col[reference_axis],bone.bone.tail_local-bone.bone.head_local).tolist())
   source_pose=posed@rest.transposed()@source_frame
   observed=Vector(bones[end]['pose_head'])-Vector(bones[start]['pose_head'])
   if observed.length>1e-7:source_pose=source_pose.col[1].rotation_difference(observed).to_matrix()@source_pose
   targets[name]=(to_rig@source_pose@target_frame.transposed()@bone.bone.matrix_local.to_3x3()).to_quaternion()
  else:targets[name]=(to_rig@posed@rest.transposed()@alignment@bone.bone.matrix_local.to_3x3()).to_quaternion()
 recovered_wrists=set(data.get('screenshot_recovered_wrists', ()))
 for role,name in mapping.items():
  if role in roles and (roles_to_apply is None or role in roles_to_apply):
   base,_,side=role.partition('.')
   if side in recovered_wrists and base in ('upper_arm','forearm','hand'):continue
   pairs={'upper_arm':('upperarm01','lowerarm01'),'forearm':('lowerarm01','wrist'),
          'thigh':('upperleg01','lowerleg01'),'shin':('lowerleg01','foot'),'foot':('foot','toe1-1')}
   pair=tuple(mapped(n+'.'+side) for n in pairs[base]) if base in pairs else None
   add(name,roles[role],pair,1 if base in ('upper_arm','forearm') else 0)
 # ARP's intermediate controls are deliberately not collapsed into one spine.
 spine=sorted([p for p in rig.pose.bones if p.name.startswith('c_spine_') and p.name.endswith('.x') and p.name[8:10].isdigit()],key=lambda p:p.bone.head_local.z)
 for i,bone in enumerate(spine):
  chain=source.get('spine_chain',['spine05','spine04','spine03','spine02','spine01'])
  index=round(i*(len(chain)-1)/max(len(spine)-1,1))
  add(bone.name,chain[index])
 for side in ('L','R'):
  if side in recovered_wrists:continue
  clavicle=source.get('shoulder_map',{}).get(side,'clavicle.'+side)
  add('c_shoulder.'+side.lower(),clavicle,(clavicle,mapped('upperarm01.'+side)),2)
 initial={n:rig.pose.bones[n].matrix.to_quaternion() for n in targets}
 ordered=sorted(targets,key=lambda n:len(rig.pose.bones[n].parent_recursive))
 for n in ordered:targets[n]=initial[n].slerp(targets[n],settings.strength)
 retarget.absolute_targets(rig,targets,batch=getattr(settings,'video_batch_updates',False))
 for n in ordered:
  err=rig.pose.bones[n].matrix.to_quaternion().rotation_difference(targets[n]).angle
  err=min(err,abs(2*math.pi-err));report['errors_deg'][n]=math.degrees(err)
  if err>math.radians(3):report['warnings'].append(n+' 完整旋转受绑定影响，需核对')
 report['touched']=ordered
 return report
