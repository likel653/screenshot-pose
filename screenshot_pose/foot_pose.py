"""Foot orientation and rigid sole references from the target's weighted shoe mesh."""
import math
import numpy as np
import bpy
from mathutils import Vector,Matrix,Quaternion
from . import retarget,orientation_math as om,contact_solver,foot_source
from . import camera_data

def _group_side(name,side):
 n=name.lower();s=side.lower()
 foot=any(k in n for k in ('foot','toe','足首','つま先','脚掌','脚趾'))
 sided=n.endswith(('.'+s,'_'+s)) or ('left' if s=='l' else 'right') in n or ('左' if s=='l' else '右') in n
 return foot and sided

def sole_calibration(rig,mapping,mesh_name=''):
 descendants=set(rig.children_recursive)
 meshes=[o for o in bpy.context.scene.objects if o.type=='MESH' and not o.hide_render and (o in descendants or any(m.type=='ARMATURE' and m.object==rig for m in o.modifiers))]
 if mesh_name:meshes=[o for o in meshes if o.name==mesh_name]
 else:
  shoes=[o for o in meshes if any(k in o.name.lower() for k in ('shoe','boot','靴','鞋'))]
  if shoes:meshes=shoes
 report={}
 for side in ('L','R'):
  foot=rig.pose.bones.get(mapping.get('foot.'+side,''))
  if not foot:continue
  points=[]
  for mesh in meshes:
   groups={g.index for g in mesh.vertex_groups if _group_side(g.name,side)}
   if not groups:continue
   transform=rig.matrix_world.inverted_safe()@mesh.matrix_world
   points.extend(tuple(transform@v.co) for v in mesh.data.vertices if sum(g.weight for g in v.groups if g.group in groups)>.5)
  if len(points)<12:continue
  P=np.array(points);ankle=np.array(foot.bone.head_local);forward=np.array(foot.bone.tail_local-foot.bone.head_local);forward[2]=0
  norm=np.linalg.norm(forward)
  if norm<1e-7:continue
  forward/=norm
  # Lower convex hull in the longitudinal/height plane yields a supporting
  # line: no rigid shoe sample may sit below the chosen rear/front contacts.
  t=(P-ankle)@forward;order=sorted(range(len(P)),key=lambda i:(t[i],P[i,2]))
  hull=[]
  for i in order:
   if hull and abs(t[i]-t[hull[-1]])<1e-8:continue
   while len(hull)>=2:
    a,b=hull[-2:];cross=(t[b]-t[a])*(P[i,2]-P[a,2])-(P[b,2]-P[a,2])*(t[i]-t[a])
    if cross>1e-10:break
    hull.pop()
   hull.append(i)
  if len(hull)<2:continue
  edges=[(a,b) for a,b in zip(hull,hull[1:]) if t[b]-t[a]>.01]
  if not edges:continue
  a,b=max(edges,key=lambda ab:(t[ab[1]]-t[ab[0]])/(1+abs((P[ab[1],2]-P[ab[0],2])/(t[ab[1]]-t[ab[0]]))))
  heel=P[a];toe=P[b];along=toe-heel
  if np.linalg.norm(along)<.01:continue
  bottom=P[P[:,2]<=np.quantile(P[:,2],.20)+.002]
  # Always retain hull endpoints and all hull samples when downsampling.
  bottom=np.vstack((bottom,P[hull]))
  frame=om.frame(np.cross(along,[0,0,1]),along)
  # Keep a bounded sample of the lower surface for target-ankle height solving.
  if len(bottom)>128:bottom=np.vstack((bottom[np.linspace(0,len(bottom)-1,124).astype(int)],heel,toe,P[np.argmin(P[:,2])]))
  inv=foot.bone.matrix_local.inverted_safe()
  anchors=[list(inv@Vector(p)) for p in bottom]
  sole=P[P[:,2]<=np.quantile(P[:,2],.35)+.002]
  longitudinal=sole@forward
  low,high=np.quantile(longitudinal,[.12,.88])
  shoe_axis=[list(inv@Vector(group.mean(0))) for group in (sole[longitudinal<=low],sole[longitudinal>=high])]
  report[side]=dict(frame=frame.tolist(),anchors=anchors,heel=list(inv@Vector(heel)),forefoot=list(inv@Vector(toe)),shoe_axis=shoe_axis,meshes=[m.name for m in meshes],samples=len(points))
 return report

def align_visible_sole(rig, foot, rest, camera, axis, strength):
 local=rest.get('shoe_axis',[rest['heel'],rest['forefoot']])
 def anchors():
  transform=rig.matrix_world@foot.matrix
  return np.array([transform@Vector(point) for point in local])
 def image_axis(points):
  heel,toe=camera_data.project(points,camera)
  line=toe-heel
  length=np.linalg.norm(line)
  angle=math.degrees(math.acos(float(np.clip(axis@line/max(np.linalg.norm(axis)*length,1e-8),-1,1))))
  return angle,length
 points=anchors()
 direction=Vector(points[1]-points[0]);length=direction.length
 if length<1e-6:return None
 if camera is None:
  # In the default front view only the image direction is known.
  camera=dict(rotation=np.eye(3),scale=np.linalg.norm(axis)/length,shift=[0.,0.])
 camera_rotation=Matrix(camera['rotation'])
 before,span_before=image_axis(points)
 screen=axis/np.linalg.norm(axis);world_axis=camera_rotation.transposed()@Vector((screen[0],0,-screen[1]))
 p=np.array(rig.matrix_world@foot.head)
 projected=camera_data.project(np.array([p,p+np.array(world_axis)*length]),camera)
 max_span=max(np.linalg.norm(projected[1]-projected[0]),1e-6)
 transverse=float(np.clip(np.linalg.norm(axis)/max_span,.35,.88))
 target_span=max_span*transverse
 if before<30 and span_before>=.5*target_span:return None
 depth=(camera_rotation@direction.normalized()).y
 in_camera=Vector((screen[0]*transverse,math.copysign(math.sqrt(1.-transverse**2),depth),-screen[1]*transverse))
 object_rotation=rig.matrix_world.to_quaternion()
 turn=direction.rotation_difference(camera_rotation.transposed()@in_camera)
 angle=min(turn.angle,abs(2*math.pi-turn.angle))
 limit=math.radians(60)
 if angle>limit:
  turn=Quaternion().slerp(turn,limit/angle)
  angle=limit
 desired=object_rotation.inverted()@turn@object_rotation@foot.matrix.to_quaternion()
 basis=foot.matrix_basis.copy()
 try:
  target=foot.matrix.to_quaternion().slerp(desired,strength)
  retarget._set_rotation(foot,retarget._to_local_rotation(foot,target));retarget._update(rig)
  after,span_after=image_axis(anchors())
  improved=before-after>20 or span_after-span_before>.15*target_span
  if not improved or after>30 or span_after<.5*target_span or np.linalg.norm(np.array(rig.matrix_world@foot.head)-p)>.002:
   foot.matrix_basis=basis;retarget._update(rig)
   return None
  error=foot.matrix.to_quaternion().rotation_difference(target).angle
  return dict(before_deg=before,after_deg=after,span_before=float(span_before),span_after=float(span_after),
              swing_deg=math.degrees(angle),rotation_error_deg=math.degrees(min(error,abs(2*math.pi-error))),
              metric='visible_heel_to_toe_projection')
 except Exception:
  foot.matrix_basis=basis;retarget._update(rig)
  raise


def apply(rig,mapping,settings,data,fit=None,calibration=None,sides=('L','R')):
 cal=calibration if calibration is not None else sole_calibration(rig,mapping,settings.shoe_mesh.name if settings.shoe_mesh else '')
 result={'warnings':[],'touched':[],'sole_anchors':{},'feet':{},'calibration':cal}
 C=Matrix(fit['camera']['rotation']).transposed() if fit else Matrix.Identity(3)
 object_rotation=rig.matrix_world.to_quaternion().to_matrix();to_rig=object_rotation.inverted()
 observations=data.get('orientation',{}).get('feet',{})
 for side in sides:
  grounded=settings.ground_feet and side in contact_solver.grounded_sides(settings)
  foot=rig.pose.bones.get(mapping.get('foot.'+side,''))
  if not foot or retarget._active_rotation_constraints(foot) or retarget._driver_controls_rotation(rig,foot):
   result['warnings'].append(side+'脚缺少可直接旋转的 FK 映射');continue
  fd=observations.get(side);rest=cal.get(side)
  if not rest:
   result['warnings'].append(side+'脚未找到鞋底参考，保留原有脚部求解；可指定鞋子网格');continue
  rest_frame=Matrix(rest['frame'])
  if foot_source.native_frame(data,side) is not None or (fd and fd.get('confidence',0)>=settings.foot_threshold):
   frame,source=foot_source.frame_for(data,side,fd or {})
   desired_world=C@Matrix(frame.tolist())
  else:
   if not grounded:continue
   desired_world=object_rotation@foot.matrix.to_3x3()@foot.bone.matrix_local.to_3x3().inverted()@rest_frame;source='target_sole_ground_heading'
  if grounded:
   forward=desired_world.col[1].copy();forward.z=0
   if forward.length<1e-6:result['warnings'].append(side+'脚水平朝向不确定，保留当前旋转');continue
   desired_world=Matrix(om.frame(forward.cross(Vector((0,0,1))),forward).tolist())
  requested=(to_rig@desired_world@rest_frame.inverted()@foot.bone.matrix_local.to_3x3()).to_quaternion()
  target=foot.matrix.to_quaternion().slerp(requested,settings.strength)
  retarget._set_rotation(foot,retarget._to_local_rotation(foot,target));retarget._update(rig)
  result['touched'].append(foot.name);result['sole_anchors'][side]=rest['anchors']
  result['feet'][side]={'source':source,'confidence':fd.get('confidence') if fd else None,'rotation_error_deg':math.degrees(foot.matrix.to_quaternion().rotation_difference(target).angle),'sole_meshes':rest['meshes'],'sample_count':rest['samples']}
  if not grounded and source=='sam3d_native_sole_rotation':
   axes=[('sam3d_sole_with_reviewed_shoe_axis',foot_source.reviewed_shoe_axis(data,side,fd or {}))]
   if fit and (fit.get('leg_refinement') or data.get('screenshot_roll_review')):
    axes.append(('sam3d_sole_with_visible_shoe_axis',foot_source.visible_shoe_axis(data,side)))
   for applied_source,axis in axes:
    if axis is None:continue
    try:alignment=align_visible_sole(rig,foot,rest,fit['camera'] if fit else None,axis,settings.strength)
    except (ValueError,FloatingPointError):alignment=None
    if alignment:
     result['feet'][side].update(source=applied_source,image_alignment=alignment,rotation_error_deg=alignment['rotation_error_deg'])
     break
 return result

def contact_report(rig,mapping,cal,floor):
 report={}
 for side,state in cal.items():
  foot=rig.pose.bones.get(mapping.get('foot.'+side,''))
  if not foot:continue
  transform=rig.matrix_world@foot.matrix
  report[side]={name:float((transform@Vector(state[name])).z-floor) for name in ('heel','forefoot')}
  report[side]['minimum_sole_clearance']=float(min((transform@Vector(p)).z-floor for p in state['anchors']))
 return report
