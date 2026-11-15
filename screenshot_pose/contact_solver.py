"""Opt-in two-bone contact solve, baked into FK without permanent constraints."""
import math
import bpy
from mathutils import Vector
from . import retarget


def solve_two_bone(hip,target,pole,upper,lower):
    """Exact two-link geometry, preserving lengths and clamping unreachable targets."""
    delta=target-hip
    if upper<=1e-8 or lower<=1e-8 or delta.length<=1e-8:
        raise ValueError('腿部骨长或脚部目标无效')
    direction=delta.normalized()
    distance=max(abs(upper-lower)+1e-6,min(delta.length,upper+lower-1e-6))
    effective=hip+direction*distance
    along=(upper*upper-lower*lower+distance*distance)/(2*distance)
    height=math.sqrt(max(0,upper*upper-along*along))
    bend=pole-hip; bend-=direction*bend.dot(direction)
    if bend.length<1e-7:
        axis=Vector((0,1,0)) if abs(direction.y)<.9 else Vector((1,0,0))
        bend=direction.cross(axis)
    knee=hip+direction*along+bend.normalized()*height
    return knee,effective,(effective-target).length


def _world_head(rig,bone):
    return rig.matrix_world @ bone.head


def _aim(rig,pbone,direction):
    # A world direction is transformed as a vector; translation must not enter.
    local=rig.matrix_world.inverted_safe().to_3x3() @ direction
    if local.length<1e-8:raise ValueError('无法求解零长度方向')
    current=pbone.matrix.to_quaternion()
    actual=current @ Vector((0,1,0))
    desired=actual.rotation_difference(local.normalized()) @ current
    retarget._set_rotation(pbone,retarget._to_local_rotation(pbone,desired))
    retarget._update(rig)


def _translate_world(rig,pbone,delta):
    desired=pbone.matrix.copy()
    desired.translation += rig.matrix_world.inverted_safe().to_3x3() @ delta
    kwargs={}
    if pbone.parent:
        kwargs={'parent_matrix':pbone.parent.matrix,'parent_matrix_local':pbone.parent.bone.matrix_local}
    basis=pbone.bone.convert_local_to_pose(desired,pbone.bone.matrix_local,invert=True,**kwargs)
    pbone.location=basis.translation
    retarget._update(rig)


def root_control(rig,mapping,root_name=''):
    names=[root_name] if root_name else ['c_root_master.x','torso',mapping.get('hips','')]
    root=next((rig.pose.bones.get(n) for n in names if rig.pose.bones.get(n)),None)
    if root is None or any(root.lock_location):
        raise ValueError('找不到可平移的骨盆控制骨，请在接地选项中指定')
    if any(c.type in {'COPY_LOCATION','COPY_TRANSFORMS','CHILD_OF','ARMATURE','TRANSFORM'} and not c.mute and c.influence>.001 for c in root.constraints):
        raise ValueError('骨盆位置由约束控制，请选择可直接平移的根控制骨')
    path=root.path_from_id('location')
    if rig.animation_data and any(f.data_path==path and not f.mute for f in rig.animation_data.drivers):
        raise ValueError('骨盆位置由驱动控制，请选择可平移的根控制骨')
    return root


def grounded_sides(settings):
    if getattr(settings,'contact_enabled',False):return ()
    return ('L','R') if getattr(settings,'ground_mode','BOTH')=='BOTH' else (settings.ground_mode,)

def ground_feet(rig,mapping,*,floor=0.,pelvis_offset=0.,clearance_ratio=.055,root_name='',flatten=True,sole_anchors=None,sides=('L','R')):
    result={'warnings':[],'touched':[],'errors':{}}
    root=root_control(rig,mapping,root_name)
    legs={}
    retarget._update(rig)
    if not sides or any(side not in ('L','R') for side in sides):raise ValueError('请指定有效的接地脚')
    for side in sides:
        bones=[rig.pose.bones.get(mapping.get(role+'.'+side,'')) for role in ('thigh','shin','foot')]
        if any(b is None for b in bones):raise ValueError('双脚接地需要两侧完整的大腿、小腿和脚映射')
        if any(retarget._active_rotation_constraints(b) or retarget._driver_controls_rotation(rig,b) for b in bones):
            raise ValueError('接地映射包含受约束的骨骼，请改选 FK 控制骨')
        upper,lower,foot=bones
        hip,knee,ankle=[_world_head(rig,b) for b in bones]
        l1,l2=(knee-hip).length,(ankle-knee).length
        target=ankle.copy();target.z=floor+(l1+l2)*clearance_ratio
        anchors=(sole_anchors or {}).get(side)
        saved_rotation=foot.matrix.to_quaternion().copy() if anchors else None
        if anchors:
            matrix=rig.matrix_world.to_3x3()@foot.matrix.to_3x3()
            target.z=floor-min((matrix@Vector(p)).z for p in anchors)
        direction=rig.matrix_world.to_3x3() @ (foot.tail-foot.head)
        if flatten and not anchors and Vector((direction.x,direction.y,0)).length>1e-7:
            # Keep the foot's rest pitch (especially high heels), only level its heading.
            rest=rig.matrix_world.to_3x3() @ (foot.bone.tail_local-foot.bone.head_local)
            horizontal=math.hypot(rest.x,rest.y)
            direction.z=math.hypot(direction.x,direction.y)*rest.z/max(horizontal,1e-7)
        legs[side]=dict(bones=bones,hip=hip,knee=knee,ankle=ankle,target=target,l1=l1,l2=l2,direction=direction,rotation=saved_rotation)
    delta_z=sum(v['target'].z-v['ankle'].z for v in legs.values())/len(legs)+pelvis_offset
    before=_world_head(rig,root)
    _translate_world(rig,root,Vector((0,0,delta_z)))
    actual_delta=_world_head(rig,root)-before
    if abs(actual_delta.z-delta_z)>1e-4:
        raise ValueError('根控制骨未按预期移动，已中止；请检查动画或位置驱动')
    result['touched'].append(root.name)
    result['pelvis_delta_z']=delta_z
    for side,item in legs.items():
        upper,lower,foot=item['bones']
        # Reconcile controller evaluation after each link, using its actual head.
        for iteration in range(3):
            hip=_world_head(rig,upper)
            pole=item['knee']+actual_delta
            knee,end,unreachable=solve_two_bone(hip,item['target'],pole,item['l1'],item['l2'])
            _aim(rig,upper,knee-hip)
            _aim(rig,lower,end-_world_head(rig,lower))
            if item['rotation']:
                retarget._set_rotation(foot,retarget._to_local_rotation(foot,item['rotation']));retarget._update(rig)
            else:_aim(rig,foot,item['direction'])
        error=(_world_head(rig,foot)-item['target']).length
        result['errors'][side]=error
        if error>max(.001,(item['l1']+item['l2'])*.005):
            result['warnings'].append(f'{side}脚目标超出骨长或受绑定影响，残差 {error:.4f}；请调整骨盆高度/关节点')
        result['touched'].extend(b.name for b in item['bones'])
    return result


def key_touched(rig,names,frame,properties=()):
    from . import animation_preview
    paths=set()
    for name in set(names):
        bone=rig.pose.bones[name]
        channel='rotation_quaternion' if bone.rotation_mode=='QUATERNION' else 'rotation_axis_angle' if bone.rotation_mode=='AXIS_ANGLE' else 'rotation_euler'
        bone.keyframe_insert(data_path=channel,frame=frame,group=name)
        paths.add(bone.path_from_id(channel));paths.add(bone.path_from_id('location'))
        bone.keyframe_insert(data_path='location',frame=frame,group=name)
    for name,key in properties:
        rig.pose.bones[name].keyframe_insert(data_path='["'+key+'"]',frame=frame,group=name)
        paths.add(rig.pose.bones[name].path_from_id()+'["'+key+'"]')
    animation_preview.unmute_keyed(rig,paths)
