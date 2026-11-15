"""Explicit ARP hand presets, not an automatic finger-recognition claim."""
import math
from mathutils import Quaternion,Vector
from . import retarget

def selected(settings,data,side):
    preset=getattr(settings,'hand_left' if side=='L' else 'hand_right')
    if preset!='NONE':return preset
    if getattr(settings,'hand_semantics',False):
        hd=data.get('orientation',{}).get('hands',{}).get(side,{})
        suggested=hd.get('gesture_suggestion','UNKNOWN')
        rig=getattr(settings,'target',None)
        compatible=bool(rig and all(f'c_{f}1_base.{side.lower()}' in rig.pose.bones and 'bend_all' in rig.pose.bones[f'c_{f}1_base.{side.lower()}'] for f in ('thumb','index','middle','ring','pinky')))
        model_suggestion=hd.get('source')=='cropped_hand_refined' or (hd.get('source')=='rtmpose_hand_crop_2d_palm' and hd.get('gesture_vote_count',0)>=2)
        if compatible and model_suggestion and hd.get('confidence',0)>=.65 and suggested in ('V','OPEN','FIST'):return suggested
    return 'NONE'

def manual_fingers(data,side):
    return set(data.get('orientation',{}).get('hands',{}).get(side,{}).get('manual_fingers',[]))&{'thumb','index','middle','ring','pinky'}

def dependent_bones(rig,side,fingers=('thumb','index','middle','ring','pinky')):
    return [n for f in fingers if (n:=f'{f}_bend_all.{side.lower()}') in rig.pose.bones and retarget._driver_controls_rotation(rig,rig.pose.bones[n])]


def apply(rig,side,preset='V',strength=1.,spread=.16,grip=.75,exclude_fingers=()):
    if preset not in ('V','OPEN','FIST','GRIP'):raise ValueError('未知手势约束')
    if not 0<=strength<=1 or not 0<=grip<=1:raise ValueError('手势强度无效')
    if strength==0:return {'touched':[],'properties':[],'warnings':[]}
    active=set(('thumb','index','middle','ring','pinky'))-set(exclude_fingers)
    if not active:return {'touched':[],'properties':[],'warnings':[]}
    suffix='.'+side.lower()
    bases={f:rig.pose.bones.get('c_'+f+'1_base'+suffix) for f in ('thumb','index','middle','ring','pinky')}
    if any(b is None or 'bend_all' not in b for b in bases.values()):
        raise ValueError('当前手势预设需要 Auto-Rig Pro 的 bend_all 手指控制器；此角色请手动调整指骨')
    hand=rig.pose.bones.get('c_hand_fk'+suffix)
    if hand and 'fingers_grasp' in hand:hand['fingers_grasp']=0.
    curls={'V':{'thumb':.7,'index':0,'middle':0,'ring':1.,'pinky':1.},
           'GRIP':{f:min(v*grip,1.) for f,v in [('thumb',.65),('index',.8),('middle',1.),('ring',1.05),('pinky',1.1)]},
           'FIST':dict.fromkeys(bases,1.),'OPEN':dict.fromkeys(bases,0.)}[preset]
    props=[];touched=[]
    # Clear residual per-joint automatic rotations before applying bend_all.
    # Otherwise a "V" preset can inherit previously curled index/middle bones.
    from . import finger_pose
    for finger in bases:
        if finger not in active:continue
        for segment in (1,2,3):
            control=finger_pose.control(rig,finger,segment,side)
            if finger_pose.writable(control,rig,finger,side):
                current=control.matrix_basis.to_quaternion()
                retarget._set_rotation(control,current.slerp(Quaternion((1,0,0,0)),strength));touched.append(control.name)
    for finger,bone in bases.items():
        if finger not in active:continue
        if 'ik_fk_switch' in bone:
            bone['ik_fk_switch']=1.
            props.append((bone.name,'ik_fk_switch'))
        bone['bend_all']=float(bone['bend_all'])*(1-strength)+curls[finger]*strength
        props.append((bone.name,'bend_all'))
    if hand and 'fingers_grasp' in hand:props.append((hand.name,'fingers_grasp'))
    # Spread along the palm plane using actual rest geometry, not a guessed Euler axis.
    retarget._update(rig)
    for finger in ('index','middle') if preset=='V' else ():
        if finger not in active:continue
        control=rig.pose.bones.get('c_'+finger+'1_rot'+suffix)
        if not finger_pose.writable(control,rig,finger,side):continue
        # Presets are absolute in local channels, so repeated application is stable.
        retarget._set_rotation(control,Quaternion((1,0,0,0)))
        retarget._update(rig)
        if preset=='V':
            along=(control.tail-control.head).normalized()
            other=bases['middle' if finger=='index' else 'index']
            outward=control.head-other.tail
            outward-=along*outward.dot(along)
            if outward.length>1e-7:
                axis=along.cross(outward).normalized()
                q=Quaternion(axis,spread*strength) @ control.matrix.to_quaternion()
                retarget._set_rotation(control,retarget._to_local_rotation(control,q))
        touched.append(control.name)
    retarget._update(rig)
    return {'touched':sorted(set(touched)),'properties':props,'warnings':[], 'dependent_bones':dependent_bones(rig,side,active)}
