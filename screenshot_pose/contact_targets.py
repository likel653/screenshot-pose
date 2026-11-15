"""Explicit world-space supports, solved after image fitting and orientation.

    Targets are user choices, never inferred scene geometry or selected objects.
    Only mapped, writable FK chains are changed; no persistent rig constraints.
"""
import importlib, json, math
import bpy
from bpy.types import Operator
from bpy.props import EnumProperty
from mathutils import Matrix, Vector
from . import retarget, contact_solver, foot_pose, orientation, hand_presets, contact_math

def addon():return importlib.import_module(__package__)

def refresh_fit(rig,mapping,fit):
    """Keep the already-shown camera; diagnose the final constrained pose."""
    import numpy as np
    a=addon();points=a.projection_fit.points(rig,mapping);cfg=fit['camera']
    from . import camera_data
    pred=camera_data.project(points,cfg)
    weights=np.asarray(fit['weights']);observed=np.asarray(fit['observed'])
    fit['before_contacts_px']=fit['after_px']
    fit['after_px']=float(np.sqrt((((pred-observed)**2)*weights[:,None]).sum()/max(weights.sum(),1e-8))*fit['image']['width'])
    fit['after_projection']=pred.tolist();fit['points']=points.tolist();fit['contact_constrained']=True

def target_object(rig, obj):
    if obj is None:return None
    pending=[obj];seen=set()
    while pending:
        current=pending.pop()
        if current==rig or current==rig.data:raise ValueError('接触目标不能依附或通过约束/驱动跟随当前角色，请选择独立目标')
        if not isinstance(current,bpy.types.Object) or current.as_pointer() in seen:continue
        seen.add(current.as_pointer())
        if current.parent:pending.append(current.parent)
        for modifier in current.modifiers:
            if modifier.type=='ARMATURE' and modifier.object:pending.append(modifier.object)
        for constraint in current.constraints:
            if constraint.mute or getattr(constraint,'influence',1.)<=.001:continue
            for field in ('target','pole_target','object'):
                value=getattr(constraint,field,None)
                if value is not None:pending.append(value)
            for value in getattr(constraint,'targets',[]):
                if getattr(value,'target',None):pending.append(value.target)
        if current.animation_data:
            for driver in current.animation_data.drivers:
                if driver.mute:continue
                for variable in driver.driver.variables:
                    for target in variable.targets:
                        if target.id:pending.append(target.id)
    if not all(math.isfinite(v) for row in obj.matrix_world for v in row):raise ValueError('接触目标的变换无效')
    return obj

def chain(rig,mapping,kind,side):
    roles=('thigh','shin','foot') if kind=='FOOT' else ('upper_arm','forearm','hand')
    bones=[rig.pose.bones.get(mapping.get(r+'.'+side,'')) for r in roles]
    if not all(bones) or not all(orientation._safe(b,rig) for b in bones):
        raise ValueError(side+('腿' if kind=='FOOT' else '臂')+'缺少可写的完整 FK 控制链，请检查映射和约束')
    return bones

def palm(rig,mapping,s,side):
    cal=orientation.calibration(rig,mapping,s)['hands'].get(side)
    if not cal:raise ValueError(side+'手缺少食指、小指根部校准，无法定位掌心')
    hand=rig.pose.bones[mapping['hand.'+side]]
    index=rig.pose.bones[cal['index']].bone.head_local
    pinky=rig.pose.bones[cal['pinky']].bone.head_local
    center=(hand.bone.head_local+(index+pinky)*.5)*.5
    local=hand.bone.matrix_local.inverted_safe()@center
    physical=hand.matrix.to_3x3().normalized()@hand.bone.matrix_local.to_3x3().normalized().transposed()@Matrix(cal['frame'])
    return cal,local,physical

def solve_chain(rig,bones,target,pole,rotation):
    upper,lower,end=bones
    hip,knee,ankle=[rig.matrix_world@b.head for b in bones]
    l1,l2=(knee-hip).length,(ankle-knee).length
    for _ in range(3):
        hip=rig.matrix_world@upper.head
        knee,effective,_=contact_solver.solve_two_bone(hip,target,pole,l1,l2)
        contact_solver._aim(rig,upper,knee-hip)
        contact_solver._aim(rig,lower,effective-(rig.matrix_world@lower.head))
        orientation._absolute(rig,end,rotation)
    return (rig.matrix_world@end.head-target).length

def prepare(rig,mapping,s,scope='ALL',calibration=None):
    """Freeze independent targets before moving the root; never write channels."""
    if not s.contact_enabled or s.strength==0:return []
    if not s.switch_fk:raise ValueError('独立接触求解需要开启自动切换 FK')
    if abs(s.strength-1.)>1e-6:raise ValueError('独立接触目标需要姿势强度为 100%，避免部分混合破坏支撑')
    scales=rig.matrix_world.to_scale()
    if min(abs(v) for v in scales)<1e-7 or max(abs(v) for v in scales)/min(abs(v) for v in scales)>1.001:
        raise ValueError('独立接触暂不支持骨架对象的非均匀缩放，请先校准对象缩放')
    # Moving an Empty changes its channels before matrix_world is evaluated.
    retarget._update(rig)
    selected=lambda kind,side:scope=='ALL' or scope==kind+'_'+side
    jobs=[]
    cal=calibration if calibration is not None else foot_pose.sole_calibration(rig,mapping,s.shoe_mesh.name if s.shoe_mesh else '')
    # Validate everything before writing any channels.
    for side in ('L','R'):
        suffix=side.lower();mode=getattr(s,'support_'+suffix)
        if selected('FOOT',side) and mode!='FREE':
            bones=chain(rig,mapping,'FOOT',side);state=cal.get(side)
            if not state:raise ValueError(side+'脚未找到鞋底参考，请指定鞋子网格或校准角色')
            obj=target_object(rig,getattr(s,'support_target_'+suffix))
            height=obj.matrix_world.translation.z if obj else s.floor_height if mode=='FLOOR' else getattr(s,'support_height_'+suffix)
            desired=(rig.matrix_world@bones[-1].matrix).translation.copy()
            if obj:desired.x=obj.matrix_world.translation.x;desired.y=obj.matrix_world.translation.y
            # Level the actual shoe sole, retaining the shoe's rest heel pitch.
            forward=(rig.matrix_world.to_quaternion().to_matrix()@bones[-1].matrix.to_3x3()@bones[-1].bone.matrix_local.to_3x3().inverted()@Matrix(state['frame'])).col[1]
            forward.z=0
            if forward.length<1e-6:raise ValueError(side+'脚水平朝向不确定，无法建立支撑面')
            physical=Matrix((forward.cross(Vector((0,0,1))).normalized(),forward.normalized(),Vector((0,0,1)))).transposed()
            rotation=(rig.matrix_world.to_quaternion().inverted().to_matrix()@physical@Matrix(state['frame']).inverted()@bones[-1].bone.matrix_local.to_3x3()).to_quaternion()
            transform=rig.matrix_world.to_3x3()@rotation.to_matrix()
            desired.z=height-min((transform@Vector(p)).z for p in state['anchors'])
            pole_obj=target_object(rig,getattr(s,'knee_target_'+suffix))
            pole=pole_obj.matrix_world.translation.copy() if pole_obj else rig.matrix_world@bones[1].head
            jobs.append(dict(kind='FOOT',side=side,bones=bones,target=desired,pole=pole,rotation=rotation,height=height,state=state,explicit_pole=bool(pole_obj),object=obj))
        mode=getattr(s,'hand_contact_'+suffix)
        if selected('HAND',side) and mode!='FREE':
            bones=chain(rig,mapping,'HAND',side)
            obj=target_object(rig,getattr(s,'hand_target_'+suffix))
            if not obj:raise ValueError(side+'手请先选择或创建掌心目标')
            hand_cal,offset,physical=palm(rig,mapping,s,side)
            desired_frame=obj.matrix_world.to_quaternion().to_matrix()
            rotation=(rig.matrix_world.to_quaternion().inverted().to_matrix()@desired_frame@Matrix(hand_cal['frame']).inverted()@bones[-1].bone.matrix_local.to_3x3()).to_quaternion()
            wrist=obj.matrix_world.translation-rig.matrix_world.to_3x3()@rotation.to_matrix()@offset
            jobs.append(dict(kind='HAND',side=side,bones=bones,target=wrist,pole=rig.matrix_world@bones[1].head,rotation=rotation,mode=mode,palm_offset=offset,object=obj))
    return jobs


def apply(rig,mapping,s,data,scope='ALL',calibration=None,prepared=None,move_root=True):
    report=dict(touched=[],properties=[],warnings=[],feet={},hands={},pelvis_delta_z=0.)
    if not s.contact_enabled or s.strength==0:return report
    jobs=prepare(rig,mapping,s,scope,calibration) if prepared is None else prepared
    before=retarget.snapshot_pose(rig)
    try:
        selected_map={r:n for r,n in mapping.items() if any(n==b.name for job in jobs for b in job['bones'])}
        for bone,key,value in retarget._fk_switches(rig,selected_map):bone[key]=value;report['properties'].append((bone.name,key))
        retarget._update(rig)
        feet=[j for j in jobs if j['kind']=='FOOT']
        # Root changes are full-pose only; scoped edits keep other limbs intact.
        if feet and scope=='ALL' and s.support_move_pelvis and move_root:
            legs=[]
            for job in feet:
                p=[rig.matrix_world@b.head for b in job['bones']]
                legs.append((p[0],job['target'],(p[1]-p[0]).length,(p[2]-p[1]).length))
            shift,reachable=contact_math.pelvis_shift(legs)
            if abs(shift)>s.support_pelvis_limit:
                shift=max(-s.support_pelvis_limit,min(s.support_pelvis_limit,shift));report['warnings'].append('支撑所需骨盆位移超过上限，仅做有界调整')
            if not reachable:report['warnings'].append('各脚支撑目标不同时可达；请调整支撑点或骨盆')
            if abs(shift)>1e-7:
                root=contact_solver.root_control(rig,mapping,s.root_bone)
                contact_solver._translate_world(rig,root,Vector((0,0,shift)))
                report['touched'].append(root.name);report['pelvis_delta_z']=shift
        for job in jobs:
            bones=job['bones'];side=job['side']
            pole=job['pole']
            if not job.get('explicit_pole'):pole=pole+Vector((0,0,report['pelvis_delta_z']))
            error=solve_chain(rig,bones,job['target'],pole,job['rotation'])
            report['touched'].extend(b.name for b in bones)
            angle=math.degrees(bones[-1].matrix.to_quaternion().rotation_difference(job['rotation']).angle)
            if error>.003:report['warnings'].append(side+('脚' if job['kind']=='FOOT' else '手')+f'接触目标不可达或受绑定影响，位置残差 {error:.4f}')
            if angle>3:report['warnings'].append(side+'侧接触朝向受绑定影响，请检查控制链')
            if job['kind']=='FOOT':
                t=rig.matrix_world@bones[-1].matrix
                clearance=min((t@Vector(p)).z-job['height'] for p in job['state']['anchors'])
                report['feet'][side]=dict(height=job['height'],ankle_target=list(job['target']),position_error=error,sole_clearance=clearance,rotation_error_deg=angle)
            else:
                actual=rig.matrix_world@bones[-1].matrix@job['palm_offset']
                report['hands'][side]=dict(mode=job['mode'],target=job['object'].name,palm_error=(actual-job['object'].matrix_world.translation).length,position_error=error,rotation_error_deg=angle)
                if report['hands'][side]['palm_error']>.003 and error<=.003:report['warnings'].append(side+'手掌心未到达接触目标，请检查目标动画或约束')
                # Contact is a user-selected structural prior. Manual fingers win.
                try:
                    preset=hand_presets.apply(rig,side,'OPEN' if job['mode']=='SURFACE' else 'GRIP',1.,s.hand_spread,getattr(s,'hand_grip_'+side.lower()),hand_presets.manual_fingers(data,side))
                    for key in ('touched','properties','warnings'):report[key].extend(preset[key])
                except ValueError:
                    report['warnings'].append(side+'手绑定不支持接触手势预设，已保留识别指骨；掌心接触仍已求解')
        report['touched']=sorted(set(report['touched']))
        report['properties']=list(set(tuple(p) for p in report['properties']))
        return report
    except Exception:
        retarget.restore_pose(rig,before);raise

class SP_OT_contact_target(Operator):
    bl_idname='screenshot_pose.contact_target';bl_label='创建接触目标';bl_options={'REGISTER','UNDO'}
    endpoint:EnumProperty(items=[(k,k,'') for k in ('FOOT_L','FOOT_R','HAND_L','HAND_R')])
    def execute(self,context):
        s=context.scene.screenshot_pose;a=addon();obj=None
        try:
            if not s.target or s.candidate_state:raise ValueError('请先选择角色并结束候选预览')
            mapping=a.pose_tools.mapping_for(s);rig=s.target;kind,side=self.endpoint.split('_');suffix=side.lower()
            bones=chain(rig,mapping,kind,side)
            if kind=='HAND':
                cal,offset,physical=palm(rig,mapping,s,side)
                matrix=(rig.matrix_world.to_quaternion().to_matrix()@physical).to_4x4()
                matrix.translation=rig.matrix_world@bones[-1].matrix@offset
            else:
                state=foot_pose.sole_calibration(rig,mapping,s.shoe_mesh.name if s.shoe_mesh else '').get(side)
                if not state:raise ValueError('未找到鞋底参考，请指定鞋子网格')
                transform=rig.matrix_world@bones[-1].matrix;matrix=Matrix.Identity(4)
                pos=transform.translation.copy();pos.z=min((transform@Vector(p)).z for p in state['anchors']);matrix.translation=pos
            obj=bpy.data.objects.new('接触目标 · '+('左' if side=='L' else '右')+('脚支撑面' if kind=='FOOT' else '掌心'),None)
            context.scene.collection.objects.link(obj);obj.matrix_world=matrix;obj.empty_display_type='ARROWS';obj.empty_display_size=.10;obj.show_in_front=True
            obj['sp_contact_target']=True
            setattr(s,('support_target_' if kind=='FOOT' else 'hand_target_')+suffix,obj)
            s.status='已创建独立目标；移动调整位置，掌心目标可旋转（Y 为指向，Z 为掌面法向）'
            return {'FINISHED'}
        except Exception as exc:
            if obj:bpy.data.objects.remove(obj,do_unlink=True)
            self.report({'ERROR'},str(exc));return {'CANCELLED'}

class SP_OT_contact_apply(Operator):
    bl_idname='screenshot_pose.contact_apply';bl_label='仅更新接触';bl_options={'REGISTER','UNDO'}
    scope:EnumProperty(items=[(k,k,'') for k in ('ALL','FOOT_L','FOOT_R','HAND_L','HAND_R')])
    def execute(self,context):
        a=addon();s=context.scene.screenshot_pose;rig=s.target;before=None;created=None
        old_result=s.last_result;tracks={}
        try:
            if s.candidate_state:raise ValueError('请先结束候选预览')
            rig,mapping,data=a.pose_tools.check(s);before=retarget.snapshot_pose(rig)
            if not s.contact_enabled:raise ValueError('请先开启独立手脚接触')
            if rig.animation_data:
                tracks={t.name:t.mute for t in rig.animation_data.nla_tracks}
                if rig.animation_data.action:rig.animation_data.action.use_fake_user=True
                rig.animation_data.action=None
                for t in rig.animation_data.nla_tracks:t.mute=True
            frozen=dict(before,action=None);retarget.restore_pose(rig,frozen)
            report=apply(rig,mapping,s,data,self.scope)
            solved=retarget.snapshot_pose(rig)
            retarget.restore_pose(rig,before)
            if s.keyframe and report['touched']:
                created=a.pose_tools.copy_action(rig,before)
            action=rig.animation_data.action if rig.animation_data else None
            slot=getattr(rig.animation_data,'action_slot',None) if rig.animation_data else None
            solved.update(action=action.name if action else None,action_slot_handle=slot.handle if slot else None,action_fake_user=bool(action.use_fake_user) if action else False)
            retarget.restore_pose(rig,solved,update=not s.keyframe)
            if s.keyframe and report['touched']:
                contact_solver.key_touched(rig,report['touched'],context.scene.frame_current,report['properties'])
                retarget._update(rig)
            result=json.loads(s.last_result) if s.last_result else {}
            result['contacts']=report
            if result.get('fit') and report['touched']:refresh_fit(rig,mapping,result['fit'])
            s.last_result=json.dumps(result,ensure_ascii=False);s.status='接触已更新'+('；部分目标有残差，请查看结果' if report['warnings'] else '')
            return {'FINISHED'}
        except Exception as exc:
            if before:retarget.restore_pose(rig,before)
            s.last_result=old_result
            if created:
                created.use_fake_user=False
                if created.users==0:bpy.data.actions.remove(created)
            self.report({'ERROR'},str(exc));return {'CANCELLED'}
        finally:
            if rig and rig.animation_data:
                for t in rig.animation_data.nla_tracks:
                    if t.name in tracks:t.mute=tracks[t.name]

OPERATORS=(SP_OT_contact_target,SP_OT_contact_apply)
