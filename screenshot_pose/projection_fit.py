"""Fit measured target-rig limb geometry, then verify the evaluated Blender result."""
import json,math
import numpy as np
import bpy
from mathutils import Matrix,Vector
from . import fit_math,retarget,preview,pose_data,contact_solver
JOINT_ROLES={'shoulder':'upper_arm','elbow':'forearm','wrist':'hand','hip':'thigh','knee':'shin','ankle':'foot'}
NAMES=[j+'.'+side for side in ('L','R') for j in JOINT_ROLES]

def points(rig,mapping):
    return np.array([tuple(rig.matrix_world@rig.pose.bones[mapping[JOINT_ROLES[n.split('.')[0]]+'.'+n[-1]]].head) for n in NAMES])

def observations(settings,data):
    if settings.coordinate_mode not in ('HYBRID','IMAGE') or settings.mirror or abs(settings.yaw)>1e-6:
        raise ValueError('贴合原图需要图片/混合模式、关闭镜像且朝向修正为零，请重建关节点')
    p=data['landmarks'];aspect=data['image']['height']/data['image']['width']
    indices=[pose_data.JOINT_INDICES[n] for n in NAMES]
    weights=np.array([min(p[i].get('visibility',1),p[i].get('presence',1)) for i in indices])
    weights[weights<settings.threshold]=0
    if getattr(settings,'fit_video_observations',False):
        return np.array([[p[i]['x'],p[i]['y']*aspect] for i in indices]),weights
    canonical,_=pose_data.to_joints(data,settings.coordinate_mode,settings.depth)
    # Recover image-units-per-guide-unit; current local marker edits are respected.
    a,b=canonical['shoulder.L'],canonical['shoulder.R']
    screen=math.hypot(p[11]['x']-p[12]['x'],(p[11]['y']-p[12]['y'])*aspect)
    factor=math.hypot(a[0]-b[0],a[2]-b[2])/max(screen,1e-8)
    if factor<1e-7:raise ValueError('肩部关节点重合，无法建立图片投影')
    cx=(p[23]['x']+p[24]['x'])/2;cy=(p[23]['y']+p[24]['y'])/2*aspect
    guides={o['sp_joint']:o.location for o in preview.marker_objects(settings)}
    obs=np.array([[guides[n].x/factor+cx,-guides[n].z/factor+cy] for n in NAMES])
    return obs,weights

def _aim_link(rig,bone,child,direction):
    local=rig.matrix_world.inverted_safe().to_3x3()@Vector(direction)
    actual=child.head-bone.head
    if min(actual.length,local.length)<1e-7:raise ValueError('映射骨骼关节重合')
    q=actual.rotation_difference(local)@bone.matrix.to_quaternion()
    retarget._set_rotation(bone,retarget._to_local_rotation(bone,q));retarget._update(rig)

def optimize(rig,mapping,settings,data,*,upper_only=False):
    if not upper_only and getattr(settings,'fit_model','ORTHO')=='PERSPECTIVE':
        from . import perspective_fit
        try:return perspective_fit.optimize(rig,mapping,settings,data)
        except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
            from .pose_tools import SettingsProxy
            result=optimize(rig,mapping,SettingsProxy(settings,fit_model='ORTHO'),data)
            result['warnings'].append('透视拟合未采用，保留原拟合：'+str(exc))
            result['perspective']={'fallback':True,'reason':str(exc)}
            return result
    for n in NAMES:
        role=JOINT_ROLES[n.split('.')[0]]+'.'+n[-1]
        if role not in mapping or mapping[role] not in rig.pose.bones:raise ValueError('贴合原图需要完整的双臂及双腿映射')
        pb=rig.pose.bones[mapping[role]]
        if (not upper_only or n.split('.')[0] in ('shoulder','elbow','wrist')) and (retarget._active_rotation_constraints(pb) or retarget._driver_controls_rotation(rig,pb)):raise ValueError('贴合原图需要可直接控制的 FK 骨骼')
    scales=rig.matrix_world.to_scale()
    if min(abs(v) for v in scales)<1e-7 or max(abs(v) for v in scales)/min(abs(v) for v in scales)>1.001:
        raise ValueError('贴合原图暂不支持骨架对象的非均匀缩放')
    obs,weights=observations(settings,data);retarget._update(rig);initial=points(rig,mapping)
    chains=[(0,1,2),(3,4,5),(6,7,8),(9,10,11)]
    if upper_only:
        weights[[4,5,10,11]]=0
        chains=[chains[0],chains[2]]
    locked=tuple(5 if side=='L' else 11 for side in contact_solver.grounded_sides(settings)) if settings.ground_feet else ()
    fitted=fit_math.solve(initial,obs,weights,chains,locked_ends=locked,steps=settings.fit_iterations,regularization=settings.fit_regularization)
    coordinated=None;coordination_error=None
    if len(locked)==2 and getattr(settings,'coordinate_pelvis',False):
        try:coordinated=fit_math.solve_grounded(initial,obs,weights,steps=settings.fit_iterations,regularization=settings.fit_regularization)
        except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:coordination_error=str(exc)
    snapshot=retarget.snapshot_pose(rig);touched=[]
    # Keep hand and foot world orientation after moving their upstream chains.
    limbs=[(('upper_arm','forearm','hand'),('shoulder','elbow','wrist'))]
    if not upper_only:limbs.append((('thigh','shin','foot'),('hip','knee','ankle')))
    ends={mapping[roles[-1]+'.'+side]:rig.pose.bones[mapping[roles[-1]+'.'+side]].matrix.to_quaternion().copy() for roles,names in limbs for side in ('L','R')}
    try:
        attempts=[];best=None
        options=[('fixed_torso',fitted)]
        if coordinated and coordinated['coordination']['accepted']:options.append(('coordinated_pelvis',coordinated))
        for kind,candidate in options:
            retarget.restore_pose(rig,snapshot);changed=[]
            try:
                if kind=='coordinated_pelvis':
                    root=contact_solver.root_control(rig,mapping,settings.root_bone)
                    origin=contact_solver._world_head(rig,root).copy();delta=Vector(candidate['pelvis_delta'])
                    contact_solver._translate_world(rig,root,delta)
                    if (contact_solver._world_head(rig,root)-origin-delta).length>1e-4:raise ValueError('骨盆未按预期移动，已回退原方案')
                    changed.append(root.name)
                for side in ('L','R'):
                    for roles,names in limbs:
                        bones=[rig.pose.bones[mapping[r+'.'+side]] for r in roles];idx=[NAMES.index(n+'.'+side) for n in names];P=candidate['points']
                        _aim_link(rig,bones[0],bones[1],P[idx[1]]-P[idx[0]])
                        _aim_link(rig,bones[1],bones[2],P[idx[2]]-P[idx[1]])
                        changed.extend(b.name for b in bones)
                for name,q in ends.items():
                    pb=rig.pose.bones[name];retarget._set_rotation(pb,retarget._to_local_rotation(pb,q));retarget._update(rig)
                actual=points(rig,mapping);pred,camera,error=fit_math.project(actual,obs,weights,candidate['camera']['angles'])
                drift=max((np.linalg.norm(actual[i]-initial[i]) for i in locked),default=0.)
                flex=[] if upper_only else [math.degrees(fit_math.knee_flex(actual[r],actual[m],actual[e])) for r,m,e in (chains[1],chains[3])]
                valid=bool(error<fitted['before']-1e-6 and drift<.002 and (kind!='coordinated_pelvis' or max(flex)<=170.01))
                attempts.append(dict(kind=kind,projection_px=error*data['image']['width'],foot_drift=drift,knee_flex_deg=flex,valid=valid))
                if valid and (best is None or error<best['error']*.995):
                    best=dict(kind=kind,fit=candidate,error=error,drift=drift,actual=actual,pred=pred,camera=camera,touched=changed,snapshot=retarget.snapshot_pose(rig))
            except Exception as exc:
                if kind=='fixed_torso':raise
                attempts.append(dict(kind=kind,valid=False,error=str(exc)))
        accepted=best is not None
        warnings=['骨盆协调不可用，保留原拟合：'+coordination_error] if coordination_error else []
        if accepted:
            retarget.restore_pose(rig,best['snapshot']);actual=best['actual'];pred=best['pred'];camera=best['camera'];error=best['error'];drift=best['drift'];touched=best['touched']
        else:
            retarget.restore_pose(rig,snapshot);actual=initial
            pred,camera,error=fit_math.project(initial,obs,weights)
            drift=0.;touched=[];warnings.append('本次拟合未改善实际骨架投影或破坏接地，已保留原姿势')
        width=data['image']['width']
        return dict(target=rig.name,accepted=accepted,before_px=fitted['before']*width,after_px=error*width,upper_only=upper_only,
                    camera_only_px=fitted['camera_only']*width,ground_drift=drift,touched=touched,warnings=warnings,
                    coordination=dict(enabled=bool(settings.ground_feet and getattr(settings,'coordinate_pelvis',False)),error=coordination_error,chosen=best['kind'] if best else 'initial',attempts=attempts,
                                      proposed=coordinated.get('coordination') if coordinated else None,
                                      pelvis_delta=best['fit'].get('pelvis_delta',[0,0,0]) if best else [0,0,0]),
                    camera=camera,image=data['image'],joint_names=NAMES,observed=obs.tolist(),weights=weights.tolist(),
                    before_projection=fitted['baseline_predicted'].tolist(),after_projection=pred.tolist(),points=actual.tolist())
    except Exception:
        retarget.restore_pose(rig,snapshot);raise

def create_camera(context,settings):
    result=json.loads(settings.last_result).get('fit')
    if not result:raise ValueError('请先勾选贴合原图并应用姿势')
    if result.get('target') and (not settings.target or result['target']!=settings.target.name):
        raise ValueError('角色已变更，请重新应用拟合')
    cfg=result['camera'];width=result['image']['width'];height=result['image']['height'];aspect=height/width
    if not settings.camera_backup:
        r=context.scene.render
        settings.camera_backup=json.dumps(dict(camera=context.scene.camera.name if context.scene.camera else '',x=r.resolution_x,y=r.resolution_y,percentage=r.resolution_percentage,pixel_x=r.pixel_aspect_x,pixel_y=r.pixel_aspect_y))
    camera=bpy.data.objects.get(settings.fit_camera)
    if camera is None or camera.type!='CAMERA' or not camera.get('sp_fit_camera'):
        camera=bpy.data.objects.new('Screenshot Pose · 匹配相机',bpy.data.cameras.new('Screenshot Pose Match'))
        context.scene.collection.objects.link(camera);camera['sp_fit_camera']=True;settings.fit_camera=camera.name
    # In fitted camera coordinates screen=(scale*x+tx, -scale*z+ty).
    R=Matrix(cfg['rotation'])
    # Camera local X=screen right, Y=up, -Z=forward (+world Y at identity).
    basis=Matrix(((1,0,0),(0,0,-1),(0,1,0)))
    camera.rotation_euler=(R.transposed()@basis).to_euler()
    if cfg.get('type')=='PERSPECTIVE':
        fx,fy,cx,cy=cfg['normalized_intrinsics']
        camera.location=R.transposed()@(-Vector(cfg['translation']))
        camera.data.type='PERSP';camera.data.sensor_fit='HORIZONTAL';camera.data.sensor_width=36.;camera.data.lens=fx*36.
        camera.data.shift_x=.5-cx;camera.data.shift_y=(cy-aspect/2)*fx/fy
        camera.data.clip_start=.001;camera.data.clip_end=1000.
        context.scene.camera=camera;r=context.scene.render;r.resolution_x=width;r.resolution_y=height;r.resolution_percentage=100
        ratio=fy/fx
        r.pixel_aspect_x=max(1.,ratio);r.pixel_aspect_y=max(1.,1/ratio)
        return camera
    scale=cfg['scale'];tx,ty=cfg['shift']
    mid=Vector(((.5-tx)/scale,0,(ty-aspect/2)/scale))
    cloud=np.array(result['points']);depth=float(np.min(cloud@np.array(cfg['rotation'])[1]))-max(5.,2/scale)
    mid.y=depth;camera.location=R.transposed()@mid
    camera.data.type='ORTHO';camera.data.sensor_fit='HORIZONTAL';camera.data.ortho_scale=1/scale
    camera.data.shift_x=0;camera.data.shift_y=0;camera.data.clip_start=.01;camera.data.clip_end=max(1000.,20/scale)
    context.scene.camera=camera;r=context.scene.render;r.resolution_x=width;r.resolution_y=height;r.resolution_percentage=100;r.pixel_aspect_x=1;r.pixel_aspect_y=1
    return camera

def restore_camera(context,settings):
    if settings.camera_backup:
        b=json.loads(settings.camera_backup);context.scene.camera=bpy.data.objects.get(b['camera'])
        r=context.scene.render;r.resolution_x=b['x'];r.resolution_y=b['y'];r.resolution_percentage=b['percentage'];r.pixel_aspect_x=b['pixel_x'];r.pixel_aspect_y=b['pixel_y'];settings.camera_backup=''
