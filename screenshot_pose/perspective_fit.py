"""Apply and verify a calibrated perspective proposal on evaluated FK controls."""
import hashlib
from pathlib import Path
import numpy as np
from . import retarget,contact_solver,camera_data,perspective_math


def optimize(rig,mapping,settings,data):
    from . import projection_fit as legacy,fit_math
    cal=data.get('camera_calibration',{})
    width=data['image']['width'];height=data['image']['height'];K=camera_data.intrinsics(cal,width,height)
    path=Path(data['image']['path'])
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=cal.get('processed_file_sha256'):
        raise ValueError('标定所用的截图内容已变化，请重新标定')
    for name in legacy.NAMES:
        role=legacy.JOINT_ROLES[name.split('.')[0]]+'.'+name[-1]
        bone=rig.pose.bones.get(mapping.get(role,''))
        if not bone or retarget._active_rotation_constraints(bone) or retarget._driver_controls_rotation(rig,bone):
            raise ValueError('透视拟合需要完整可写的 FK 肢体')
    scales=rig.matrix_world.to_scale()
    if min(abs(v) for v in scales)<1e-7 or max(abs(v) for v in scales)/min(abs(v) for v in scales)>1.001:
        raise ValueError('透视拟合不支持骨架对象非均匀缩放')
    obs,weights=legacy.observations(settings,data);initial=legacy.points(rig,mapping)
    chains=[(0,1,2),(3,4,5),(6,7,8),(9,10,11)]
    locked=tuple(5 if side=='L' else 11 for side in contact_solver.grounded_sides(settings)) if settings.ground_feet else ()
    coordinate=bool(len(locked)==2 and getattr(settings,'coordinate_pelvis',False));root=None
    if coordinate:root=contact_solver.root_control(rig,mapping,settings.root_bone)
    fit=perspective_math.solve(initial,obs,weights,K,width,chains,locked,steps=settings.fit_iterations,regularization=settings.fit_regularization,coordinate_pelvis=coordinate)
    snapshot=retarget.snapshot_pose(rig);touched=[];warnings=[]
    ends={mapping[r+'.'+side]:rig.pose.bones[mapping[r+'.'+side]].matrix.to_quaternion().copy() for side in ('L','R') for r in ('hand','foot')}
    try:
        if fit['pose_accepted']:
            if root and np.linalg.norm(fit['pelvis_delta'])>1e-8:
                from mathutils import Vector
                before=contact_solver._world_head(rig,root).copy();delta=Vector(fit['pelvis_delta'])
                contact_solver._translate_world(rig,root,delta)
                if (contact_solver._world_head(rig,root)-before-delta).length>1e-4:raise ValueError('骨盆控制骨没有达到透视协调位移')
                touched.append(root.name)
            for side in ('L','R'):
                for roles,joints in [(('upper_arm','forearm','hand'),('shoulder','elbow','wrist')),(('thigh','shin','foot'),('hip','knee','ankle'))]:
                    bones=[rig.pose.bones[mapping[r+'.'+side]] for r in roles];ids=[legacy.NAMES.index(j+'.'+side) for j in joints]
                    q=fit['points'];legacy._aim_link(rig,bones[0],bones[1],q[ids[1]]-q[ids[0]])
                    legacy._aim_link(rig,bones[1],bones[2],q[ids[2]]-q[ids[1]])
                    touched.extend(b.name for b in bones)
            for name,q in ends.items():
                bone=rig.pose.bones[name];retarget._set_rotation(bone,retarget._to_local_rotation(bone,q));retarget._update(rig)
        actual=legacy.points(rig,mapping);pred=camera_data.project(actual,fit['camera']);error=camera_data.error(pred,obs,weights)
        drift=max((float(np.linalg.norm(actual[i]-initial[i])) for i in locked),default=0.)
        lengths=[np.linalg.norm(actual[j]-actual[i])/max(np.linalg.norm(initial[j]-initial[i]),1e-10) for a,b,c in chains for i,j in ((a,b),(b,c))]
        if drift>.002 or max(abs(v-1) for v in lengths)>.002 or error>fit['camera_only']*1.02+1e-6:
            retarget.restore_pose(rig,snapshot);actual=initial;touched=[]
            # Refit the camera only: a target binding mismatch must not be
            # represented as a successful pose update.
            fit['camera']=fit['camera_only_camera'];fit['pelvis_delta']=[0.,0.,0.]
            pred=camera_data.project(actual,fit['camera']);error=camera_data.error(pred,obs,weights);drift=0.
            warnings.append('目标绑定未达到透视姿势提案，保留原姿势并仅匹配相机')
        baseline=fit_math.solve(initial,obs,weights,chains,locked_ends=locked,steps=settings.fit_iterations,regularization=settings.fit_regularization)
        if error>max(baseline['after']*1.2,baseline['after']+.003):
            raise ValueError('透视结果明显差于原有近似，已撤回透视提案')
        return dict(target=rig.name,accepted=bool(touched),before_px=fit['before']*width,after_px=error*width,
            camera_only_px=fit['camera_only']*width,ground_drift=drift,touched=sorted(set(touched)),warnings=warnings,
            camera=fit['camera'],image=data['image'],joint_names=legacy.NAMES,observed=obs.tolist(),weights=weights.tolist(),
            before_projection=fit['baseline_predicted'].tolist(),after_projection=pred.tolist(),points=actual.tolist(),
            perspective=dict(calibration_source=cal.get('source'),estimated_not_measured=True,intrinsics_fixed=True,
                pose_accepted=bool(touched),bone_lengths_preserved=True,planted_ankles_preserved=True,
                torso_roots_fixed=not bool(np.linalg.norm(fit['pelvis_delta'])>1e-8),pelvis_delta=fit['pelvis_delta'],
                pelvis_coordination=coordinate,contact_model='existing_explicit_contacts_only'))
    except Exception:
        retarget.restore_pose(rig,snapshot);raise
