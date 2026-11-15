"""Verify a source-preserving proposal on real FK, otherwise retain the pose."""
import numpy as np
import hashlib
from pathlib import Path
from . import projection_fit,pose_data,source_fit_math as math_fit,retarget,whole_body_fit,camera_data


def refine_bent_legs(rig, mapping, actual, camera, reference, source, observed, weights, image, steps):
    proposal = math_fit.bent_leg_proposal(actual, camera, reference, source, observed, weights, image, steps)
    if proposal is None:
        return None
    snapshot = retarget.snapshot_pose(rig)
    try:
        touched = []
        for side, (hip, knee, ankle) in zip(('L','R'), math_fit.LEG_CHAINS):
            upper, lower, foot = (rig.pose.bones[mapping[role+'.'+side]] for role in ('thigh','shin','foot'))
            foot_rotation = foot.matrix.to_quaternion().copy()
            points = proposal['points']
            projection_fit._aim_link(rig, upper, lower, points[knee]-points[hip])
            projection_fit._aim_link(rig, lower, foot, points[ankle]-points[knee])
            retarget._set_rotation(foot, retarget._to_local_rotation(foot, foot_rotation))
            retarget._update(rig)
            touched.extend((upper.name, lower.name, foot.name))
        evaluated = projection_fit.points(rig, mapping)
        expected = proposal['points']
        if np.max(np.linalg.norm(evaluated - expected, axis=1)) > .002:
            retarget.restore_pose(rig, snapshot)
            return None
        new_camera = proposal['camera']
        report = math_fit.bent_leg_report(evaluated, new_camera, actual, camera, reference, observed, weights, image['width'])
        if not report['accepted']:
            retarget.restore_pose(rig, snapshot)
            return None
        report['camera_source'] = 'fitted_perspective_candidate'
        return dict(points=evaluated, camera=new_camera, report=report, touched=touched)
    except Exception:
        retarget.restore_pose(rig, snapshot)
        raise


def optimize(rig,mapping,s,data):
 for name in projection_fit.NAMES:
  role=projection_fit.JOINT_ROLES[name.split('.')[0]]+'.'+name[-1];bone=rig.pose.bones.get(mapping.get(role,''))
  if not bone or retarget._active_rotation_constraints(bone) or retarget._driver_controls_rotation(rig,bone):raise ValueError('源姿势拟合需要完整可写的 FK 肢体，请检查映射')
 scale=rig.matrix_world.to_scale()
 if min(abs(v) for v in scale)<1e-7 or max(abs(v) for v in scale)/min(abs(v) for v in scale)>1.001:raise ValueError('源姿势拟合暂不支持骨架对象非均匀缩放')
 initial=projection_fit.points(rig,mapping);snapshot=retarget.snapshot_pose(rig)
 joints,_=pose_data.to_joints(data,'WORLD');source=np.array([joints[n] for n in projection_fit.NAMES])
 obs,raw_weights=projection_fit.observations(s,data);weights=whole_body_fit.effective_weights(data,raw_weights)
 explicit=bool(s.contact_enabled and any(v!='FREE' for v in (s.support_l,s.support_r,s.hand_contact_l,s.hand_contact_r)))
 warnings=[];fit=None
 if s.fit_model=='PERSPECTIVE':
  try:
   cal=data.get('camera_calibration',{});K=camera_data.intrinsics(cal,data['image']['width'],data['image']['height'])/data['image']['width']
   path=Path(data['image']['path'])
   if hashlib.sha256(path.read_bytes()).hexdigest()!=cal.get('processed_file_sha256'):raise ValueError('标定截图已变化')
   fit=math_fit.perspective_camera_only(initial,source,obs,weights,K,s.fit_iterations)
  except (ValueError,OSError) as exc:warnings.append('透视标定未采用，保留源姿势并使用弱透视：'+str(exc))
 if fit is None:fit=math_fit.solve(initial,source,obs,weights,s.fit_iterations,camera_only=bool(s.ground_feet or explicit))
 touched=[]
 ends={mapping[r+'.'+side]:rig.pose.bones[mapping[r+'.'+side]].matrix.to_quaternion().copy() for r in ('hand','foot') for side in ('L','R')}
 reason='camera_only_no_clear_articulation_gain'
 try:
  if fit['pose_accepted']:
   for side in ('L','R'):
    for roles,names in [(('upper_arm','forearm','hand'),('shoulder','elbow','wrist')),(('thigh','shin','foot'),('hip','knee','ankle'))]:
     bones=[rig.pose.bones[mapping[r+'.'+side]] for r in roles];ids=[projection_fit.NAMES.index(n+'.'+side) for n in names];p=fit['points']
     projection_fit._aim_link(rig,bones[0],bones[1],p[ids[1]]-p[ids[0]])
     projection_fit._aim_link(rig,bones[1],bones[2],p[ids[2]]-p[ids[1]]);touched.extend(b.name for b in bones)
   for name,q in ends.items():retarget._set_rotation(rig.pose.bones[name],retarget._to_local_rotation(rig.pose.bones[name],q));retarget._update(rig)
  actual=projection_fit.points(rig,mapping);cfg=fit['camera'];pred=camera_data.project(actual,cfg);error=camera_data.error(pred,obs,weights);angles=math_fit.deviation(actual,initial)
  links=[(a,b) for r,m,e in math_fit.CHAINS for a,b in ((r,m),(m,e))]
  length=max(abs(np.linalg.norm(actual[b]-actual[a])/max(np.linalg.norm(initial[b]-initial[a]),1e-9)-1) for a,b in links)
  if max(angles)>8.5 or length>.002 or (touched and error>=fit['camera_only']-.0015):
   retarget.restore_pose(rig,snapshot);actual=initial;touched=[]
   cfg=fit['baseline_camera'];pred=camera_data.project(initial,cfg);error=camera_data.error(pred,obs,weights);reason='actual_fk_guard_restored_source'
  elif touched:reason='bounded_articulation_gain_verified'
  leg_refinement=None
  if s.fit_model=='ORTHO' and data.get('source')=='sam3d_mhr' and not s.ground_feet and not explicit:
   try:leg_refinement=refine_bent_legs(rig,mapping,actual,cfg,initial,source,obs,weights,data['image'],s.fit_iterations)
   except (ValueError,FloatingPointError,np.linalg.LinAlgError):pass
   if leg_refinement:
    actual=leg_refinement['points'];cfg=leg_refinement['camera'];pred=camera_data.project(actual,cfg)
    error=camera_data.error(pred,obs,weights);touched.extend(leg_refinement['touched'])
    reason='bent_leg_perspective_gain_verified'
  guard=dict(enabled=True,accepted=bool(touched),reason=reason,camera_bound_deg=3.,max_link_change_deg=max(math_fit.deviation(actual,initial)),
   camera_anchor=fit['source_camera_anchor'],reference_points=initial.tolist(),torso_preserved=True,explicit_support_priority=bool(s.ground_feet or explicit))
  width=data['image']['width']
  return dict(target=rig.name,accepted=bool(touched),before_px=fit['before']*width,after_px=error*width,camera_only_px=fit['camera_only']*width,
   ground_drift=0.,touched=sorted(set(touched)),warnings=warnings,source_guard=guard,
   leg_refinement=leg_refinement['report'] if leg_refinement else None,camera=cfg,image=data['image'],joint_names=projection_fit.NAMES,
   observed=obs.tolist(),weights=weights.tolist(),before_projection=fit['baseline_predicted'].tolist(),after_projection=pred.tolist(),points=actual.tolist())
 except Exception:retarget.restore_pose(rig,snapshot);raise
