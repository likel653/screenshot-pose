"""Bounded image fitting around an evaluated full-source pose, never GT."""
import numpy as np
try:from . import fit_math,camera_data,whole_body_math
except ImportError:import fit_math,camera_data,whole_body_math
CHAINS=whole_body_math.CHAINS
LEG_CHAINS=(CHAINS[1],CHAINS[3])
LEG_JOINTS=(4,5,10,11)
UPPER_JOINTS=(0,1,2,6,7,8)
def rotation_vector(R):
 R=np.asarray(R,float);angle=float(np.arccos(np.clip((np.trace(R)-1)/2,-1,1)));skew=np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])
 if angle<1e-7:return skew*.5
 if np.pi-angle<1e-5:
  _,vectors=np.linalg.eigh((R+R.T)*.5);return vectors[:,-1]*angle
 return skew*angle/(2*np.sin(angle))
def anchor_rotation(points,source):
 p=np.asarray(points,float);q=np.asarray(source,float)
 if p.shape!=(12,3) or q.shape!=(12,3) or not np.isfinite(p).all() or not np.isfinite(q).all():raise ValueError('源姿势相机参考无效')
 u,_,vt=np.linalg.svd((p-p.mean(0)).T@(q-q.mean(0)));d=np.eye(3);d[-1,-1]=np.linalg.det(u@vt)
 return (u@d@vt).T
def camera(points,observed,weights,rotation):
 pred,cfg,error=fit_math.project(np.asarray(points)@rotation.T,observed,weights)
 cfg['rotation']=rotation.tolist();cfg['angles']=rotation_vector(rotation).tolist()
 return pred,cfg,error
def deviation(points,reference):
 p=np.asarray(points);r=np.asarray(reference);angles=[]
 for a,b,c in CHAINS:
  for u,v in ((a,b),(b,c)):
   x=p[v]-p[u];y=r[v]-r[u];angles.append(float(np.degrees(np.arccos(np.clip(x@y/max(np.linalg.norm(x)*np.linalg.norm(y),1e-10),-1,1)))))
 return angles
def perspective_camera_only(points,source,observed,weights,intrinsics,steps=40):
 p=np.asarray(points,float);obs=np.asarray(observed,float);w=np.asarray(weights,float);K=np.asarray(intrinsics,float)
 if obs.shape!=(12,2) or w.shape!=(12,) or K.shape!=(4,) or not all(np.isfinite(v).all() for v in (obs,w,K)) or (w<0).any() or (w>.1).sum()<6 or min(K[:2])<=0:raise ValueError('透视源姿势拟合数据无效')
 anchor=anchor_rotation(p,source);center=p.mean(0);span=max(float(np.linalg.norm(np.ptp(p,axis=0))),.01)
 _,weak,_=camera(p,obs,w,anchor);depth=max(K[0]/weak['scale'],span*1.3);mean=(obs*w[:,None]).sum(0)/w.sum()
 base=np.array([(mean[0]-K[2])*depth/K[0],depth,-(mean[1]-K[3])*depth/K[1]])
 def evaluate(x):
  R=fit_math.rotation(x[:3])@anchor;t=base.copy();t[[0,2]]+=x[[3,5]]*span;t[1]*=np.exp(x[4]);translation=t-R@center
  q=p@R.T+translation;y=np.maximum(q[:,1],1e-4);pred=np.column_stack((K[0]*q[:,0]/y+K[2],-K[1]*q[:,2]/y+K[3]))
  cfg=dict(type='PERSPECTIVE',rotation=R.tolist(),translation=translation.tolist(),normalized_intrinsics=K.tolist(),angles=rotation_vector(R).tolist())
  return pred,cfg,camera_data.error(pred,obs,w),np.maximum(1e-4-q[:,1],0)/span
 def residual(x):
  pred,_,_,behind=evaluate(x);return np.r_[((pred-obs)*np.sqrt(w[:,None])).ravel(),x[:3]*.18,x[3:]*.002,behind*20]
 limit=np.r_[np.repeat(np.radians(3)/np.sqrt(3),3),2.,np.log(4),2.];zero=np.zeros(6);before_pred,base_camera,before,_=evaluate(zero)
 x=fit_math.least_squares(residual,zero,-limit,limit,steps=min(steps,45));pred,cfg,error,behind=evaluate(x)
 if behind.max()>0 or error>before:pred,cfg,error=before_pred,base_camera,before
 return dict(points=p.copy(),camera=cfg,predicted=pred,before=before,after=error,camera_only=error,baseline_predicted=before_pred,baseline_camera=base_camera,
  pose_accepted=False,link_deviation_deg=[0.]*8,source_camera_anchor=anchor.tolist(),camera_bound_deg=3.,per_link_parameter_bound_deg=0.,metric='fixed_intrinsics_source_pose_camera_only')
def solve(points,source,observed,weights,steps=40,camera_only=False):
 p=np.asarray(points,float);obs=np.asarray(observed,float);w=np.asarray(weights,float)
 if obs.shape!=(12,2) or w.shape!=(12,) or not np.isfinite(obs).all() or not np.isfinite(w).all() or (w<0).any() or (w>.1).sum()<6:raise ValueError('源姿势拟合需要至少六个可靠的图片关节')
 anchor=anchor_rotation(p,source);span=max(float(np.linalg.norm(np.ptp(p,axis=0))),.01)
 cam_limit=np.radians(3)/np.sqrt(3);limb_limit=np.radians(4)/np.sqrt(3)
 size=3 if camera_only else 27;limit=np.r_[np.repeat(cam_limit,3),np.repeat(limb_limit,size-3)]
 def geometry(x):
  q=p.copy()
  if len(x)>3:
   for i,(a,b,c) in enumerate(CHAINS):
    k=3+i*6;R=fit_math.rotation(x[k:k+3]);S=fit_math.rotation(x[k+3:k+6])
    q[b]=q[a]+R@(p[b]-p[a]);q[c]=q[b]+R@S@(p[c]-p[b])
  return q
 def evaluate(x):
  q=geometry(x);R=fit_math.rotation(x[:3])@anchor;pred,cfg,e=camera(q,obs,w,R);return q,pred,cfg,e
 def residual(x):
  q,pred,_,_=evaluate(x);r=(pred-obs)*np.sqrt(w[:,None]);r*=np.sqrt(2/(np.sqrt(1+(r/.06)**2)+1))
  return np.r_[r.ravel(),x[:3]*.18,x[3:]*.18,((q-p)/span*.12).ravel()]
 zero=np.zeros(size);_,base_pred,base_camera,before=evaluate(zero)
 cam=fit_math.least_squares(lambda z:residual(np.r_[z,np.zeros(size-3)]),np.zeros(3),-limit[:3],limit[:3],steps=min(steps,30))
 seed=np.r_[cam,np.zeros(size-3)];_,only_pred,only_camera,only_error=evaluate(seed)
 params=fit_math.least_squares(residual,seed,-limit,limit,steps=min(steps,40)) if not camera_only else seed
 q,pred,cfg,error=evaluate(params);angles=deviation(q,p)
 accepted=bool(error<only_error-max(.0015,only_error*.05) and max(angles)<8.1)
 if not accepted:q=p.copy();pred=only_pred;cfg=only_camera;error=only_error
 return dict(points=q,camera=cfg,predicted=pred,before=before,after=error,camera_only=only_error,
  baseline_predicted=base_pred,baseline_camera=base_camera,pose_accepted=accepted,link_deviation_deg=deviation(q,p),
  source_camera_anchor=anchor.tolist(),camera_bound_deg=3.,per_link_parameter_bound_deg=4.,metric='source_prediction_prior_not_ground_truth')


def bent_leg_report(points, camera, original, original_camera, reference, observed, weights, width):
    """Evaluate both sides independently and reject a reversed depth order."""
    before = camera_data.project(original, original_camera)
    after = camera_data.project(points, camera)
    old_error = np.linalg.norm(before - observed, axis=1) * width
    new_error = np.linalg.norm(after - observed, axis=1) * width
    old_depth = original @ np.asarray(original_camera['rotation'])[1]
    new_depth = points @ np.asarray(camera['rotation'])[1]
    depth_pairs = [(old_depth[a] - old_depth[b], new_depth[a] - new_depth[b]) for a, b in ((4,10),(5,11))]
    span = max(np.linalg.norm(np.ptp(original, axis=0)), .01)
    depth_ok = all(abs(old) < .02 * span or old * new > 0 for old, new in depth_pairs)
    bends = whole_body_math.anatomy(points, reference)
    errors = lambda values, ids: float(np.sqrt(np.mean(values[list(ids)] ** 2)))
    old_legs = [errors(old_error, ids) for ids in ((4,5),(10,11))]
    new_legs = [errors(new_error, ids) for ids in ((4,5),(10,11))]
    gain = errors(old_error, LEG_JOINTS) - errors(new_error, LEG_JOINTS)
    change = max(deviation(points, reference))
    valid = (depth_ok and bends['valid'] and change <= 20
             and np.all(new_error[weights > .65] <= old_error[weights > .65] + width * .008)
             and all(new <= old + width * .0002 for old, new in zip(old_legs, new_legs))
             and errors(new_error, UPPER_JOINTS) <= errors(old_error, UPPER_JOINTS) + width * .003
             and gain > width * .006)
    return dict(accepted=bool(valid), before_px=old_legs, after_px=new_legs,
                before_joints_px=old_error[list(LEG_JOINTS)].tolist(), after_joints_px=new_error[list(LEG_JOINTS)].tolist(),
                depth_order_preserved=bool(depth_ok), max_link_change_deg=change, gain_px=gain)


def bent_leg_proposal(points, camera, reference, source, observed, weights, image, steps):
    """Keep hips and upper body fixed while testing a few plausible focal lengths."""
    if any(weights[i] < .65 for i in LEG_JOINTS):
        return None
    if min(fit_math.knee_flex(*points[list(chain)]) for chain in LEG_CHAINS) < np.radians(55):
        return None
    width, height = image['width'], image['height']
    prior_error = np.linalg.norm(camera_data.project(points, camera) - observed, axis=1)
    if max(prior_error[list(LEG_JOINTS)]) < .025:
        return None
    camera_weights = weights.copy()
    camera_weights[list(UPPER_JOINTS)] *= 4.
    diagonal = np.hypot(width, height) / width
    limit = np.repeat(np.radians(18) / np.sqrt(3), 6)
    best = None
    for focal in diagonal * np.array([1., 1.5, 1.75, 2.]):
        intrinsics = [focal, focal, .5, height / (2 * width)]
        proposal = perspective_camera_only(points, source, observed, camera_weights, intrinsics, steps)['camera']
        depth_axis = np.asarray(proposal['rotation'])[1]
        fitted = points.copy()
        for a, b, c in LEG_CHAINS:
            original = points[[a,b,c]]
            thigh, shin = np.diff(original, axis=0)
            def geometry(x):
                turn = fit_math.rotation(x[:3])
                knee = original[0] + turn @ thigh
                return np.array([original[0], knee, knee + turn @ fit_math.rotation(x[3:]) @ shin])
            def residual(x):
                leg = geometry(x)
                error = (camera_data.project(leg, proposal)[1:] - observed[[b,c]]) * np.sqrt(weights[[b,c],None])
                return np.r_[error.ravel(), x * .06, ((leg - original) @ depth_axis)[1:] * .12]
            params = fit_math.least_squares(residual, np.zeros(6), -limit, limit, steps=min(steps,35))
            fitted[[a,b,c]] = geometry(params)
        report = bent_leg_report(fitted, proposal, points, camera, reference, observed, weights, width)
        if report['accepted'] and (best is None or report['gain_px'] > best['report']['gain_px']):
            best = dict(points=fitted, camera=proposal, report=report)
    return best
