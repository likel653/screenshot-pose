"""Fixed-intrinsic perspective camera + bounded, bone-length-preserving pose fit."""
import numpy as np
try:from . import fit_math,camera_data
except ImportError:import fit_math,camera_data


def solve(points,observed,weights,intrinsics,width,chains,locked_ends=(),steps=55,regularization=.035,coordinate_pelvis=False):
    P=np.asarray(points,float);obs=np.asarray(observed,float);w=np.asarray(weights,float);K=np.asarray(intrinsics,float)/width
    if P.shape!=(12,3) or obs.shape!=(12,2) or w.shape!=(12,) or K.shape!=(4,):raise ValueError('透视拟合输入尺寸错误')
    if not all(np.isfinite(a).all() for a in (P,obs,w,K)) or (w<0).any() or np.sum(w>.1)<6:raise ValueError('透视拟合需要六个可靠关节')
    if min(K[:2])<=0:raise ValueError('透视相机焦距必须为正')
    center=(P[3]+P[9])*.5;initial=P-center;span=max(float(np.linalg.norm(np.ptp(P,axis=0))),.01)
    _,weak,_=fit_math.project(initial,obs,w)
    depth=max(K[0]/weak['scale'],span)
    x0=[0.,0.,0.,(weak['shift'][0]-K[2])/weak['scale']/span,np.log(depth/span),(K[3]-weak['shift'][1])/weak['scale']/span]
    coordinate_pelvis=bool(coordinate_pelvis and 5 in locked_ends and 11 in locked_ends)
    layout=[];size=9 if coordinate_pelvis else 6
    for chain in chains:
        locked=chain[2] in locked_ends;layout.append((chain,size,locked));size+=1 if locked else 6
    lower=np.r_[[-.436]*3,[-10.,-.5,-10.],[-.8]*(size-6)]
    upper=np.r_[[.436]*3,[10.,np.log(100),10.],[.8]*(size-6)]
    if coordinate_pelvis:lower[6:9]=-.18;upper[6:9]=.18
    x0=np.r_[x0,np.zeros(size-6)]
    x0=np.clip(x0,lower,upper)

    def geometry(x):
        delta=x[6:9]*span if coordinate_pelvis else np.zeros(3)
        q=initial.copy()+delta
        for end in locked_ends:q[end]=initial[end]
        for (root,mid,end),offset,locked in layout:
            a=initial[mid]-initial[root];b=initial[end]-initial[mid]
            if locked:
                q[mid],_=fit_math.planted_knee(q[root],initial[end],initial[mid]+delta,np.linalg.norm(a),np.linalg.norm(b),x[offset])
            else:
                R=fit_math.rotation(x[offset:offset+3]);Q=fit_math.rotation(x[offset+3:offset+6])
                q[mid]=q[root]+R@a;q[end]=q[mid]+R@Q@b
        return q

    def camera(x):
        R=fit_math.rotation(x[:3]);translation=np.array([x[3]*span,np.exp(x[4])*span,x[5]*span])-R@center
        return dict(type='PERSPECTIVE',rotation=R.tolist(),translation=translation.tolist(),
                    normalized_intrinsics=K.tolist(),angles=x[:3].tolist(),intrinsics_fixed=True)

    def residual(x,pose=True):
        q=geometry(x) if pose else initial
        view=q@fit_math.rotation(x[:3]).T+[x[3]*span,np.exp(x[4])*span,x[5]*span]
        behind=np.maximum(.03*span-view[:,1],0)/span
        safe=view.copy();safe[:,1]=np.maximum(safe[:,1],.03*span)
        pred=np.column_stack((K[0]*safe[:,0]/safe[:,1]+K[2],-K[1]*safe[:,2]/safe[:,1]+K[3]))
        r=(pred-obs)*np.sqrt(w[:,None])
        # Soft-L1 grows with every residual. A bounded/saturating loss can
        # reduce cost by abandoning an entire foot while fitting the torso.
        if pose:r=r*np.sqrt(2/(np.sqrt(1+(r/.08)**2)+1))
        length_errors=[]
        for (root,mid,end),offset,locked in layout:
            if locked:length_errors.append((np.linalg.norm(q[end]-q[mid])-np.linalg.norm(initial[end]-initial[mid]))/span*3)
        return np.r_[r.ravel(),x[6:]*regularization,x[:3]*.01,(x[4]-x0[4])*.005,behind*3,length_errors]

    baseline=geometry(x0)+center;baseline_camera=camera(x0)
    baseline_pred=camera_data.project(baseline,baseline_camera);before=camera_data.error(baseline_pred,obs,w)
    cam_options=[x0[:6].copy()]
    for factor in (1.,1.6,2.5):
        start=x0[:6].copy();start[4]=np.clip(start[4]+np.log(factor),lower[4],upper[4])
        candidate=fit_math.least_squares(lambda a:residual(np.r_[a,np.zeros(size-6)],False),start,lower[:6],upper[:6],steps=min(steps,45))
        cam_options.append(candidate)
    def camera_rms(candidate):
        try:return camera_data.error(camera_data.project(P,camera(np.r_[candidate,np.zeros(size-6)])),obs,w)
        except ValueError:return float('inf')
    camx=min(cam_options,key=camera_rms)
    seed=np.r_[camx,np.zeros(size-6)];only_camera=camera(seed)
    only_pred=camera_data.project(P,only_camera);camera_error=camera_data.error(only_pred,obs,w)
    params=fit_math.least_squares(residual,seed,lower,upper,steps=steps)
    fitted=geometry(params)+center;cfg=camera(params)
    pred=camera_data.project(fitted,cfg);after=camera_data.error(pred,obs,w)
    flex=[np.degrees(fit_math.knee_flex(fitted[a],fitted[b],fitted[c])) for (a,b,c) in chains[1::2]]
    pose_accepted=bool(after<camera_error*.995 and max(flex)<170.01)
    if not pose_accepted:fitted=P.copy();cfg=only_camera;pred=only_pred;after=camera_error
    return dict(points=fitted,camera=cfg,predicted=pred,before=before,after=after,camera_only=camera_error,
      camera_only_camera=only_camera,camera_only_predicted=only_pred,
      baseline_camera=baseline_camera,baseline_predicted=baseline_pred,parameters=params.tolist(),
      pelvis_delta=(params[6:9]*span if coordinate_pelvis and pose_accepted else np.zeros(3)).tolist(),
      pelvis_coordination=coordinate_pelvis,
      pose_accepted=pose_accepted,bone_lengths_preserved=True,intrinsics_fixed=True)
