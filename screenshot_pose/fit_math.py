"""Bounded, confidence-weighted weak-perspective fit. NumPy only, no bpy/SciPy."""
import numpy as np

def rotation(v):
    v=np.asarray(v,dtype=float);theta=np.linalg.norm(v)
    K=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    if theta<1e-8:return np.eye(3)+K+.5*K@K
    return np.eye(3)+np.sin(theta)/theta*K+(1-np.cos(theta))/(theta*theta)*K@K

def project(points,observed,weights,angles=(0,0,0)):
    R=rotation(angles);q=np.asarray(points)@R.T
    xy=q[:,[0,2]]*np.array([1,-1]);w=np.asarray(weights)[:,None]
    mean=(xy*w).sum(0)/w.sum();target=(observed*w).sum(0)/w.sum()
    den=((xy-mean)**2*w).sum()
    if den<1e-12:raise ValueError('关节点重合，无法匹配相机')
    scale=max(1e-7,float(((xy-mean)*(observed-target)*w).sum()/den))
    shift=target-scale*mean
    pred=scale*xy+shift
    rms=float(np.sqrt(((pred-observed)**2*w).sum()/w.sum()))
    return pred,dict(rotation=R.tolist(),scale=scale,shift=shift.tolist(),angles=list(angles)),rms

def least_squares(fun,x,lower,upper,steps=45):
    x=np.asarray(x,dtype=float).copy();r=fun(x);cost=r@r;damping=.001
    for iteration in range(steps):
        J=np.empty((len(r),len(x)));eps=1e-5
        for j in range(len(x)):
            trial=x.copy();trial[j]+=eps;J[:,j]=(fun(trial)-r)/eps
        try:delta=np.linalg.solve(J.T@J+damping*np.eye(len(x)),-J.T@r)
        except np.linalg.LinAlgError:break
        candidate=np.clip(x+delta,lower,upper);nr=fun(candidate);nc=nr@nr
        if np.isfinite(nc) and nc<cost:
            gain=cost-nc;x,r,cost=candidate,nr,nc;damping=max(damping*.4,1e-7)
            if gain<1e-10 or np.linalg.norm(delta)<1e-6:break
        else:damping=min(damping*5,1e5)
    return x

def solve(points,observed,weights,chains,locked_ends=(),steps=45,regularization=.025):
    """chains: (root, elbow/knee, wrist/ankle) indices; grounded ankles stay EXACTLY fixed.

    A fixed torso and measured segment vectors preserve target lengths. Camera
    rotations are bounded to 25 degrees; limb deltas to ~46 degrees per axis.
    This is a local fit, not an anatomical joint-limit or collision solver.
    """
    initial=np.asarray(points,dtype=float);obs=np.asarray(observed,dtype=float);w=np.asarray(weights,dtype=float)
    if initial.ndim!=2 or initial.shape[1]!=3 or obs.shape!=(len(initial),2) or w.shape!=(len(initial),):raise ValueError('投影数据形状错误')
    if not all(np.isfinite(a).all() for a in (initial,obs,w)) or (w<0).any() or (w>.1).sum()<6:raise ValueError('至少需要六个可靠且有效的关节点')
    spans=np.linalg.norm(initial.max(0)-initial.min(0));spans=max(spans,.01)
    layout=[];size=3
    for chain in chains:
        locked=chain[2] in locked_ends
        layout.append((chain,size,locked));size+=1 if locked else 6
    limits=np.array([.436]*3+[.8]*(size-3));x0=np.zeros(size)
    def geometry(x):
        P=initial.copy()
        for (root,mid,end),offset,locked in layout:
            v1=initial[mid]-initial[root];v2=initial[end]-initial[mid]
            if locked:
                axis=initial[end]-initial[root];norm=np.linalg.norm(axis)
                if norm>1e-8:P[mid]=P[root]+rotation(axis/norm*x[offset])@v1
            else:
                R=rotation(x[offset:offset+3]);Q=rotation(x[offset+3:offset+6])
                P[mid]=P[root]+R@v1;P[end]=P[mid]+R@Q@v2
        return P
    def residual(x):
        P=geometry(x);pred,_,_=project(P,obs,w,x[:3])
        # Robust loss prevents a single misidentified point from dominating.
        r=(pred-obs)*np.sqrt(w[:,None]);r=r/np.sqrt(1+(r/.12)**2)
        depth=(P[:,1]-initial[:,1])/spans*.035
        return np.concatenate((r.ravel(),x[3:]*regularization,x[:3]*.02,depth))
    baseline_pred,baseline_camera,baseline=project(initial,obs,w)
    camx=least_squares(lambda x:np.concatenate((((project(initial,obs,w,x)[0]-obs)*np.sqrt(w[:,None])).ravel(),x*.02)),np.zeros(3),-limits[:3],limits[:3],steps=25)
    camera_pred,camera_only,camera_error=project(initial,obs,w,camx)
    x0[:3]=camx
    params=least_squares(residual,x0,-limits,limits,steps)
    P=geometry(params);pred,camera,error=project(P,obs,w,params[:3])
    if error>camera_error:
        P=initial.copy();pred=camera_pred;camera=camera_only;error=camera_error
    return dict(points=P,camera=camera,predicted=pred,before=baseline,after=error,camera_only=camera_error,
                baseline_camera=baseline_camera,baseline_predicted=baseline_pred,parameters=params)

def knee_flex(hip,knee,ankle):
    a=np.asarray(knee)-hip;b=np.asarray(ankle)-knee
    return float(np.arccos(np.clip(a@b/max(np.linalg.norm(a)*np.linalg.norm(b),1e-12),-1,1)))

def planted_knee(hip,ankle,pole,upper,lower,angle=0.):
    """Exact bone lengths for reachable targets, with explicit reach residual."""
    v=ankle-hip;distance=float(np.linalg.norm(v))
    if min(upper,lower,distance)<1e-8:raise ValueError('腿部骨长或目标无效')
    axis=v/distance;d=np.clip(distance,abs(upper-lower)+1e-7,upper+lower-1e-7)
    along=(upper*upper-lower*lower+d*d)/(2*d);height=np.sqrt(max(0,upper*upper-along*along))
    bend=pole-hip; bend=bend-axis*(bend@axis)
    if np.linalg.norm(bend)<1e-7:
        hint=np.eye(3)[np.argmin(abs(axis))];bend=np.cross(axis,hint)
    bend=rotation(axis*angle)@(bend/np.linalg.norm(bend))
    return hip+axis*along+bend*height,float(abs(distance-d))

def solve_grounded(points,observed,weights,steps=45,regularization=.025):
    """Compare fixed-torso fit with bounded pelvis + planted-leg coordination.

    Both feet stay fixed. Torso translates rigidly; shoulders/hips retain their
    relative positions. This is a local weak-perspective fit, not collision or
    physical balance reconstruction. Existing solver remains the fallback.
    """
    initial=np.asarray(points,float);obs=np.asarray(observed,float);w=np.asarray(weights,float)
    chains=[(0,1,2),(3,4,5),(6,7,8),(9,10,11)]
    base=solve(initial,obs,w,chains,locked_ends=(5,11),steps=steps,regularization=regularization)
    reference=base['points']
    span=max(float(np.linalg.norm(initial.max(0)-initial.min(0))),.01)
    lengths=[(np.linalg.norm(initial[m]-initial[r]),np.linalg.norm(initial[e]-initial[m])) for r,m,e in chains]
    # Camera (3), root translation normalized by body span (3; XY locked), knee pole angles
    # (2), arm rotations (12). Geometry checks below reject unreachable legs.
    lower=np.array([-.436]*3+[0.,0.,-.20]+[-.8]*2+[-.8]*12)
    upper=-lower;x0=np.zeros(20);x0[:3]=base['camera']['angles']
    def geometry(x):
        delta=x[3:6]*span;P=reference+delta;reach=[]
        for ci,(r,m,e) in enumerate(chains):
            if ci in (1,3):
                P[e]=initial[e];P[m],err=planted_knee(P[r],P[e],reference[m]+delta,*lengths[ci],x[6+(ci==3)]);reach.append(err/span)
            else:
                offset=8+(6 if ci==2 else 0);R=rotation(x[offset:offset+3]);Q=rotation(x[offset+3:offset+6])
                P[m]=P[r]+R@(reference[m]-reference[r]);P[e]=P[m]+R@Q@(reference[e]-reference[m])
        return P,np.array(reach)
    def residual(x):
        P,reach=geometry(x);pred,_,_=project(P,obs,w,x[:3]);r=(pred-obs)*np.sqrt(w[:,None]);r=r/np.sqrt(1+(r/.12)**2)
        flex=np.array([knee_flex(P[r],P[m],P[e]) for r,m,e in (chains[1],chains[3])])
        limit=np.maximum(flex-np.radians(170),0)
        return np.concatenate((r.ravel(),x[3:6]*.10,x[6:]*regularization,x[:3]*.02,
                               (P[:,1]-initial[:,1])/span*.035,reach*20,limit*.3))
    params=least_squares(residual,x0,lower,upper,steps=steps);P,reach=geometry(params)
    pred,camera,error=project(P,obs,w,params[:3]);flex=[knee_flex(P[r],P[m],P[e]) for r,m,e in (chains[1],chains[3])]
    valid=bool(max(reach)*span<1e-5 and max(flex)<=np.radians(170)+1e-5 and np.isfinite(P).all())
    use=valid and error<base['after']-max(1e-6,base['after']*.005)
    base['coordination']=dict(accepted=use,legacy_projection=base['after'],candidate_projection=error,
                              candidate_valid=valid,pelvis_delta=(params[3:6]*span).tolist(),
                              knee_flex_deg=np.degrees(flex).tolist(),reach_error=(reach*span).tolist())
    base['pelvis_delta']=(params[3:6]*span).tolist() if use else [0.,0.,0.]
    if use:base.update(points=P,predicted=pred,camera=camera,after=error,parameters=params)
    return base
