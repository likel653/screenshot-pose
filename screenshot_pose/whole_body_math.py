"""One bounded objective for root XYZ, torso FK, all limbs and fixed contacts.

No Blender/SciPy imports. Camera is fixed; contacts are explicit world-space
targets, not inferred floors. Source-depth and bend priors are not 3D truth.
"""
import numpy as np
try:from . import fit_math,camera_data
except ImportError:import fit_math,camera_data

CHAINS=((0,1,2),(3,4,5),(6,7,8),(9,10,11))

def unit(v):return v/max(float(np.linalg.norm(v)),1e-10)

def anatomy(points,reference):
    p=np.asarray(points,float);b=np.asarray(reference,float);flex=[];plane=[]
    for r,m,e in CHAINS:
        a=p[m]-p[r];v=p[e]-p[m];old_a=b[m]-b[r];old_v=b[e]-b[m]
        flex.append(float(np.degrees(fit_math.knee_flex(p[r],p[m],p[e]))))
        normal=np.cross(a,v);old_normal=np.cross(old_a,old_v)
        reliable=np.linalg.norm(old_normal)>np.linalg.norm(old_a)*np.linalg.norm(old_v)*.14 and np.linalg.norm(normal)>1e-7
        plane.append(float(np.degrees(np.arccos(np.clip(unit(normal)@unit(old_normal),-1,1)))) if reliable else 0.)
    shoulder=unit(p[6]-p[0]);hip=unit(p[9]-p[3]);old_s=unit(b[6]-b[0]);old_h=unit(b[9]-b[3])
    twist=float(np.degrees(np.arccos(np.clip(shoulder@hip,-1,1))));old_twist=float(np.degrees(np.arccos(np.clip(old_s@old_h,-1,1))))
    torso=np.linalg.norm((p[0]+p[6]-p[3]-p[9])*.5);old_torso=np.linalg.norm((b[0]+b[6]-b[3]-b[9])*.5)
    limits=[165.,170.,165.,170.];valid=all(f<=max(limit,o+2.) and f<178. for f,limit,o in zip(flex,limits,[np.degrees(fit_math.knee_flex(b[r],b[m],b[e])) for r,m,e in CHAINS]))
    valid=bool(valid and max(plane)<85 and .75<torso/max(old_torso,1e-8)<1.25 and twist<max(45.,old_twist+20.))
    return dict(valid=valid,flex_deg=flex,bend_plane_change_deg=plane,shoulder_hip_angle_deg=twist,source_shoulder_hip_angle_deg=old_twist,torso_length_ratio=float(torso/max(old_torso,1e-8)))

def solve(points,observed,weights,camera,controls=(),contacts=None,root_lower=None,root_upper=None,steps=45,regularization=.025):
    P=np.asarray(points,float);obs=np.asarray(observed,float);w=np.asarray(weights,float);contacts=contacts or {}
    if P.shape!=(12,3) or obs.shape!=(12,2) or w.shape!=(12,):raise ValueError('全身求解需要12个有效身体观测')
    if not all(np.isfinite(a).all() for a in (P,obs,w)) or (w<0).any() or (w>.1).sum()<6:raise ValueError('全身求解需要至少六个可靠关节')
    span=max(float(np.linalg.norm(np.ptp(P,axis=0))),.01)
    lo=np.asarray(root_lower if root_lower is not None else [-.10*span]*3,float)/span
    hi=np.asarray(root_upper if root_upper is not None else [.10*span]*3,float)/span
    if lo.shape!=(3,) or hi.shape!=(3,) or (lo>hi).any():raise ValueError('骨盆位移范围无效')
    layout=[];size=3+len(controls)*3
    for r,m,e in CHAINS:
        locked=e in contacts;layout.append((r,m,e,size,locked));size+=1 if locked else 6
    lower=np.r_[lo,*[np.repeat(-c['limit'],3) for c in controls],np.repeat(-.30,size-3-len(controls)*3)]
    upper=np.r_[hi,*[np.repeat(c['limit'],3) for c in controls],np.repeat(.30,size-3-len(controls)*3)]
    initial=np.clip(np.zeros(size),lower,upper)
    lengths={(r,m,e):(float(np.linalg.norm(P[m]-P[r])),float(np.linalg.norm(P[e]-P[m]))) for r,m,e in CHAINS}

    def geometry(x):
        delta=x[:3]*span;q=P.copy()+delta;pivots=np.array([c['pivot'] for c in controls],float).reshape(-1,3)+delta
        for i,c in enumerate(controls):
            R=fit_math.rotation(x[3+i*3:6+i*3]);origin=pivots[i].copy();ids=c['points'];q[ids]=(q[ids]-origin)@R.T+origin
            for child in c['children']:pivots[child]=(pivots[child]-origin)@R.T+origin
        for r,m,e,offset,locked in layout:
            a=q[m]-q[r];b=q[e]-q[m]
            if locked:
                job=contacts[e];target=np.asarray(job['target'],float);pole=np.asarray(job['pole'],float) if job.get('explicit_pole') else q[m]
                l1,l2=lengths[r,m,e];q[m],_=fit_math.planted_knee(q[r],target,pole,l1,l2,x[offset]);q[e]=target
            else:
                R=fit_math.rotation(x[offset:offset+3]);Q=fit_math.rotation(x[offset+3:offset+6]);q[m]=q[r]+R@a;q[e]=q[m]+R@Q@b
        return q

    def residual(x):
        q=geometry(x)
        try:pred=camera_data.project(q,camera)
        except ValueError:pred=np.ones((12,2))*100
        r=(pred-obs)*np.sqrt(w[:,None]);r*=np.sqrt(2/(np.sqrt(1+(r/.05)**2)+1))
        reach=[];angle=[];bend=[]
        for r0,m,e in CHAINS:
            l1,l2=lengths[r0,m,e]
            reach.extend([(np.linalg.norm(q[m]-q[r0])-l1)/span*30,(np.linalg.norm(q[e]-q[m])-l2)/span*30])
            flex=fit_math.knee_flex(q[r0],q[m],q[e]);limit=np.radians(170 if r0 in (3,9) else 165)
            angle.append(max(0.,flex-limit)*.3)
            n=np.cross(q[m]-q[r0],q[e]-q[m]);old=np.cross(P[m]-P[r0],P[e]-P[m])
            bend.extend((unit(n)-unit(old))*.012 if np.linalg.norm(old)>l1*l2*.14 else np.zeros(3))
        depth=((q-P)@np.asarray(camera['rotation']).T)[:,1]/span*.035*(2-w)
        shoulder=unit(q[6]-q[0])-unit(P[6]-P[0]);hip=unit(q[9]-q[3])-unit(P[9]-P[3])
        return np.r_[r.ravel(),x[:3]*.06,x[3:]*regularization,depth,reach,angle,bend,shoulder*.012,hip*.012]

    before_pred=camera_data.project(P,camera);before=camera_data.error(before_pred,obs,w)
    free=np.flatnonzero(upper-lower>1e-10)
    def expand(z):
        value=initial.copy();value[free]=z;return value
    params=expand(fit_math.least_squares(lambda z:residual(expand(z)),initial[free],lower[free],upper[free],steps=steps)) if len(free) else initial.copy()
    q=geometry(params)
    try:pred=camera_data.project(q,camera);after=camera_data.error(pred,obs,w)
    except ValueError:q=P.copy();pred=before_pred;after=before;params=initial
    lengths_error=max(abs(np.linalg.norm(q[v]-q[u])-lengths[r,m,e][i]) for r,m,e in CHAINS for i,(u,v) in enumerate(((r,m),(m,e))))
    contact_error=max((float(np.linalg.norm(q[e]-np.asarray(job['target']))) for e,job in contacts.items()),default=0.)
    original_reach=max((float(np.linalg.norm(P[e]-np.asarray(job['target']))) for e,job in contacts.items()),default=0.)
    old_errors=np.linalg.norm(before_pred-obs,axis=1);errors=np.linalg.norm(pred-obs,axis=1);reliable=w>.75
    allowance=.03 if original_reach>.003 else .006
    joint_guard=bool(np.all(errors[reliable]<=np.maximum(old_errors[reliable]*1.2,old_errors[reliable]+allowance)))
    score=anatomy(q,P);valid=bool(score['valid'] and lengths_error<.002 and contact_error<.002 and joint_guard)
    image_gain=before-after>max(.0005,before*.02)
    contact_gain=original_reach>.003 and after<before*1.2+.015
    accepted=bool(valid and (image_gain or contact_gain))
    return dict(points=q,parameters=params,root_delta=params[:3]*span,torso_parameters=params[3:3+len(controls)*3],
        before=before,after=after,predicted=pred,accepted=accepted,valid=valid,anatomy=score,reliable_joint_guard=joint_guard,
        max_length_error=float(lengths_error),contact_error=contact_error,initial_contact_error=original_reach,
        reason='constraints_or_anatomy_guard_failed' if not valid else 'image_and_anatomy_gain' if image_gain else 'previously_unreachable_contacts' if contact_gain else 'no_clear_gain')
