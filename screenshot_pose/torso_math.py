"""Bounded hierarchical pelvis/spine/clavicle proposals with a fixed camera."""
import numpy as np
try:from . import fit_math,camera_data
except ImportError:import fit_math,camera_data

def geometry(points,controls,parameters,locked=()):
    q=np.array(points,float,copy=True);pivots=np.array([c['pivot'] for c in controls],float)
    for i,c in enumerate(controls):
        R=fit_math.rotation(parameters[3*i:3*i+3]);origin=pivots[i].copy();ids=c['points']
        q[ids]=(q[ids]-origin)@R.T+origin
        for j in c['children']:pivots[j]=(pivots[j]-origin)@R.T+origin
    for root,mid,end in ((3,4,5),(9,10,11)):
        if end in locked:
            q[mid],_=fit_math.planted_knee(q[root],points[end],q[mid],np.linalg.norm(points[mid]-points[root]),np.linalg.norm(points[end]-points[mid]),0.)
            q[end]=points[end]
    return q

def solve(points,observed,weights,camera,controls,locked=(),steps=35):
    P=np.asarray(points,float);obs=np.asarray(observed,float);w=np.asarray(weights,float)
    if P.shape!=(12,3) or obs.shape!=(12,2) or w.shape!=(12,) or not controls:raise ValueError('躯干联合求解输入不完整')
    if not all(np.isfinite(a).all() for a in (P,obs,w)) or (w<0).any():raise ValueError('躯干观测包含无效值')
    span=max(np.linalg.norm(np.ptp(P,axis=0)),.01);limits=np.repeat([c['limit'] for c in controls],3)
    start=np.zeros(len(limits));initial=camera_data.project(P,camera);before=camera_data.error(initial,obs,w)
    def residual(x):
        q=geometry(P,controls,x,locked);pred=camera_data.project(q,camera)
        r=(pred-obs)*np.sqrt(w[:,None]);r*=np.sqrt(2/(np.sqrt(1+(r/.04)**2)+1))
        lengths=[]
        for a,b,c in ((0,1,2),(3,4,5),(6,7,8),(9,10,11)):
            for u,v in ((a,b),(b,c)):lengths.append((np.linalg.norm(q[v]-q[u])-np.linalg.norm(P[v]-P[u]))/span*5)
        return np.r_[r.ravel(),x*.028,(q[:,1]-P[:,1])/span*.02,lengths]
    x=fit_math.least_squares(residual,start,-limits,limits,steps)
    q=geometry(P,controls,x,locked);pred=camera_data.project(q,camera);after=camera_data.error(pred,obs,w)
    old=np.linalg.norm(initial-obs,axis=1);new=np.linalg.norm(pred-obs,axis=1)
    reliable=(w>.75);guard=bool(np.all(new[reliable]<np.maximum(old[reliable]*1.2,old[reliable]+.006)))
    accepted=bool(before-after>max(.0005,before*.02) and guard)
    return dict(points=q,parameters=x,accepted=accepted,before=before,after=after,reliable_joint_guard=guard)
