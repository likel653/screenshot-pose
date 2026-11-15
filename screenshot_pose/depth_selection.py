"""Comparable body-depth decisions with explicit, non-ground-truth priors."""
import math
import numpy as np
try:from . import camera_data,whole_body_math
except ImportError:import camera_data,whole_body_math

def measure(points,reference,observed,weights,camera,width,contacts=()):
    p=np.asarray(points,float);ref=np.asarray(reference,float);width=float(width)
    if p.shape!=(12,3) or ref.shape!=(12,3) or not np.isfinite(p).all() or not np.isfinite(ref).all():raise ValueError('候选身体关节无效')
    if not math.isfinite(width) or width<=0:raise ValueError('图片尺寸无效')
    detail=whole_body_math.anatomy(p,ref)
    span=max(float(np.linalg.norm(np.ptp(ref,axis=0))),.01)
    center=lambda q:q-(q[3]+q[9])*.5
    depth=(center(p)-center(ref))@np.asarray(camera['rotation'],float).T
    depth_rms=float(np.sqrt(np.mean(depth[:,1]**2))/span)
    plane=float(np.mean(detail['bend_plane_change_deg']))
    reach=0.
    for endpoint,target in contacts:
        r,m,e=next(chain for chain in whole_body_math.CHAINS if chain[2]==endpoint)
        l1=np.linalg.norm(p[m]-p[r]);l2=np.linalg.norm(p[e]-p[m]);d=np.linalg.norm(np.asarray(target)-p[r])
        reach+=max(0.,d-l1-l2,abs(l1-l2)-d)/span
    pixel=camera_data.error(camera_data.project(p,camera),np.asarray(observed,float),weights)*width
    penalty=width*(.0004*plane+.012*depth_rms+.0002*max(0.,detail['shoulder_hip_angle_deg']-45)+.4*reach)
    # Anatomy is a visible guard, not a label of human pose correctness.
    valid=bool(np.isfinite(pixel+penalty) and detail['valid'])
    return dict(valid=valid,projection_px=float(pixel),score_px=float(pixel+penalty),
        depth_prior=depth_rms,bend_plane_prior_deg=plane,contact_reach_prior=reach,anatomy=detail,
        metric='fixed_camera_2d_plus_source_and_explicit_contact_priors_not_3d_accuracy')

def choose(items,width,default=0):
    width=float(width)
    if not math.isfinite(width) or width<=0:raise ValueError('图片尺寸无效')
    valid=[i for i,v in enumerate(items) if v.get('valid') and math.isfinite(float(v.get('score_px',math.inf)))]
    if not valid:raise ValueError('所有身体深度候选均未通过检查')
    best=min(valid,key=lambda i:items[i]['score_px']);baseline=default if default in valid else best
    selected=baseline
    gain=items[baseline]['score_px']-items[best]['score_px']
    if gain>max(width*.002,items[baseline]['score_px']*.05):selected=best
    ordered=sorted(items[i]['score_px'] for i in valid)
    ambiguous=len(ordered)>1 and ordered[1]-ordered[0]<=max(width*.004,ordered[0]*.05)
    return selected,dict(ambiguous=bool(ambiguous),reason='clear_combined_evidence' if selected!=baseline else 'retain_default_without_clear_combined_gain',
        automatic_3d_accuracy_validated=False)
