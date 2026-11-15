"""Independent face evidence for head candidates; not a measure of 3D truth."""
import hashlib,json
from pathlib import Path
import numpy as np
try:from . import camera_data,fit_math,orientation_math as om
except ImportError:import camera_data,fit_math,orientation_math as om

IDS=(0,2,1,31,71,77)
FACE=json.loads(Path(__file__).with_name('face_template.json').read_text(encoding='utf8'))
TEMPLATE=np.asarray(FACE['fitting_points'],float)
VALIDATION=np.asarray(FACE['validation_points'],float)

def reference(data):
    b=data.get('wholebody',[]);w=data['image']['width'];h=data['image']['height']
    if len(b)!=133 or set(data.get('edited_indices',[])).intersection((0,7,8)):return None
    scores=np.array([b[i].get('confidence',0) for i in IDS],float)
    xy=np.array([[b[i]['x']*w,b[i]['y']*h] for i in IDS],float)
    if not np.isfinite(xy).all() or not np.isfinite(scores).all() or min(scores)<.45:return None
    span=float(np.linalg.norm(np.ptp(xy,axis=0)))
    if span<24 or np.linalg.norm(xy[1]-xy[2])<8:return None
    result=dict(xy=xy,weights=scores,span=span)
    indices=[23+i for i in FACE['validation_face68_indices']]
    vx=np.array([[b[i]['x']*w,b[i]['y']*h] for i in indices],float);vw=np.array([b[i].get('confidence',0) for i in indices],float)
    good=(vw>=.5)&np.isfinite(vx).all(axis=1)
    if np.sum(good)>=10:result.update(validation_points=VALIDATION[good],validation_xy=vx[good],validation_weights=vw[good])
    return result

def camera(data,ref):
    w=data['image']['width'];h=data['image']['height'];cal=data.get('camera_calibration',{});reason=None
    if cal:
        try:
            a=camera_data.intrinsics(cal,w,h)
            digest=cal.get('processed_file_sha256');path=Path(data['image']['path'])
            if not digest or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
                raise ValueError('头部标定与图片内容未绑定')
            return a,dict(source=cal.get('source','estimated_intrinsics'),estimated=True,calibrated=True)
        except (ValueError,OSError) as exc:reason=str(exc)
    # The historical 512px crop/650px focal approximation, explicitly labelled.
    xy=ref['xy'];size=max(np.ptp(xy,axis=0))*1.8;center=(xy.max(0)+xy.min(0))/2
    return np.r_[np.repeat(650*size/512,2),center],dict(source='generic_face_crop_approximation',calibrated=False,reason=reason)

def score(frame,ref,K):
    R=om.C_DOWN.T@om.proper_rotation(frame);p=TEMPLATE@R.T
    fx,fy,cx,cy=K;xy=ref['xy'];w=ref['weights'];center=np.average(xy,axis=0,weights=w)
    depth=max((fx+fy)/2*90/ref['span'],100.)
    initial=np.array([(center[0]-cx)/fx*depth,(center[1]-cy)/fy*depth,np.log(depth)])
    def project(x):
        q=p+[x[0],x[1],np.exp(x[2])]
        if q[:,2].min()<=1:return None
        return q[:,:2]/q[:,2,None]*[fx,fy]+[cx,cy]
    def residual(x):
        pred=project(x)
        return np.ones(12)*1e3 if pred is None else ((pred-xy)*np.sqrt(w[:,None])/ref['span']).ravel()
    x=fit_math.least_squares(residual,initial,[-1e5,-1e5,np.log(80)], [1e5,1e5,np.log(1e6)],steps=28)
    pred=project(x)
    if pred is None:raise ValueError('头部候选位于相机后方')
    fitting=float(np.sqrt(np.sum((pred-xy)**2*w[:,None])/w.sum()))
    held_out='validation_points' in ref
    if held_out:
        q=np.asarray(ref['validation_points'])@R.T+[x[0],x[1],np.exp(x[2])]
        if q[:,2].min()<=1:raise ValueError('验证脸点位于相机后方')
        vp=q[:,:2]/q[:,2,None]*[fx,fy]+[cx,cy];vw=ref['validation_weights']
        rms=float(np.sqrt(np.sum((vp-ref['validation_xy'])**2*vw[:,None])/vw.sum()))
    else:rms=fitting
    return dict(error_px=rms,normalized_error=rms/ref['span'],fitting_error_px=fitting,held_out_validation=held_out,
        validation_count=len(ref.get('validation_xy',[])),projected=pred.tolist())

def angle(a,b):
    return float(np.degrees(np.arccos(np.clip((np.trace(np.asarray(a).T@b)-1)/2,-1,1))))

def select(body,candidates,ref,K,threshold=.45,neck_frame=None,prior=None):
    try:baseline=score(body,ref,K)
    except (ValueError,np.linalg.LinAlgError):return None,dict(chosen='body_head',reason='body_face_score_unavailable')
    fallback=None
    if prior and prior.get('confidence',0)>=threshold:
        try:
            frame=om.proper_rotation(prior['frame']);old=score(frame,ref,K)
            if angle(body,frame)<=85 and (neck_frame is None or angle(neck_frame,frame)<=110) and old['error_px']<baseline['error_px']:
                baseline=old;fallback=prior
        except (ValueError,KeyError,np.linalg.LinAlgError):pass
    rows=[];best=None
    for candidate in candidates:
        row=dict(source=candidate.get('source','unknown'),confidence=candidate.get('confidence',0),valid=False)
        try:
            frame=om.proper_rotation(candidate['frame']);a=angle(body,frame)
            if candidate.get('source')=='rtmw_6point_pnp' and 'validation_points' not in ref:raise ValueError('六点拟合缺少保留脸点复核，不覆盖旧头向')
            if not np.isfinite(row['confidence']) or row['confidence']<threshold:raise ValueError('局部置信度不足')
            if a>85:raise ValueError('与身体头向相差超过85度，保留身体候选')
            if neck_frame is not None and angle(neck_frame,frame)>110:raise ValueError('头颈相对旋转超出保守范围')
            item=score(frame,ref,K);row.update(item,delta_deg=a,valid=True)
            gain=baseline['error_px']-item['error_px'];required=max(.02*ref['span'],baseline['error_px']*.08)
            # Local confidence alone is insufficient; require independent gain.
            row['accepted']=bool(gain>required)
            row['reason']='independent_face_gain' if row['accepted'] else 'no_clear_face_gain_over_body'
            value=item['normalized_error']+.0003*a
            if row['accepted'] and (best is None or value<best[0]):best=(value,candidate)
        except (KeyError,ValueError,np.linalg.LinAlgError) as exc:row['reason']=str(exc)
        rows.append(row)
    chosen=best[1] if best else fallback
    return chosen,dict(source='held_out_face_evidence',baseline=baseline,candidates=rows,
        chosen=chosen.get('source') if chosen else 'body_head',preserved_prior=bool(fallback and not best),metric='held_out_generic_face_landmarks_not_3d_accuracy' if 'validation_points' in ref else 'generic_6point_fit_not_independent_for_pnp')
