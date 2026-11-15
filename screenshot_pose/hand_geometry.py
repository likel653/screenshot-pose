"""Palm-local bounded hand fitting and conservative gesture suggestions."""
import numpy as np
try:from . import orientation_math as om
except ImportError:import orientation_math as om
FINGERS=('thumb','index','middle','ring','pinky')

def suggest(points):
 p=np.asarray(points,float)
 if p.shape!=(21,3) or not np.isfinite(p).all():return 'UNKNOWN'
 curl=om.curls(p)
 if max(curl['index'],curl['middle'])<.12 and min(curl['ring'],curl['pinky'])>.45:return 'V'
 if max(curl[f] for f in FINGERS[1:])<.1:return 'OPEN'
 if min(curl[f] for f in FINGERS[1:])>.6:return 'FIST'
 return 'UNKNOWN'

def transport(hand,rotation):
 """Rotate an auxiliary observation to the primary camera, never its body."""
 import copy
 result=copy.deepcopy(hand);r=om.proper_rotation(rotation)
 result['frame']=(r@om.proper_rotation(hand['frame'])).tolist()
 result['alternatives']=[(r@om.proper_rotation(f)).tolist() for f in hand.get('alternatives',[])]
 if 'world' in hand:
  p=np.asarray(hand['world'],float)
  if p.shape!=(21,3) or not np.isfinite(p).all():raise ValueError('手部三维点无效')
  result['world']=((p-p[0])@r.T).tolist()
 # Auxiliary pixels remain in their own image; do not call them primary pixels.
 if 'landmarks' in hand:result['reference_landmarks']=result.pop('landmarks')
 if hand.get('native_source')=='MANO' and hand.get('bone_frames'):
  # Bind frames stay in the model's neutral coordinates; only posed global
  # rotations move into the primary view. Conjugating would rotate bind twice.
  for item in result['bone_frames'].values():item['pose_rotation']=(r@om.proper_rotation(item['pose_rotation'])).tolist()
 else:result.pop('bone_frames',None)
 result['view_rotation']=r.tolist()
 return result

def refine(points,observations,gesture='AUTO'):
 """Fit valid 21-point estimates to crop observations, fixing palm and bone lengths.

 Each segment may rotate within 12 degrees of the initial estimate. This is
 a small refinement, not a recovery of invisible fingers or calibrated depth.
 """
 p=np.asarray(points,float);xy=np.asarray(observations,float)
 if p.shape!=(21,3) or xy.shape!=(21,2) or not np.isfinite(p).all() or not np.isfinite(xy).all():raise ValueError('手部精修需要完整的 21 点')
 palm=om.palm_frame(p);length=np.linalg.norm(p[9]-p[0])
 if length<1e-6:raise ValueError('手掌尺寸无效')
 centered=p-p[0];target=xy-xy[0]
 projection=centered[:,[0,2]]*np.array([1.,-1.]);scale=float(np.sum(projection*target)/max(np.sum(projection**2),1e-12))
 if scale<=0:raise ValueError('手部投影与三维方向不一致')
 scale=float(np.clip(scale,1e-4,1e4));vectors=[]
 for i in range(5):vectors.extend(np.diff(p[1+4*i:5+4*i],axis=0))
 vectors=np.array(vectors)
 if min(np.linalg.norm(vectors,axis=1))<length*.01:raise ValueError('手指骨长退化')
 def decode(x):
  rv=x.reshape(15,3)@palm.T;angle=np.linalg.norm(rv,axis=1,keepdims=True);axis=np.divide(rv,angle,out=np.zeros_like(rv),where=angle>1e-12)
  rotated=vectors*np.cos(angle)+np.cross(axis,vectors)*np.sin(angle)+axis*np.sum(axis*vectors,axis=1,keepdims=True)*(1-np.cos(angle))
  out=p.copy()
  for i in range(5):
   base=1+4*i
   out[base+1:base+4]=out[base]+np.cumsum(rotated[i*3:i*3+3],axis=0)
  return out
 def error(x):
  q=decode(x);pred=(q-q[0])[:,[0,2]]*np.array([1.,-1.])*scale
  return np.r_[(pred-target).ravel(),x*.03]
 initial=np.zeros(45);before=float(np.sqrt(np.mean(error(initial)[:42]**2)))
 x=initial.copy();damping=.002
 for _ in range(16):
  residual=error(x);jac=np.column_stack([(error(x+np.eye(45)[i]*1e-5)-residual)/1e-5 for i in range(45)])
  step=np.linalg.solve(jac.T@jac+np.eye(45)*damping,-jac.T@residual);improved=False
  for rate in (1.,.5,.25):
   trial=np.clip(x+step*rate,-.12,.12)
   if np.sum(error(trial)**2)<np.sum(residual**2)-1e-12:x=trial;improved=True;break
  if not improved or np.linalg.norm(step)<1e-5:break
 q=decode(x);after=float(np.sqrt(np.mean(error(x)[:42]**2)))
 # A guessed gesture must not be made contradictory by small image adjustments.
 accepted=bool(np.isfinite(q).all() and after<before*.98 and (gesture in ('AUTO','UNKNOWN') or suggest(q) in (gesture,'UNKNOWN')))
 return (q if accepted else p),dict(method='bounded_crop_reprojection_not_depth_truth',accepted=accepted,before=before,after=after if accepted else before,units='crop_normalized',max_component_rotation_rad=.12,bone_lengths_preserved=True)
