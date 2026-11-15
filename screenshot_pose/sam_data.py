"""Native MHR frames to pose interchange; retain all 127 source joints."""
import copy,json
from pathlib import Path
import numpy as np
try:
 from . import pose_data,orientation_math as om
except ImportError:
 import pose_data,orientation_math as om

ROLES={'hips':'root','spine':'c_spine0','chest':'c_spine3','neck':'c_neck','head':'c_head'}
BODY={}
for side,prefix,ids in [('L','l_',(11,13,15,23,25,27,31)),('R','r_',(12,14,16,24,26,28,32))]:
 for role,name in [('upper_arm','uparm'),('forearm','lowarm'),('hand','wrist'),('thigh','upleg'),('shin','lowleg'),('foot','foot')]:ROLES[role+'.'+side]=prefix+name
 for i,name in zip(ids,('uparm','lowarm','wrist','upleg','lowleg','foot','ball')):BODY[i]=prefix+name


def check_screenshot_observations(data, raw):
 """Cross-check image observations with the independently projected MHR70."""
 projected = np.asarray(raw.get('keypoints_pixels'))
 if projected.shape != (1, 70, 2) or not np.isfinite(projected).all():
  return
 w, h = data['image']['width'], data['image']['height']
 points = data['landmarks']
 reference = np.array([[p['x'] * w, p['y'] * h] for p in points])
 projected = projected[0]
 try:from .screenshot_recovery import leg_evidence
 except ImportError:from screenshot_recovery import leg_evidence
 evidence=leg_evidence(data,projected)
 if evidence:data['screenshot_leg_evidence']=evidence
 warnings = data.setdefault('backend_warnings', [])
 # MHR70 knees, ankles, heels and big toes. Both image-model chains must
 # also pile up at the lower edge before discarding their visibility.
 legs = ((25, 27, 29, 31), (26, 28, 30, 32))
 mhr_legs = ((11, 13, 17, 15), (12, 14, 20, 18))
 if all(reference[list(ids), 1].min() >= h * .9 and
        projected[list(mhr), 1].min() > h * 1.02 for ids, mhr in zip(legs, mhr_legs)):
  for ids in legs:
   for index in ids:
    points[index].update(visibility=0., presence=0., outside_frame=True)
  wholebody = data.get('wholebody', ())
  if len(wholebody) == 133:
   for index in range(13, 23):
    wholebody[index].update(confidence=0., outside_frame=True)
  warnings.append('二维模型在图像下缘补出了腿脚，三维模型也判断在画面外，已标记为不可见')
 review=[]
 for side,ids,mhr in (('L',(27,29,31),(13,17,15,16)),('R',(28,30,32),(14,20,18,19))):
  if any(points[i].get('outside_frame') or min(points[i]['visibility'],points[i]['presence'])<.5 for i in ids):
   continue
  ankle,heel,toe=reference[list(ids)]
  sam_ankle,sam_heel,big,small=projected[list(mhr)]
  sam_toe=(big+small)*.5
  if any(np.any(p<0) or np.any(p>[w,h]) for p in (ankle,heel,toe,sam_ankle,sam_heel,sam_toe)):
   continue
  span=np.linalg.norm(toe-heel)
  if span>w*.035 and max(np.linalg.norm(toe-sam_toe),np.linalg.norm(heel-sam_heel))>max(w*.025,span*.28):
   review.append(side)
 if review:data['foot_review_sides']=review
 shoulder_span = np.linalg.norm(reference[11] - reference[12])
 for name, index, mhr in (('左肘', 13, 7), ('右肘', 14, 8), ('左腕', 15, 62), ('右腕', 16, 41)):
  if min(points[index]['visibility'], points[index]['presence']) < .35:
   continue
  error = np.linalg.norm(reference[index] - projected[mhr])
  if error > max(w * .025, shoulder_span * .12):
   warnings.append(f'{name}的二维与三维模型定位相差 {error:.0f} 像素，请核对原图')
   if index in (15,16):
    data.setdefault('screenshot_wrist_disagreements', []).append('L' if index==15 else 'R')

def quaternion_matrix(q):
 x,y,z,w=np.asarray(q,float)/np.linalg.norm(q)
 return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                  [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                  [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def convert(raw,reference,geometry=None,camera=None):
 data=copy.deepcopy(pose_data.validate(reference))
 geometry=geometry or json.loads(Path(__file__).with_name('sam_geometry.json').read_text())
 names=geometry['names'];parents=geometry['parents'];index={n:i for i,n in enumerate(names)}
 p=np.asarray(raw['joints'],float)[0];rot=np.asarray(raw['joint_rotations'],float)[0]
 if p.shape!=(127,3) or rot.shape!=(127,3,3) or not np.isfinite(p).all() or not np.isfinite(rot).all():raise ValueError('SAM 127 关节数据无效')
 # Native results have CV coordinates for points, but original MHR GLOBAL
 # rotations. Left-multiply C_FACE for rotations, C_DOWN for result points.
 posed=p@om.C_DOWN.T
 bind=np.asarray(geometry['inverse_bind'],float)
 restrot=np.array([quaternion_matrix(v[3:7]).T for v in bind])
 resthead=np.array([-r@v[:3]/v[7]/100 for r,v in zip(restrot,bind)])
 root=p[[index['l_upleg'],index['r_upleg']]].mean(0)
 worlds=copy.deepcopy(data['world_landmarks'])
 def put(i,n):
  v=p[index[n]]-root;worlds[i].update(x=float(v[0]),y=float(v[1]),z=float(v[2]))
 for i,n in BODY.items():put(i,n)
 for i in range(11):put(i,'c_head')
 for side,prefix,ids in [('L','l_',(17,19,21,29)),('R','r_',(18,20,22,30))]:
  for i,n in zip(ids,('pinky_null','index_null','thumb_null','foot')):put(i,prefix+n)
 candidates=copy.deepcopy(data.get('body_candidates',[]))
 if not candidates:candidates=[dict(source=data.get('depth_source','reference'),world_landmarks=copy.deepcopy(data['world_landmarks']))]
 candidates=[c for c in candidates if c['source']!='sam3d_mhr_depth']
 candidates.insert(0,dict(source='sam3d_mhr_depth',world_landmarks=worlds))
 # Keep SAM as the default depth; existing target-aware selection can compare
 # all candidates against the same independent RTMW image observations.
 data['world_landmarks']=worlds
 bones={}
 for i,n in enumerate(names):
  children=[k for k,v in enumerate(parents) if v==i]
  child=next((k for k in children if not names[k].endswith('_proc')),children[0] if children else None)
  rh=resthead[i];ph=posed[i];pr=om.C_FACE@rot[i]
  rt=resthead[child] if child is not None else rh+(rh-resthead[parents[i]])*.25 if parents[i]>=0 else rh+np.array([0,.02,0])
  if np.linalg.norm(rt-rh)<1e-6:rt=rh+restrot[i][:,1]*.005
  pt=posed[child] if child is not None else ph+pr@restrot[i].T@(rt-rh)
  bones[n]=dict(parent=names[parents[i]] if parents[i]>=0 else None,rest_head=rh.tolist(),rest_tail=rt.tolist(),rest_rotation=restrot[i].tolist(),pose_head=ph.tolist(),pose_tail=pt.tolist(),pose_rotation=om.proper_rotation(pr).tolist())
 data['full_skeleton']=dict(schema='screenshot_pose/skeleton_v1',source='sam3d_mhr',pose_axes='X_right_Y_away_Z_up',rest_axes='MHR_native',role_map=ROLES,bones=bones,
                           spine_chain=['c_spine0','c_spine1','c_spine2','c_spine3'],shoulder_map={'L':'l_clavicle','R':'r_clavicle'})
 details=data.setdefault('orientation',dict(schema='screenshot_pose/orientation_v1',camera_axes='X_right_Y_away_Z_up',hands={},feet={},warnings=[]))
 # SAM supplies a head frame with its actual bind roll removed. Existing
 # independent local hand refinement is retained, including a visible V sign.
 head=index['c_head'];details['head']=dict(frame=om.proper_rotation(om.C_FACE@rot[head]@restrot[head].T).tolist(),confidence=.65,source='sam3d_mhr_head',confidence_method='uncalibrated_model_candidate')
 details.setdefault('warnings',[]).append('SAM 当前为身体分支；手指沿用独立局部精修，遮挡手仍需核对')
 data.update(source='sam3d_mhr',depth_source='sam3d_mhr_depth',body_candidates=candidates,suggested_mode='HYBRID',sam3d=dict(joint_count=127,hand_crop_refinement=False,camera_estimate=camera,coordinates='CV points; original MHR global rotations converted with C_FACE'))
 return pose_data.validate(data)
