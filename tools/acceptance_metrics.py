"""Camera-aware controlled-proxy metrics, explicitly separate from real photos."""
import numpy as np
CHAINS=((0,1,2),(3,4,5),(6,7,8),(9,10,11))
LINKS=tuple((a,b) for r,m,e in CHAINS for a,b in ((r,m),(m,e)))
def finite(points,shape):
 p=np.asarray(points,float)
 if p.shape!=shape or not np.isfinite(p).all():raise ValueError('Invalid evaluated joint proxy shape')
 return p
def unit(v):return v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-10)
def angles(a,b):return np.degrees(np.arccos(np.clip(np.sum(unit(a)*unit(b),axis=-1),-1,1)))
def torso(p):return max(float(np.linalg.norm((p[0]+p[6]-p[3]-p[9])*.5)),1e-8)
def normalize(p):return (p-(p[3]+p[9])*.5)/torso(p)
def body(predicted,reference):
 p=finite(predicted,(12,3));q=finite(reference,(12,3));error=np.linalg.norm(normalize(p)-normalize(q),axis=1)
 degrees=angles(np.array([p[b]-p[a] for a,b in LINKS]),np.array([q[b]-q[a] for a,b in LINKS]))
 return dict(torso_normalized_mean=float(error.mean()),torso_normalized_max=float(error.max()),mean_direction_deg=float(degrees.mean()),per_link_deg=degrees.tolist(),
  alignment='root_translation_and_independent_torso_scale_only_no_rotation_or_reflection')
def fidelity(source,target):
 p=finite(source,(12,3));q=finite(target,(12,3));p=p-p.mean(0);q=q-q.mean(0)
 u,_,vt=np.linalg.svd(p.T@q);sign=np.eye(3);sign[-1,-1]=np.linalg.det(u@vt);rot=u@sign@vt
 scale=float(np.sum((p@rot)*q)/max(np.sum(p*p),1e-10));mapped=p@rot*scale
 r=body(mapped,q);r['alignment']='proper_similarity_global_rotation_translation_scale_removed_no_reflection';return r
def image_points(predicted,reference,size):
 p=finite(predicted,(12,2));q=finite(reference,(12,2));e=np.linalg.norm(p-q,axis=1)
 return dict(mean_px=float(e.mean()),max_px=float(e.max()),mean_image_height=float(e.mean()/size[1]),per_joint_px=e.tolist())
def palm_frame(p):
 y=unit((p[5]+p[9]+p[17])/3-p[0]);x=p[5]-p[17];x=unit(x-y*np.dot(x,y));z=unit(np.cross(x,y));return np.column_stack((x,y,z))
def hand(predicted,reference,frame=None):
 if predicted is None:return dict(available=False,reason='no_21_point_observation')
 p=finite(predicted,(21,3));q=finite(reference,(21,3));a=palm_frame(p) if frame is None else finite(frame,(3,3));b=palm_frame(q)
 lengths=[np.linalg.norm(p[9]-p[0]),np.linalg.norm(q[9]-q[0])]
 if min(lengths)<1e-7:return dict(available=False,reason='degenerate_palm')
 x=(p-p[0])@a;y=(q-q[0])@b;directions=[]
 for base in (1,5,9,13,17):directions.extend(angles(np.diff(x[base:base+4],axis=0),np.diff(y[base:base+4],axis=0)))
 angle=float(np.degrees(np.arccos(np.clip((np.trace(a.T@b)-1)/2,-1,1))))
 return dict(available=True,palm_frame_deg=angle,finger_direction_deg=float(np.mean(directions)),per_segment_deg=np.asarray(directions).tolist(),
  palm_local_shape_error=float(np.mean(np.linalg.norm(x/lengths[0]-y/lengths[1],axis=1))),alignment='finger_shape_in_each_palm_frame_and_explicit_camera_palm_rotation')
def suspect_layers(gt,source_body,source_2d,output_body,transfer,limits):
 if not gt:return ['real_photo_no_ground_truth_manual_review_required']
 rows=[]
 if source_2d is None or source_2d['mean_image_height']>limits['body_2d_image_height']:rows.append('image_localization')
 if source_body is None or source_body['torso_normalized_mean']>limits['body_3d_torso_normalized']:rows.append('source_depth_or_articulation')
 if transfer and transfer['mean_direction_deg']>limits['body_direction_deg']:rows.append('retarget_or_deliberate_fit_change')
 if output_body and output_body['torso_normalized_mean']>limits['body_3d_torso_normalized'] and not rows:rows.append('final_pose_or_camera_alignment')
 return rows or ['within_provisional_controlled_targets']
