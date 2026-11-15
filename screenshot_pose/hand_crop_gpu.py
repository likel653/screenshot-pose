"""Dedicated 21-point hand crops on CUDA. This model predicts 2D, not depth."""
import copy
import numpy as np
try:from . import local_pose_worker as local
except ImportError:import local_pose_worker as local

def suggest_v(xy):
 p=np.asarray(xy,float)
 if p.shape!=(21,2) or not np.isfinite(p).all():return 'UNKNOWN'
 palm=np.linalg.norm(p[9]-p[0])
 if palm<1e-6:return 'UNKNOWN'
 straight=[];curled=[]
 for i in (5,9,13,17):
  v=np.diff(p[i:i+4],axis=0);norm=np.linalg.norm(v,axis=1)
  if min(norm)<palm*.025:return 'UNKNOWN'
  cos=[float(np.dot(v[j],v[j+1])/norm[j]/norm[j+1]) for j in (0,1)]
  straight.append(min(cos)>.7 and np.linalg.norm(p[i+3]-p[0])>np.linalg.norm(p[i+1]-p[0])*1.05)
  curled.append(min(cos)<.45 or np.linalg.norm(p[i+3]-p[0])<np.linalg.norm(p[i+1]-p[0])*.85)
 return 'V' if all(straight[:2]) and all(curled[2:]) and np.linalg.norm(p[8]-p[12])>palm*.2 else 'UNKNOWN'

def engine(path):
 import onnxruntime as ort
 from rtmlib import RTMPose
 if hasattr(ort,'preload_dlls'):ort.preload_dlls(directory='')
 model=RTMPose(str(path),model_input_size=(256,256),backend='onnxruntime',device='cuda',to_openpose=False)
 model.session.disable_fallback()
 if 'CUDAExecutionProvider' not in model.session.get_providers():raise ValueError('手部专用模型未使用 CUDA')
 return model

def detect(data,side,model):
 im=local.worker.prepare_image(data['image']['path']);w,h=im.size;b=data.get('wholebody',[])
 if len(b)!=133:raise ValueError('GPU 手部裁剪需要全身模型的定位点')
 start=91 if side=='L' else 112;wi=15 if side=='L' else 16;oi=16 if side=='L' else 15
 refs=np.array([[v['x']*w,v['y']*h] for v in b[start:start+21]]);wrist=np.array([data['landmarks'][wi]['x']*w,data['landmarks'][wi]['y']*h]);other=np.array([data['landmarks'][oi]['x']*w,data['landmarks'][oi]['y']*h]);candidates=[]
 for padding in (2.,3.2):
  box=local.square_box(refs,padding);roi=local.crop(im,box);size=box[2]-box[0]
  for angle in (0,90,180,270):
   rgb=np.asarray(roi.rotate(angle,resample=2));points,scores=model(np.ascontiguousarray(rgb[:,:,::-1]));q=np.asarray(points[0],float)/512;scores=np.asarray(scores[0],float)
   if q.shape!=(21,2) or scores.shape!=(21,) or not np.isfinite(q).all() or min(scores[[0,5,9,17]])<.3:continue
   rad=np.deg2rad(angle);r=np.array([[np.cos(rad),-np.sin(rad)],[np.sin(rad),np.cos(rad)]])
   q=(q-.5)@r.T+.5;normalized=local.image_points(q,box,w,h);px=normalized*[w,h]
   error=float(np.linalg.norm(px[0]-wrist)/size)
   if error>.28 or (np.linalg.norm(other-wrist)>size*.15 and np.linalg.norm(px[0]-other)<np.linalg.norm(px[0]-wrist)):continue
   palm=np.linalg.norm(px[9]-px[0])
   if palm<6 or palm>size*.65 or max(np.linalg.norm(px-px[0],axis=1))>palm*4:continue
   candidates.append(dict(landmarks=normalized.tolist(),scores=np.clip(scores,0,1).tolist(),wrist_error=error,gesture=suggest_v(px),angle=angle,padding=padding))
 if not candidates:raise ValueError('专用 GPU 手部模型未得到可靠定位')
 # Use rotation consensus; do not label an isolated accidental prediction as V.
 votes=sum(c['gesture']=='V' for c in candidates);v_candidates=[c for c in candidates if c['gesture']=='V'] if votes>=2 else []
 chosen=min(v_candidates or candidates,key=lambda c:c['wrist_error']);result=copy.deepcopy(data)
 for i,(point,score) in enumerate(zip(chosen['landmarks'],chosen['scores'])):result['wholebody'][start+i]=dict(x=point[0],y=point[1],confidence=score)
 info=dict(source='rtmpose_hand_crop_2d',device='cuda',providers=model.session.get_providers(),candidate_count=len(candidates),gesture_suggestion='V' if votes>=2 else 'UNKNOWN',gesture_vote_count=votes,landmarks=chosen['landmarks'],scores=chosen['scores'],wrist_error=chosen['wrist_error'],depth_available=False)
 return result,info
