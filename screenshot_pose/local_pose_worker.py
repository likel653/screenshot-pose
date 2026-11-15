"""Standalone local face / cropped hand inference. No downloads or uploads."""
import argparse,copy,json,math,sys
from pathlib import Path
import numpy as np
try:
 from . import worker,pose_data,orientation_math as om,head_math
except ImportError:
 import worker,pose_data,orientation_math as om,head_math

def square_box(xy,padding=1.6):
 p=np.asarray(xy);lo=p.min(0);hi=p.max(0);center=(lo+hi)/2;size=max(hi-lo)*padding
 if size<12:raise ValueError('局部区域太小，无法可靠估计朝向')
 return [float(center[0]-size/2),float(center[1]-size/2),float(center[0]+size/2),float(center[1]+size/2)]

def crop(im,box):return im.crop(tuple(round(x) for x in box)).resize((512,512))
def image_points(points,box,w,h):
 p=np.asarray(points);size=box[2]-box[0]
 return (p*size+np.array(box[:2]))/np.array([w,h])

def face_fallback(d,w,h,calibrated=True):
 """Generic six-point PnP candidate for RTMW face points, used only on failure."""
 import cv2
 b=d.get('wholebody',[])
 if len(b)!=133:return None
 ids=[0,2,1,31,71,77]
 if min(b[i]['confidence'] for i in ids)<.45:return None
 if calibrated:
  ref=head_math.reference(d)
  if ref is None:return None
  obj=head_math.TEMPLATE;xy=ref['xy'];box=square_box(xy,1.8);size=box[2]-box[0];local=xy
  intrinsics,provenance=head_math.camera(d,ref);fx,fy,cx,cy=intrinsics
  K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]],float);error_scale=size
 else:
  # Preserve the pre-v0.34 fallback for direct/video infer() callers.
  obj=np.array([[0,0,0],[-30,28,-25],[30,28,-25],[0,-60,-15],[-23,-28,-15],[23,-28,-15]],float)
  xy=np.array([[b[i]['x']*w,b[i]['y']*h] for i in ids]);box=square_box(xy,1.8);size=box[2]-box[0];local=(xy-np.array(box[:2]))/size*512
  K=np.array([[650,0,256],[0,650,256],[0,0,1.]],float);intrinsics=np.array([650,650,256,256]);error_scale=512
  provenance=dict(source='legacy_crop_focal',calibrated=False)
 ok,rv,tv=cv2.solvePnP(obj,local,K,None,flags=cv2.SOLVEPNP_EPNP)
 if not ok:return None
 ok,rv,tv=cv2.solvePnP(obj,local,K,None,rv,tv,True,flags=cv2.SOLVEPNP_ITERATIVE)
 proj,_=cv2.projectPoints(obj,rv,tv,K,None);error=float(np.sqrt(np.mean(np.sum((proj[:,0]-local)**2,axis=1)))/error_scale)
 R,_=cv2.Rodrigues(rv)
 if not ok or error>.1 or tv[2,0]<0:return None
 return dict(frame=om.proper_rotation(om.C_DOWN@R).tolist(),source='rtmw_6point_pnp',confidence=max(.35,1-error*5),reprojection_error=error,box=box,camera=provenance,intrinsics=intrinsics.tolist(),
             warning='通用脸形六点拟合；夸张二次元比例可能存在角度偏差')

def palm_fallback(data,side,w,h):
 import cv2
 b=data.get('wholebody',[])
 if len(b)!=133:return None
 start=91 if side=='L' else 112;ids=[0,5,9,13,17]
 if min(b[start+i]['confidence'] for i in ids)<.4:return None
 template=np.zeros((21,3));template[5]=[.035,.065,0];template[9]=[0,.075,0];template[13]=[-.018,.07,0];template[17]=[-.035,.055,0]
 xy=np.array([[b[start+i]['x']*w,b[start+i]['y']*h] for i in ids]);box=square_box(xy,2.5);size=box[2]-box[0]
 image=(xy-np.array(box[:2]))/size*512;K=np.array([[600,0,256],[0,600,256],[0,0,1.]],float)
 out=cv2.solvePnPGeneric(template[ids],image,K,None,flags=cv2.SOLVEPNP_IPPE)
 candidates=[]
 for rv,tv in zip(out[1],out[2]):
  if tv[2,0]<=0:continue
  projected,_=cv2.projectPoints(template[ids],rv,tv,K,None);error=float(np.sqrt(np.mean(np.sum((projected[:,0]-image)**2,axis=1)))/512)
  if error>.09:continue
  R,_=cv2.Rodrigues(rv);frame=om.proper_rotation(om.C_DOWN@R@om.palm_frame(template))
  candidates.append(dict(frame=frame.tolist(),source='rtmw_palm_planar_pnp',confidence=float(np.clip(1-error*6,.35,.7)),reprojection_error=error,box=box,anatomical_side=side))
 if not candidates:return None
 candidates.sort(key=lambda c:c['reprojection_error']);best=candidates[0];best['alternatives']=[c['frame'] for c in candidates[1:]]
 best['warning']='遮挡手使用通用掌面几何拟合，深度与掌背存在歧义，可切换候选'
 return best

def infer(data,root,engines=None,continuity=None,head_only=False,head_precision=False):
 from contextlib import nullcontext
 import mediapipe as mp
 head_precision=bool(head_precision or head_only)
 im=worker.prepare_image(data['image']['path']);w,h=im.size
 if (w,h)!=(data['image']['width'],data['image']['height']):raise ValueError('参考图片尺寸已改变，请重新识别')
 result=copy.deepcopy(data);local={'schema':'screenshot_pose/orientation_v1','camera_axes':'X_right_Y_away_Z_up','hands':{},'warnings':[]}
 base=data.get('orientation',{})
 if data.get('source')=='rtmw3d_wholebody':
  local['feet']=copy.deepcopy(base.get('feet',{}))
  local['hands']={side:copy.deepcopy(h) for side,h in base.get('hands',{}).items() if h.get('source')=='rtmw3d_hand'}
  local['warnings'].extend(base.get('warnings',[]))
 root=Path(root)
 for name in (('face_landmarker.task',) if head_only else ('face_landmarker.task','hand_landmarker.task')):
  if not (root/name).is_file():raise ValueError('请先点击“准备头部 / 手部模型”')
 b=data.get('wholebody',[]);p=data['landmarks'];edited=set(data.get('edited_indices',[]));face_edited=bool(edited.intersection((0,7,8)))
 def bodyxy(indices):return [[p[i]['x']*w,p[i]['y']*h] for i in indices]
 face_xy=[[v['x']*w,v['y']*h] for v in b[23:91] if v['confidence']>.5] if len(b)==133 else []
 if len(face_xy)<6 or face_edited:face_xy=bodyxy((0,7,8))
 face_options=mp.tasks.vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_buffer=(root/'face_landmarker.task').read_bytes()),num_faces=1,min_face_detection_confidence=.3,min_face_presence_confidence=.3,output_facial_transformation_matrixes=True)
 faces=[]
 create=lambda:mp.tasks.vision.FaceLandmarker.create_from_options(face_options)
 manager=nullcontext(engines.get(('face_image_cpu',str(root)),create)) if engines else create()
 with manager as detector:
  for padding in (1.55,2.):
   try:
    box=square_box(face_xy,padding);roi=crop(im,box);r=detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.array(roi)))
    if not r.face_landmarks:continue
    landmarks=r.face_landmarks[0];xy=image_points([[a.x,a.y] for a in landmarks],box,w,h)
    refs=[(0,1),(11,263),(12,33)]
    # Compare nose/eye positions, not body shoulders; face point fits use RTMW eyes.
    if len(b)==133 and not face_edited:
     error=float(np.mean([np.linalg.norm((xy[j]-[b[i]['x'],b[i]['y']])*[w,h]) for i,j in [(0,1),(1,263),(2,33)]])/(box[2]-box[0]))
    else:error=.08
    R=om.proper_rotation(np.asarray(r.facial_transformation_matrixes[0])[:3,:3])
    if error>.25:continue
    faces.append(dict(frame=(om.C_FACE@R).tolist(),source='mediapipe_face_matrix',confidence=float(np.clip(1-error*3,0,1)),reprojection_error=error,box=box,landmarks=xy.tolist()))
   except ValueError:continue
 try:pnp=face_fallback(data,w,h,calibrated=head_precision) if not face_edited and (head_precision or not faces) else None
 except (ValueError,RuntimeError):pnp=None
 if head_precision:
  local['head_candidates']=faces+([pnp] if pnp else [])
  local['head_before_refinement']=copy.deepcopy(base.get('head_before_refinement',base.get('head')))
 local['head']=max(faces,key=lambda x:x['confidence']) if faces else pnp
 if not local.get('head'):local['warnings'].append('头部局部识别失败，保留身体模型的头部姿势')
 if head_only:
  merged=copy.deepcopy(base);merged.update({k:v for k,v in local.items() if k!='hands'})
  merged['hands']=copy.deepcopy(base.get('hands',{}));result['orientation']=merged;return result
 opts=mp.tasks.vision.HandLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_buffer=(root/'hand_landmarker.task').read_bytes()),num_hands=2,min_hand_detection_confidence=.25,min_hand_presence_confidence=.25)
 create=lambda:mp.tasks.vision.HandLandmarker.create_from_options(opts)
 manager=nullcontext(engines.get(('hand_image_cpu',str(root)),create)) if engines else create()
 with manager as detector:
  for side,start,wrist in [('L',91,15),('R',112,16)]:
   hand_edited=bool(edited.intersection((wrist,wrist+2,wrist+4,wrist+6)))
   refs=np.array([[v['x'],v['y']] for v in b[start:start+21]]) if len(b)==133 and not hand_edited else None
   scores=np.array([v['confidence'] for v in b[start:start+21]]) if len(b)==133 else None
   xy=refs[scores>.4]*[w,h] if refs is not None else np.array(bodyxy((wrist,wrist+2,wrist+4,wrist+6)))
   candidates=[]
   for padding in (1.6,2.2):
    try:box=square_box(xy,padding)
    except ValueError:continue
    roi=crop(im,box)
    for angle in (0,90,180,270):
     image=roi.rotate(angle,resample=2);r=detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.array(image)))
     for k,landmarks in enumerate(r.hand_landmarks):
      world=np.array([[p.x,p.y,p.z] for p in r.hand_world_landmarks[k]])
      xyroi=np.array([[p.x,p.y] for p in landmarks]);world,unrot=om.unrotate_hand(world,xyroi,angle)
      normalized=image_points(unrot,box,w,h)
      if refs is not None:
       mask=scores>.4;error=float(np.mean(np.linalg.norm((normalized[mask]-refs[mask])*[w,h],axis=1))/(box[2]-box[0]))
      else:error=float(np.linalg.norm((normalized[0]-[p[wrist]['x'],p[wrist]['y']])*[w,h])/(box[2]-box[0]))
      if error>.2:continue
      try:frame=om.palm_frame(world);curl=om.curls(world)
      except ValueError:continue
      hand=r.handedness[k][0]
      candidates.append(dict(frame=frame.tolist(),world=world.tolist(),landmarks=normalized.tolist(),curls=curl,source='mediapipe_hand_world',confidence=float(np.clip(1-error*4,0,1)),reprojection_error=error,box=box,crop_rotation=angle,predicted_handedness=hand.category_name,anatomical_side=side))
   if candidates and max(c['confidence'] for c in candidates)>=.45:
    candidates.sort(key=lambda c:c['reprojection_error']);local['hands'][side]=continuity.choose_hand(side,candidates) if continuity else candidates[0]
    local['hands'][side]['candidate_count']=len(candidates)
   else:
    if side in local['hands'] and not hand_edited:continue
    local['hands'].pop(side,None)
    try:fallback=palm_fallback(data,side,w,h) if not hand_edited else None
    except (ValueError,RuntimeError):fallback=None
    if fallback:
     local['hands'][side]=fallback;local['warnings'].append(side+'手使用掌面几何候选，请核对掌心方向')
    else:local['warnings'].append(side+'手局部识别失败；保持当前手腕，仍可手动翻转/微调')
 result['orientation']=local
 return result

def run(input,output,model_dir,head_only=False,head3d_config=None):
 data=pose_data.read_pose(input)
 if Path(input).resolve()==Path(output).resolve():raise ValueError('局部识别输出不能覆盖输入 JSON')
 result=infer(data,model_dir,head_only=head_only,head_precision=True);worker.write_json(output,result)
 if head3d_config:
  import subprocess
  cfg=json.loads(Path(head3d_config).read_text(encoding='utf8'));gpu=Path(output).with_suffix('.head3d.json');log=Path(output).with_suffix('.head3d.log')
  try:
   with log.open('w',encoding='utf8') as stream:
    subprocess.run([cfg['python'],str(Path(__file__).with_name('worker_head3d.py')),'--input',str(output),'--output',str(gpu),'--config',str(head3d_config)],check=True,timeout=120,stdout=stream,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
   result=pose_data.read_pose(gpu)
  except (OSError,subprocess.SubprocessError) as exc:
   result['orientation']['warnings'].append('独立头部 GPU 候选失败，保留局部与原头向：'+str(exc));result['head3d_runtime']=dict(actual_inference=False,error=str(exc),log=str(log))
  worker.write_json(output,result)
 print(json.dumps({'ok':True,'head':bool(result['orientation'].get('head')),'hands':list(result['orientation']['hands']),'head_only':head_only}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--model-dir',required=True)
 p.add_argument('--head-only',action='store_true')
 p.add_argument('--head3d-config')
 try:run(**vars(p.parse_args()))
 except Exception as exc:print('局部识别失败：'+str(exc),file=sys.stderr);sys.exit(1)
