"""RTMW3D full-body inference with verified CUDA sessions and metric lifting."""
import argparse,json,time
from pathlib import Path
import numpy as np
try:
 from . import worker,worker_high,pose_data,orientation_math as om
except ImportError:
 import worker,worker_high,pose_data,orientation_math as om

def lift_points(xy,depth,size):
 xy=np.asarray(xy,float);depth=np.asarray(depth,float)
 if xy.shape!=(133,2) or depth.shape!=(133,) or not np.isfinite(xy).all() or not np.isfinite(depth).all():raise ValueError('RTMW3D 结果坐标无效')
 depth=depth-depth[[11,12]].mean();pairs=[(5,6,.38),(5,7,.28),(6,8,.28),(7,9,.25),(8,10,.25),(11,13,.43),(12,14,.43),(13,15,.43),(14,16,.43)]
 scales=[]
 for a,b,length in pairs:
  dz=abs(depth[a]-depth[b]);pixel=np.linalg.norm(xy[a]-xy[b])
  if dz<length*.9 and pixel>20:scales.append(pixel/np.sqrt(length*length-dz*dz))
 torso=np.linalg.norm(xy[[5,6]].mean(0)-xy[[11,12]].mean(0))
 if torso>20:scales.append(torso/.52)
 ppm=float(np.median(scales)) if scales else max(size)/1.8
 focal=1.2*max(size);root_distance=focal/max(ppm,1);z=np.maximum(root_distance+depth,.1)
 center=np.array(size)/2;world=np.column_stack(((xy-center)/focal*z[:,None],z))
 world-=world[[11,12]].mean(0)
 return world,dict(focal=focal,root_distance=root_distance,pixels_per_meter=ppm,method='anatomical_scale_estimate_not_camera_calibration')

def make_document(xy,raw,scores,size,image,original):
 data=worker_high.convert(xy,scores,size);world,camera=lift_points(xy,np.asarray(raw)[:,2],size)
 for i,j in worker_high.MAP.items():
  data['world_landmarks'][i].update(x=float(world[j,0]),y=float(world[j,1]),z=float(world[j,2]),visibility=float(np.clip(scores[j],0,1)),presence=float(np.clip(scores[j],0,1)))
 data.update(source='rtmw3d_wholebody',depth_source='rtmw3d_relative_depth',suggested_mode='HYBRID',image=dict(path=str(image),original_path=str(original),width=size[0],height=size[1]),wholebody_3d=world.tolist(),camera_estimate=camera)
 local={'schema':'screenshot_pose/orientation_v1','camera_axes':'X_right_Y_away_Z_up','hands':{},'feet':{},'warnings':[]}
 camera_points=world@om.C_DOWN.T
 for side,start in [('L',91),('R',112)]:
  conf=float(np.clip(np.min(np.asarray(scores)[[start,start+5,start+9,start+17]]),0,1))
  if conf<.4:local['warnings'].append(side+'手 GPU 关节点置信度不足');continue
  points=camera_points[start:start+21]
  palm=np.linalg.norm(points[[5,9,17]].mean(0)-points[0]);width=np.linalg.norm(points[5]-points[17])
  finger=np.median([np.linalg.norm(points[i+1]-points[i]) for i in (5,6,7,9,10,11,13,14,15,17,18,19)])
  if palm<1e-6 or width/palm<.4 or finger/palm<.15:
   local['warnings'].append(side+'手 GPU 三维比例不稳定，尝试局部模型或掌面几何回退');continue
  try:frame=om.palm_frame(points);curls=om.curls(points)
  except ValueError:continue
  local['hands'][side]=dict(frame=frame.tolist(),world=points.tolist(),landmarks=[[float(v[0]/size[0]),float(v[1]/size[1])] for v in xy[start:start+21]],curls=curls,confidence=conf,source='rtmw3d_hand',anatomical_side=side)
 for side,indices in [('L',(15,17,18,19)),('R',(16,20,21,22))]:
  ankle,big,small,heel=indices;conf=float(np.clip(np.min(np.asarray(scores)[list(indices)]),0,1))
  along=(camera_points[big]+camera_points[small])*.5-camera_points[heel]
  across=(camera_points[big]-camera_points[small])*(1 if side=='L' else -1)
  try:frame=om.frame(across,along)
  except ValueError:continue
  local['feet'][side]=dict(frame=frame.tolist(),confidence=conf,source='rtmw3d_foot',points={n:camera_points[i].tolist() for n,i in [('ankle',ankle),('big_toe',big),('small_toe',small),('heel',heel)]})
 data['orientation']=local
 return data

def detect(image,model,output,overlay,person_index=0,max_poses=4,confidence=.5,device='cuda',profile=False,engines=None,person_box=None,continuity=None):
 import onnxruntime as ort
 try:from .person_detector import make_detector
 except ImportError:from person_detector import make_detector
 from rtmlib.tools.pose_estimation.rtmpose3d import RTMPose3d
 worker.validate_options(person_index,max_poses,confidence)
 if device not in ('cuda','cpu'):raise ValueError('推理设备无效')
 if device=='cuda':
  if hasattr(ort,'preload_dlls'):ort.preload_dlls(directory='')
  if 'CUDAExecutionProvider' not in ort.get_available_providers():raise ValueError('未找到 CUDA 推理环境，请准备 GPU 模型与环境')
 root=Path(model).resolve().parent;original=Path(image).resolve();output=Path(output).resolve();overlay=Path(overlay).resolve();processed=output.with_name(output.stem+'.input.png')
 worker._check_paths(original,Path(model).resolve(),output,overlay,processed)
 for n in ('rtmw_detector.onnx','rtmw3d_pose.onnx'):
  if not (root/n).is_file():raise ValueError('缺少 GPU 三维模型，请先准备 GPU 环境')
  if (root/n) in (output,overlay,processed):raise ValueError('输出不能覆盖模型')
 im=worker.prepare_image(original);bgr=np.ascontiguousarray(np.asarray(im)[:,:,::-1]);start=time.perf_counter()
 create=lambda:make_detector(str(root/'rtmw_detector.onnx'),score_thr=confidence,backend='onnxruntime',device=device)
 detector=engines.get(('yolox_'+device,str(root),confidence),create) if engines else create()
 create=lambda:RTMPose3d(str(root/'rtmw3d_pose.onnx'),backend='onnxruntime',device=device)
 pose=engines.get(('rtmw3d_'+device,str(root)),create) if engines and not profile else create()
 if device=='cuda':
  detector.session.disable_fallback();pose.session.disable_fallback()
 providers={name:t.session.get_providers() for name,t in [('detector',detector),('pose',pose)]}
 if device=='cuda' and any('CUDAExecutionProvider' not in p for p in providers.values()):raise ValueError('CUDA 初始化失败，已停止；没有静默改用 CPU')
 if profile:
  options=ort.SessionOptions();options.enable_profiling=True;output.parent.mkdir(parents=True,exist_ok=True);options.profile_file_prefix=str(output.with_suffix('.profile'))
  pose.session=ort.InferenceSession(str(root/'rtmw3d_pose.onnx'),sess_options=options,providers=['CUDAExecutionProvider','CPUExecutionProvider'] if device=='cuda' else ['CPUExecutionProvider'])
  if device=='cuda':pose.session.disable_fallback()
 ready=time.perf_counter();boxes=detector(bgr) if person_box is None else [person_box]
 if len(boxes)==0:raise ValueError('未检测到人物，请裁剪图片或降低阈值')
 boxes=sorted(boxes,key=lambda b:float((b[2]-b[0])*(b[3]-b[1])),reverse=True)[:max_poses]
 if person_index>=len(boxes):raise ValueError(f'只检测到 {len(boxes)} 人')
 raw,scores,sim,xy=pose(bgr,bboxes=[boxes[person_index]]);elapsed=time.perf_counter()-ready
 if device=='cuda' and any('CUDAExecutionProvider' not in t.session.get_providers() for t in (detector,pose)):raise ValueError('GPU 推理过程中发生 CPU 回退，已停止')
 data=make_document(xy[0],raw[0],scores[0],im.size,processed,original)
 data.update(person_index=person_index,pose_count=len(boxes),runtime=dict(device=device,providers=providers,inference_seconds=elapsed,load_seconds=ready-start,model='RTMW3D-X 384x288'))
 if profile:
  profile_path=pose.session.end_profiling();events=json.loads(Path(profile_path).read_text());counts={}
  for e in events:
   provider=e.get('args',{}).get('provider')
   if provider:counts[provider]=counts.get(provider,0)+1
  data['runtime']['profile_nodes']=counts
  if device=='cuda' and counts.get('CUDAExecutionProvider',0)==0:raise ValueError('执行记录中没有 CUDA 计算节点，已停止')
 # Use the stronger 2D model as a common image observation for depth comparison.
 # Candidates share X/Z observations, so fitting scores compare depth estimates
 # rather than rewarding a model for agreeing with its own detections.
 try:
  reference=worker_high.detect(original,model,output.with_name(output.stem+'.reference.json'),output.with_name(output.stem+'.reference.png'),person_index,max_poses,confidence,engines=engines,person_box=person_box)
  data['body_candidates']=[{'source':'rtmw3d_relative_depth','world_landmarks':data['world_landmarks']}, {'source':reference['depth_source'],'world_landmarks':reference['world_landmarks']}]
  data['landmarks']=reference['landmarks'];data['wholebody']=reference['wholebody'];data['reference_2d_source']='rtmw_384x288'
 except Exception as exc:data.setdefault('backend_warnings',[]).append('二维参考比较不可用，仅使用 GPU 三维结果：'+str(exc)[:100])
 worker.save_png(im,processed)
 if all((root/n).is_file() for n in ('face_landmarker.task','hand_landmarker.task')):
  try:
   try:from .local_pose_worker import infer
   except ImportError:from local_pose_worker import infer
   data=infer(data,root,engines=engines,continuity=continuity)
  except Exception as exc:data.setdefault('backend_warnings',[]).append('局部精修不可用，保留 GPU 三维结果：'+str(exc)[:100])
 data['runtime']['total_seconds']=time.perf_counter()-start
 worker.save_png(worker.draw_overlay(im,data['landmarks'],.35,person_index),overlay);worker.write_json(output,data);return data

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ('image','model','output','overlay'):p.add_argument('--'+n,required=True)
 p.add_argument('--person-index',type=int,default=0);p.add_argument('--max-poses',type=int,default=4);p.add_argument('--confidence',type=float,default=.5);p.add_argument('--device',choices=['cuda','cpu'],default='cuda');p.add_argument('--profile',action='store_true')
 try:
  d=detect(**vars(p.parse_args()));print(json.dumps(d['runtime']))
 except Exception as exc:print('GPU 三维识别失败：'+str(exc),file=__import__('sys').stderr);raise SystemExit(1)
