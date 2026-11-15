"""Independent hand crops and optional aligned auxiliary views; local execution."""
import argparse,copy,hashlib,json,sys,time
from pathlib import Path
import numpy as np
try:from . import worker,pose_data,local_pose_worker as local,orientation_math as om,multi_view,worker_multiview as mv,worker_sam,hand_geometry as hg,hand_crop_gpu
except ImportError:import worker,pose_data,local_pose_worker as local,orientation_math as om,multi_view,worker_multiview as mv,worker_sam,hand_geometry as hg,hand_crop_gpu

def reference(image,person,model,confidence,native):
 digest=mv.pixels(image);key=hashlib.sha256(json.dumps([digest,person,confidence,mv.fingerprint(model,native)]).encode()).hexdigest()
 folder=Path(model).resolve().parent/'multiview_cache'/key;pose=folder/'pose.json'
 if pose.is_file():
  try:d=pose_data.read_pose(pose);multi_view.skeleton(d);return d,True
  except (ValueError,KeyError,OSError):pass
 folder.mkdir(parents=True,exist_ok=True)
 return worker_sam.detect(image,model,pose,folder/'overlay.png',person_index=person,confidence=confidence,native_root_path=native),False

def infer_hand(data,side,detector,mp):
 im=worker.prepare_image(data['image']['path']);w,h=im.size
 b=data.get('wholebody',[]);start=91 if side=='L' else 112;wi=15 if side=='L' else 16;other=16 if side=='L' else 15
 if len(b)==133:
  refs=np.array([[v['x'],v['y']] for v in b[start:start+21]]);confidence=np.array([v['confidence'] for v in b[start:start+21]])
  selected=refs[confidence>.35]*[w,h]
 else:
  refs=None;selected=np.array([[data['landmarks'][j]['x']*w,data['landmarks'][j]['y']*h] for j in (wi,wi+2,wi+4,wi+6)])
 if len(selected)<3:raise ValueError('可靠的手部定位点不足')
 wrist=np.array([data['landmarks'][wi]['x']*w,data['landmarks'][wi]['y']*h]);other_wrist=np.array([data['landmarks'][other]['x']*w,data['landmarks'][other]['y']*h])
 candidates=[];attempts=0
 for padding in (2.,3.2):
  box=local.square_box(selected,padding);roi=local.crop(im,box)
  for angle in (0,45,90,135,180,225,270,315):
   attempts+=1;r=detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.array(roi.rotate(angle,resample=2))))
   for k,landmarks in enumerate(r.hand_landmarks):
    points=np.array([[v.x,v.y,v.z] for v in r.hand_world_landmarks[k]]);xy=np.array([[v.x,v.y] for v in landmarks]);points,unrot=om.unrotate_hand(points,xy,angle)
    normalized=local.image_points(unrot,box,w,h);position=normalized[0]*[w,h];size=box[2]-box[0]
    if np.linalg.norm(position-wrist)>size*.28:continue
    if np.linalg.norm(other_wrist-wrist)>size*.15 and np.linalg.norm(position-other_wrist)<np.linalg.norm(position-wrist):continue
    anchors=[0,5,9,17];err=float(np.mean(np.linalg.norm((normalized[anchors]-refs[anchors])*[w,h],axis=1))/size) if refs is not None else float(np.linalg.norm(position-wrist)/size)
    if err>.22:continue
    try:
     before=hg.suggest(points);points,fit=hg.refine(points,unrot,before);frame=om.palm_frame(points);curl=om.curls(points)
    except ValueError:continue
    candidates.append(dict(frame=frame.tolist(),world=points.tolist(),landmarks=normalized.tolist(),curls=curl,source='cropped_hand_refined',confidence=float(np.clip(1-err*4,0,.95)),reprojection_error=err,box=box,crop_rotation=angle,anatomical_side=side,gesture_suggestion=hg.suggest(points),fit=fit))
 if not candidates:return None,dict(attempts=attempts,candidates=0)
 candidates.sort(key=lambda v:v['reprojection_error']);return candidates[0],dict(attempts=attempts,candidates=len(candidates))

def run(input,request,model,output,native_root_path=None):
 if Path(request).stat().st_size>65536:raise ValueError('手部请求过大')
 requested=json.loads(Path(request).read_text(encoding='utf-8'))
 if isinstance(requested,dict) and any(isinstance(v,dict) and v.get('backend','LOCAL')!='LOCAL' for v in requested.values()):
  try:from .worker_hand3d import run as mesh_run
  except ImportError:from worker_hand3d import run as mesh_run
  return mesh_run(input,request,model,output,native_root_path)
 import mediapipe as mp
 source=Path(input).resolve();out=Path(output).resolve();req=Path(request).resolve();report_path=out.with_suffix('.report.json');base=pose_data.read_pose(source)
 if req.stat().st_size>65536:raise ValueError('手部请求过大')
 selections=json.loads(req.read_text(encoding='utf-8'))
 if not isinstance(selections,dict) or not selections or any(s not in ('L','R') for s in selections):raise ValueError('请选择有效的左右手')
 if any(not isinstance(v,dict) or v.get('gesture','NONE') not in ('NONE','V','GRIP','OPEN','FIST') for v in selections.values()):raise ValueError('手部参数无效')
 protected={source,req,Path(model).resolve(),Path(model).resolve().parent/'hand_landmarker.task',Path(model).resolve().parent/'rtmpose_hand.onnx',Path(base['image']['path']).resolve()}|{Path(v['image']).resolve() for v in selections.values() if v.get('image')}
 if out==report_path or {out,report_path}&protected:raise ValueError('手部输出不能覆盖输入')
 asset=Path(model).resolve().parent/'hand_landmarker.task'
 if not asset.is_file():raise ValueError('缺少手部局部模型，请先准备头部 / 手部模型')
 result=copy.deepcopy(base);orientation=result.setdefault('orientation',dict(schema='screenshot_pose/orientation_v1',camera_axes='X_right_Y_away_Z_up',hands={},warnings=[]));orientation.setdefault('hands',{})
 report=dict(primary_json=str(source),primary_image=base['image']['path'],hands={},body_preserved=True,auto_applied=False);start=time.perf_counter();native=Path(native_root_path) if native_root_path else worker_sam.native_root();references={};gpu=None
 gpu_path=Path(model).resolve().parent/'rtmpose_hand.onnx'
 if any(v.get('gpu_2d',False) for v in selections.values()):
  if not gpu_path.is_file():raise ValueError('请先准备可选的 GPU 手部二维模型')
  gpu=hand_crop_gpu.engine(gpu_path)
 options=mp.tasks.vision.HandLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_buffer=asset.read_bytes()),num_hands=2,min_hand_detection_confidence=.2,min_hand_presence_confidence=.2)
 with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
  for side,selection in selections.items():
   info=dict(reference=selection.get('image') or base['image']['path'],gesture=selection.get('gesture','NONE'))
   try:
    image=selection.get('image');reference_data=base;rotation=np.eye(3);cache=False
    if image and mv.pixels(image)==mv.pixels(base['image']['path']) and int(selection.get('person_index',1))-1!=base.get('person_index',0):raise ValueError('相同参考图的人物序号与主图不一致')
    if image and mv.pixels(image)!=mv.pixels(base['image']['path']):
     person=int(selection.get('person_index',1))-1;worker.validate_options(person,4,.5)
     cache_key=(str(Path(image).resolve()),person)
     if cache_key not in references:references[cache_key]=reference(image,person,model,.5,native)
     reference_data,cache=references[cache_key];alignment=multi_view.align(base,reference_data);rotation=np.array(alignment['rotation']);info['alignment']=alignment
    hand_input=reference_data;gpu_info=None
    if gpu and selection.get('gpu_2d',False):
     try:hand_input,gpu_info=hand_crop_gpu.detect(reference_data,side,gpu);info['gpu_2d']=gpu_info
     except ValueError as exc:info['gpu_2d_warning']=str(exc)
    observation,stats=infer_hand(hand_input,side,detector,mp);info.update(stats,cache_hit=cache)
    if observation is None:
     if gpu_info:
      observation=local.palm_fallback(hand_input,side,hand_input['image']['width'],hand_input['image']['height'])
      if observation:observation.update(source='rtmpose_hand_crop_2d_palm',gesture_suggestion=gpu_info['gesture_suggestion'],gesture_vote_count=gpu_info['gesture_vote_count'],landmarks=gpu_info['landmarks'])
     if observation is None:observation=copy.deepcopy(reference_data.get('orientation',{}).get('hands',{}).get(side))
     info['fallback']=True
    if not observation:raise ValueError('未检测到可靠手掌；保留主图，可手动指定手势约束')
    if observation.get('confidence',0)<.45:raise ValueError('手部结果质量不足，保留主图')
    transformed=hg.transport(observation,rotation);transformed['reference_image']=reference_data['image']['path'];transformed['reference_original']=info['reference']
    transformed['gesture_constraint']=selection.get('gesture','NONE');orientation['hands'][side]=transformed
    info.update(adopted=True,source=observation['source'],has_21_points='world' in observation,confidence=observation.get('confidence',0),suggestion=observation.get('gesture_suggestion','UNKNOWN'),fit=observation.get('fit'))
   except Exception as exc:info.update(adopted=False,reason=str(exc)[:300])
   report['hands'][side]=info
 report['seconds']=time.perf_counter()-start;result['hand_refinement']=report
 worker.write_json(out,pose_data.validate(result));worker.write_json(report_path,report)
 print(json.dumps(dict(ok=True,adopted=sum(v['adopted'] for v in report['hands'].values()),report=str(report_path))),flush=True);return result

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for key in ('input','request','model','output'):p.add_argument('--'+key,required=True)
 p.add_argument('--native-root-path')
 try:run(**vars(p.parse_args()))
 except Exception as exc:print('手部精修失败：'+str(exc),file=sys.stderr);raise SystemExit(1)
