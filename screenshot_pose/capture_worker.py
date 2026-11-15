"""Capture a region, then infer with the selected local body model in one job."""
import argparse,json,sys
from pathlib import Path
try:
 from . import worker,screen_capture
except ImportError:
 import worker,screen_capture

def run(image_output,output,overlay,status,model,backend='MEDIAPIPE',delay=0,person_index=0,max_poses=4,confidence=.5,capture_fn=None,detect_fn=None,native_root_path=None,hand_mode='OFF',progress=None):
 paths=[Path(p).resolve() for p in (image_output,output,overlay,status)]
 if len(set(paths))!=4 or Path(model).resolve() in paths:raise ValueError('截图、姿势、预览、状态及模型路径不能重合')
 if not Path(model).is_file():raise ValueError('请先准备本地识别模型')
 if backend not in ('MEDIAPIPE','RTMW','RTMW3D','SAM3D'):raise ValueError('识别模型类型无效')
 if hand_mode not in ('OFF','AUTO','HAMER','HAMER_TEXTURE'):raise ValueError('截图手部模式无效')
 if not 0<=delay<=10:raise ValueError('截图等待时间应为 0 至 10 秒')
 worker.validate_options(person_index,max_poses,confidence)
 image=(capture_fn or screen_capture.capture)(delay)
 if image is None:worker.write_json(status,{'cancelled':True});return None
 worker.save_png(image,image_output)
 if detect_fn is None and hand_mode!='OFF':
  try:from .worker_screenshot import detect
  except ImportError:from worker_screenshot import detect
  document=detect(image_output,model,output,overlay,backend=backend,hand_mode=hand_mode,person_index=person_index,max_poses=max_poses,confidence=confidence,native_root_path=native_root_path,progress=progress)
  worker.write_json(status,{'cancelled':False,'image':str(Path(image_output).resolve())});return document
 if detect_fn is None:
  if backend=='SAM3D':
   try:from .worker_sam import detect as sam
   except ImportError:from worker_sam import detect as sam
   detect_fn=sam
  elif backend=='RTMW3D':
   try:from .worker_gpu import detect as gpu
   except ImportError:from worker_gpu import detect as gpu
   detect_fn=gpu
  elif backend=='RTMW':
   try:from .worker_high import detect as high
   except ImportError:from worker_high import detect as high
   detect_fn=high
  else:detect_fn=worker.detect
 extra={'native_root_path':native_root_path} if backend=='SAM3D' and native_root_path else {}
 document=detect_fn(image_output,model,output,overlay,person_index=person_index,max_poses=max_poses,confidence=confidence,**extra)
 worker.write_json(status,{'cancelled':False,'image':str(Path(image_output).resolve())});return document

def main(argv=None,capture_fn=None):
 p=argparse.ArgumentParser()
 for key in ('image-output','output','overlay','status','model'):p.add_argument('--'+key,required=True)
 p.add_argument('--backend',default='MEDIAPIPE',choices=['MEDIAPIPE','RTMW','RTMW3D','SAM3D']);p.add_argument('--delay',type=int,default=0);p.add_argument('--native-root-path')
 p.add_argument('--person-index',type=int,default=0);p.add_argument('--max-poses',type=int,default=4);p.add_argument('--confidence',type=float,default=.5)
 p.add_argument('--hand-mode',choices=['OFF','AUTO','HAMER','HAMER_TEXTURE'],default='OFF');p.add_argument('--progress')
 try:run(**vars(p.parse_args(argv)),capture_fn=capture_fn);return 0
 except Exception as exc:print('截图识别失败：'+str(exc),file=sys.stderr);return 1

if __name__=='__main__':raise SystemExit(main())
