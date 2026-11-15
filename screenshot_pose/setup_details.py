"""Download and verify optional local face/hand task models; no image access."""
import argparse,hashlib,json,urllib.request
from pathlib import Path
MODELS={name:f'https://storage.googleapis.com/mediapipe-models/{stem}/{stem}/float16/1/{name}' for name,stem in [('face_landmarker.task','face_landmarker'),('hand_landmarker.task','hand_landmarker')]}
def install(root):
 root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True);report={}
 for name,url in MODELS.items():
  dst=root/name;tmp=root/(name+'.download')
  if not dst.is_file():
   try:
    with urllib.request.urlopen(url,timeout=60) as response,tmp.open('wb') as out:
     size=0
     while chunk:=response.read(1024*1024):
      size+=len(chunk)
      if size>100*1024*1024:raise ValueError('Unexpected model size')
      out.write(chunk)
    if tmp.stat().st_size<1000000:raise ValueError('Model download incomplete')
    tmp.replace(dst)
   finally:tmp.unlink(missing_ok=True)
  report[name]={'url':url,'sha256':hashlib.sha256(dst.read_bytes()).hexdigest()}
 import mediapipe as mp
 for name,cls,opts in [('face_landmarker.task',mp.tasks.vision.FaceLandmarker,mp.tasks.vision.FaceLandmarkerOptions),('hand_landmarker.task',mp.tasks.vision.HandLandmarker,mp.tasks.vision.HandLandmarkerOptions)]:
  with cls.create_from_options(opts(base_options=mp.tasks.BaseOptions(model_asset_buffer=(root/name).read_bytes()))):pass
 (root/'details_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('LOCAL_DETAILS_READY')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--runtime',required=True);a=p.parse_args();install(a.runtime)
