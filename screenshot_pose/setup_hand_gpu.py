"""Fetch the official RTMPose hand-only 2D model, then verify CUDA loading."""
import argparse,hashlib,json,urllib.request,zipfile
from pathlib import Path
URL='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip'
def install(root):
 root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True);dst=root/'rtmpose_hand.onnx';archive=root/'rtmpose_hand.download'
 if not dst.is_file():
  try:
   with urllib.request.urlopen(URL,timeout=45) as response,archive.open('wb') as stream:
    count=0
    while chunk:=response.read(1024*1024):
     count+=len(chunk)
     if count>150*1024*1024:raise ValueError('手部模型包过大')
     stream.write(chunk)
   with zipfile.ZipFile(archive) as z:
    names=[n for n in z.namelist() if n.endswith('.onnx')]
    if len(names)!=1 or z.getinfo(names[0]).file_size>150*1024*1024:raise ValueError('模型包内容无效')
    content=z.read(names[0])
    if len(content)<1000000:raise ValueError('模型下载不完整')
   temporary=dst.with_suffix('.tmp');temporary.write_bytes(content);temporary.replace(dst)
  finally:archive.unlink(missing_ok=True)
 import onnxruntime as ort
 if hasattr(ort,'preload_dlls'):ort.preload_dlls(directory='')
 session=ort.InferenceSession(str(dst),providers=['CUDAExecutionProvider','CPUExecutionProvider']);session.disable_fallback()
 if 'CUDAExecutionProvider' not in session.get_providers():raise ValueError('手部模型未能使用 CUDA')
 report=dict(model='RTMPose-M Hand5 256x256, 2D only',url=URL,sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),providers=session.get_providers())
 (root/'hand_gpu_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('HAND_GPU_READY',json.dumps(report))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--runtime',required=True);install(p.parse_args().runtime)
