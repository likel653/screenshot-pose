"""Verify real CUDA kernels, never accepting a silent CPU-only fallback."""
import argparse,json,tempfile
from pathlib import Path

def verify(model,output=None):
 import numpy as np,onnxruntime as ort
 ort.preload_dlls(directory='')
 if 'CUDAExecutionProvider' not in ort.get_available_providers():raise RuntimeError('CUDA provider 不可用')
 with tempfile.TemporaryDirectory(prefix='screenshot_pose_gpu_') as tmp:
  options=ort.SessionOptions();options.enable_profiling=True;options.profile_file_prefix=str(Path(tmp)/'probe')
  session=ort.InferenceSession(str(model),sess_options=options,providers=['CUDAExecutionProvider','CPUExecutionProvider']);session.disable_fallback()
  if 'CUDAExecutionProvider' not in session.get_providers():raise RuntimeError('CUDA 库加载失败')
  session.run(None,{session.get_inputs()[0].name:np.zeros((1,3,384,288),dtype=np.float32)})
  profile=Path(session.end_profiling());events=json.loads(profile.read_text());counts={}
  for e in events:
   provider=e.get('args',{}).get('provider')
   if provider:counts[provider]=counts.get(provider,0)+1
  del session
  if not counts.get('CUDAExecutionProvider'):raise RuntimeError('执行记录没有 CUDA 节点')
  report={'ready':True,'onnxruntime':ort.__version__,'providers':counts}
 if output:Path(output).write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(json.dumps(report));return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output');a=p.parse_args();verify(a.model,a.output)
