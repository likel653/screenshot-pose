"""Private CUDA runtime installer plus explicit model download and kernel probe."""
import argparse,hashlib,json,os,subprocess,sys,urllib.request,venv
from pathlib import Path
URL='https://huggingface.co/Soykaf/RTMW3D-x/resolve/main/onnx/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.onnx'
def download(url,path):
 path=Path(path)
 if path.is_file():return
 tmp=path.with_suffix('.download')
 try:
  with urllib.request.urlopen(url,timeout=90) as src,tmp.open('wb') as out:
   size=0
   while chunk:=src.read(1024*1024):
    size+=len(chunk)
    if size>1500*1024*1024:raise ValueError('Unexpected model size')
    out.write(chunk)
  if tmp.stat().st_size<1000000:raise ValueError('Incomplete model')
  tmp.replace(path)
 finally:tmp.unlink(missing_ok=True)
def install(runtime,env=None,skip_install=False):
 root=Path(runtime).resolve();root.mkdir(parents=True,exist_ok=True);here=Path(__file__).resolve().parent
 env=Path(env).resolve() if env else root/'gpu_env';python=env/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
 if not python.is_file():venv.EnvBuilder(with_pip=True).create(env)
 if not skip_install:
  subprocess.run([str(python),'-m','pip','install','--disable-pip-version-check','-r',str(here/'requirements_gpu.txt')],check=True)
  # RTMLib declares the CPU package; both distributions share the Python namespace.
  # Matching versions plus a final GPU reinstall ensure the CUDA build owns it.
  subprocess.run([str(python),'-m','pip','install','--disable-pip-version-check','--force-reinstall','--no-deps','onnxruntime-gpu==1.23.2'],check=True)
 from setup_high import install as models2d
 models2d(root,python,skip_deps=True)
 from setup_runtime import MODEL_URL
 download(MODEL_URL,root/'pose_landmarker_full.task');download(URL,root/'rtmw3d_pose.onnx')
 subprocess.run([str(python),str(here/'setup_details.py'),'--runtime',str(root)],check=True)
 subprocess.run([str(python),str(here/'gpu_probe.py'),'--model',str(root/'rtmw3d_pose.onnx'),'--output',str(root/'gpu_ready.json')],check=True)
 (root/'rtmw3d_manifest.json').write_text(json.dumps({'url':URL,'sha256':hashlib.sha256((root/'rtmw3d_pose.onnx').read_bytes()).hexdigest(),'python':str(python)},indent=2),encoding='utf-8')
 print('GPU_RUNTIME_READY',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--runtime',required=True);p.add_argument('--env');p.add_argument('--skip-install',action='store_true');a=p.parse_args();install(a.runtime,a.env,a.skip_install)
