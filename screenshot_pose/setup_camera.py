"""Optional official camera weights, using the existing isolated CUDA runtime."""
import argparse,hashlib,json,subprocess
from pathlib import Path
try:from .setup_hand3d import download
except ImportError:from setup_hand3d import download


def prepare(root,runtime):
    root=Path(root).resolve();runtime=Path(runtime).resolve();folder=runtime/'camera';folder.mkdir(parents=True,exist_ok=True)
    hand=json.loads((runtime/'hand3d/config.json').read_text(encoding='utf8'));python=Path(hand['python'])
    if not python.is_file():raise ValueError('请先准备三维手部 GPU 环境，相机标定共用这个独立环境')
    repo=root/'vendor/AnyCalib'
    if not (repo/'.git').is_dir():subprocess.run(['git','clone','--depth','1','https://github.com/javrtg/AnyCalib.git',str(repo)],check=True)
    checkpoint=folder/'anycalib_edit.pt';source='https://github.com/javrtg/AnyCalib/releases/download/v1.0.0/anycalib_edit.pt'
    download(source,checkpoint)
    with checkpoint.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    config=dict(schema='screenshot_pose/camera_runtime_v1',python=str(python),repo=str(repo),checkpoint=str(checkpoint),sha256=sha,
      source=source,source_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip())
    (folder/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(config),flush=True)
    return config

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--runtime',required=True);prepare(**vars(p.parse_args()))
