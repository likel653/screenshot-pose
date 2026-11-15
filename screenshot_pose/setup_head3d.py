"""Official optional head weights, reusing the isolated hand CUDA environment."""
import argparse,hashlib,json,subprocess
from pathlib import Path
try:from .setup_hand3d import download
except ImportError:from setup_hand3d import download

def prepare(runtime):
 root=Path(runtime).resolve();folder=root/'head3d';folder.mkdir(parents=True,exist_ok=True)
 hand=json.loads((root/'hand3d/config.json').read_text(encoding='utf8'));python=Path(hand['python'])
 if not python.is_file():raise ValueError('请先准备独立三维手部 CUDA 环境；头向模型共用其 PyTorch')
 source='https://cloud.ovgu.de/s/TewGC9TDLGgKkmS/download/6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth'
 checkpoint=folder/'6DRepNet360_full.pth';download(source,checkpoint)
 sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
 if sha!='3ee08f1e04b8d452a6c4a40926a6f38051894ae6d0aaa6d191fe6d8bc6e4f9c6':raise ValueError('作者权重与已验证版本不同，未启用；请检查来源和版本')
 cfg=dict(schema='screenshot_pose/head3d_runtime_v1',python=str(python),checkpoint=str(checkpoint),sha256=sha,source=source,official_repo='https://github.com/thohemp/6DRepNet360')
 trial=folder/'config.pending.json';trial.write_text(json.dumps(cfg,indent=2),encoding='utf8')
 subprocess.run([str(python),str(Path(__file__).with_name('worker_head3d.py')),'--config',str(trial),'--verify-only'],check=True,timeout=90)
 trial.replace(folder/'config.json');print('HEAD_SETUP_READY',flush=True);return cfg

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--runtime',required=True);prepare(**vars(p.parse_args()))
