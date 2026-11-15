"""AnyCalib in the isolated CUDA environment; never import Torch into Blender."""
import argparse,json,sys,time
from pathlib import Path


def run(image,config,output):
    import numpy as np
    from PIL import Image,ImageOps
    import torch
    try:from . import camera_data
    except ImportError:import camera_data
    cfg=json.loads(Path(config).read_text(encoding='utf8'));sys.path.insert(0,cfg['repo'])
    from anycalib import AnyCalib
    if not torch.cuda.is_available():raise RuntimeError('相机标定需要可用的 CUDA 环境')
    with Image.open(image) as source:rgb=np.asarray(ImageOps.exif_transpose(source).convert('RGB')).copy()
    h,w=rgb.shape[:2];start=time.perf_counter()
    model=AnyCalib(model_id=None).eval()
    model.load_state_dict(torch.load(cfg['checkpoint'],map_location='cpu',weights_only=True),strict=True)
    model=model.to('cuda')
    tensor=torch.from_numpy(rgb).to('cuda').permute(2,0,1).float()/255
    pred=model.predict(tensor,cam_id='pinhole')
    values=pred['intrinsics'].detach().cpu().numpy().reshape(-1).tolist()
    success=bool(pred.get('success',True))
    report=dict(schema='screenshot_pose/camera_v1',valid=success,intrinsics=values,image_size=[w,h],
      image_sha256=camera_data.pixel_digest(rgb),source='AnyCalib_edit',model=cfg['checkpoint'],model_sha256=cfg.get('sha256'),
      device='cuda',seconds=time.perf_counter()-start,metric='estimated_intrinsics_not_measured_camera',
      fov_x_deg=float(np.degrees(2*np.arctan(w/(2*values[0])))))
    camera_data.intrinsics(report,w,h)
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(report,ensure_ascii=False),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('image','config','output'):p.add_argument('--'+key,required=True)
    run(**vars(p.parse_args()))
