"""Isolated CUDA SixDRepNet360 candidates; confidence is not calibrated."""
import argparse,copy,hashlib,json,time
from pathlib import Path
import numpy as np
try:from . import pose_data,worker,head_math,orientation_math as om,fit_math
except ImportError:import pose_data,worker,head_math,orientation_math as om,fit_math

def load_model(config):
 import torch,torchvision
 cfg=json.loads(Path(config).read_text(encoding='utf8'));checkpoint=Path(cfg['checkpoint'])
 if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=cfg['sha256']:raise ValueError('头部权重校验失败')
 model=torchvision.models.resnet50(weights=None);model.fc=torch.nn.Linear(2048,6)
 state=torch.load(checkpoint,map_location='cpu',weights_only=True);state=state.get('model_state_dict',state)
 state={k.replace('linear_reg.','fc.'):v for k,v in state.items()};model.load_state_dict(state,strict=True)
 if any(not torch.isfinite(v).all() for v in state.values() if torch.is_floating_point(v)):raise ValueError('头部权重含非有限值')
 return model.eval(),cfg

def run(input,output,config):
 if Path(output).exists():raise ValueError('独立头部任务不能覆盖已有文件，请使用新的输出路径')
 import torch
 from PIL import Image
 from torchvision import transforms
 if Path(input).resolve()==Path(output).resolve():raise ValueError('头部输出不能覆盖输入')
 if not torch.cuda.is_available():raise ValueError('独立头部模型需要可用 CUDA；未冒充 GPU')
 model,cfg=load_model(config);model=model.cuda()
 doc=pose_data.read_pose(input);im=worker.prepare_image(doc['image']['path']);ref=head_math.reference(doc)
 if ref is None:raise ValueError('脸部观测不足，未运行独立头部模型')
 preprocess=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(mean=[.485,.456,.406],std=[.229,.224,.225])])
 crops=[];boxes=[]
 b=doc['wholebody'];xy=np.array([[p['x']*im.width,p['y']*im.height] for p in b[23:91] if p.get('confidence',0)>.5])
 low=xy.min(0);high=xy.max(0);center=(low+high)/2
 for padding in (1.5,1.9):
  size=max(high-low)*padding;box=[*(center-size/2),*(center+size/2)];roi=im.crop(tuple(box))
  boxes.append([float(x) for x in box]);crops += [preprocess(roi),preprocess(roi.rotate(15,resample=Image.Resampling.BICUBIC))]
 start=time.perf_counter()
 with torch.inference_mode():
  raw=model(torch.stack(crops).cuda());x=torch.nn.functional.normalize(raw[:,:3],dim=1);z=torch.nn.functional.normalize(torch.linalg.cross(x,raw[:,3:],dim=1),dim=1);y=torch.linalg.cross(z,x,dim=1)
  rotations=torch.stack((x,y,z),dim=2).cpu().numpy()
 cv_face=np.diag([1.,-1.,-1.]);frames=[om.C_DOWN@R@cv_face for R in rotations];roll=om.C_DOWN@fit_math.rotation([0,0,-np.deg2rad(15)])@om.C_DOWN.T
 candidates=[]
 for i,box in enumerate(boxes):
  frame=om.proper_rotation(frames[i*2]);equiv=head_math.angle(roll@frame,frames[i*2+1])
  candidates.append(dict(source='sixdrepnet360_cuda',frame=frame.tolist(),confidence=.65 if equiv<12 else .3,confidence_method='uncalibrated_candidate_with_image_roll_check',
   box=box,roll_equivariance_deg=equiv,device='cuda',model_sha256=cfg['sha256'],camera_axes='X_right_Y_away_Z_up',conversion='CV_right_down_away_to_head_right_up_front'))
 result=copy.deepcopy(doc);result.setdefault('orientation',{})['head_candidates']=doc.get('orientation',{}).get('head_candidates',[])+candidates
 result['head3d_runtime']=dict(model='6DRepNet360 full rotation',device='cuda',checkpoint_sha256=cfg['sha256'],seconds=time.perf_counter()-start,actual_inference=True)
 worker.write_json(output,result);print(json.dumps(dict(runtime=result['head3d_runtime'],candidates=candidates)),flush=True)
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for name in ('input','output','config'):p.add_argument('--'+name)
 p.add_argument('--verify-only',action='store_true');args=p.parse_args()
 if args.verify_only:load_model(args.config);print('HEAD_MODEL_STRICT_LOAD_PASS',flush=True)
 else:run(args.input,args.output,args.config)
