"""Two-scale lower-leg crops through the existing RTMW model, no downloads."""
import argparse,json,sys
from pathlib import Path
import numpy as np
try:from . import foot_data,worker
except ImportError:
 import types
 package=types.ModuleType('sp_foot_worker');package.__path__=[str(Path(__file__).parent)];sys.modules['sp_foot_worker']=package
 foot_data=__import__('sp_foot_worker.foot_data',fromlist=['foot_data'])
 import worker

def refine_image(data,model,sides,device='cpu'):
 from rtmlib import RTMPose
 # RTMW's original image-space 133 point observation.
 xy=foot_data.image_points(data)
 if xy.shape!=(133,2):raise ValueError('需要原始 133 个二维关节点，请先用 SAM / RTMW 识别')
 im=worker.prepare_image(data['image']['path'])
 if im.size!=(data['image']['width'],data['image']['height']):raise ValueError('参考图尺寸与识别数据不一致')
 report={};engine=None
 for s in sides:
  wholebody=data['wholebody']
  if isinstance(wholebody,list) and any(isinstance(wholebody[i],dict) and wholebody[i].get('outside_frame') for i in foot_data.INDICES[s][2:]):
   report[s]=dict(adopted=False,reason='脚部不在画面内')
   continue
  if engine is None:
   engine=RTMPose(str(model),model_input_size=(288,384),backend='onnxruntime',device=device,to_openpose=False)
  obs=[];reasons=[]
  for margin in (.25,.65):
   box=foot_data.crop_box(xy,s,im.size,margin);crop=np.asarray(im.crop(box));bgr=np.ascontiguousarray(crop[:,:,::-1])
   points,scores=engine(bgr,bboxes=[[0,0,crop.shape[1],crop.shape[0]]]);pred=points[0]+np.array(box[:2])
   item,reason=foot_data.observation(data,s,pred,scores[0])
   if item:item['box']=box;obs.append(item)
   else:reasons.append(reason)
  data,info=foot_data.refine(data,s,obs);info['rejections']=reasons;report[s]=info
 return data,report


def run(source,model,output,side='BOTH',device='cpu'):
 if Path(output).resolve() in (Path(source).resolve(),Path(model).resolve()):raise ValueError('输出不能覆盖原姿势或模型')
 data=json.loads(Path(source).read_text(encoding='utf8'))
 data,report=refine_image(data,model,('L','R') if side=='BOTH' else (side,),device)
 data['foot_refinement']=report;worker.write_json(Path(output),data)
 print(json.dumps(dict(ok=True,adopted=sum(x['adopted'] for x in report.values()))),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ('source','model','output'):p.add_argument('--'+k,required=True)
 p.add_argument('--side',choices=['L','R','BOTH'],default='BOTH');p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
 args=p.parse_args();run(**vars(args))
