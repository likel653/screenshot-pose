"""One screenshot job: body, optional CUDA hands, then one final commit marker."""
import argparse,copy,json,time
from pathlib import Path
try:from . import worker,pose_data,hand3d
except ImportError:import worker,pose_data,hand3d
MODES=('AUTO','OFF','HAMER','HAMER_TEXTURE')
def detect(image,model,output,overlay,backend='SAM3D',hand_mode='AUTO',person_index=0,max_poses=4,confidence=.5,native_root_path=None,progress=None,body_detect=None,session_factory=None):
 if hand_mode not in MODES:raise ValueError('截图手部模式无效')
 if backend not in ('MEDIAPIPE','RTMW','RTMW3D','SAM3D'):raise ValueError('截图身体模型无效')
 output=Path(output).resolve();body_path=output.with_name(output.stem+'.body.json');report_path=output.with_suffix('.report.json');snapshot=output.with_name(output.stem+'.reference.png')
 managed=[output,body_path,report_path,snapshot,Path(overlay).resolve()]+([Path(progress).resolve()] if progress else [])
 if len(set(managed))!=len(managed) or set(managed)&{Path(image).resolve(),Path(model).resolve()}:raise ValueError('截图输出不能覆盖图片、模型或预览')
 if any(p.exists() for p in managed):raise ValueError('截图任务需使用新的输出路径')
 output.parent.mkdir(parents=True,exist_ok=True)
 def stage(message):
  if progress:worker.write_progress(progress,dict(message=message))
 stage('识别身体姿势')
 screenshot_checks = body_detect is None and backend == 'SAM3D'
 if body_detect is None:
  if backend=='SAM3D':
   try:from . import worker_sam,screenshot_recovery
   except ImportError:import worker_sam,screenshot_recovery
   body_detect=worker_sam.detect
  elif backend=='RTMW3D':
   try:from .worker_gpu import detect as body_detect
   except ImportError:from worker_gpu import detect as body_detect
  elif backend=='RTMW':
   try:from .worker_high import detect as body_detect
   except ImportError:from worker_high import detect as body_detect
  else:body_detect=worker.detect
 args=dict(person_index=person_index,max_poses=max_poses,confidence=confidence)
 if screenshot_checks:args['screenshot_checks']=True
 if backend=='SAM3D' and native_root_path:args['native_root_path']=native_root_path
 # All stages read one owned snapshot. A changing source file cannot mix a
 # body from one screenshot with hands from a later overwrite.
 worker.save_png(worker.prepare_image(image),snapshot)
 if screenshot_checks:
  review_dir=output.with_suffix('.roll_review')
  try:base=body_detect(snapshot,model,body_path,overlay,**args)
  except worker_sam.PersonNotFound:
   base=screenshot_recovery.recover(snapshot,model,review_dir,body_detect,worker_sam.PersonNotFound,args,None,stage)
   if base is None:raise
  else:
   if pose_data.feet_overlap(base):
    try:base=screenshot_recovery.recover(snapshot,model,review_dir,body_detect,worker_sam.PersonNotFound,args,base,stage) or base
    except (ValueError,OSError,RuntimeError) as exc:
     base.setdefault('backend_warnings',[]).append('倾斜图复核失败，保留原观测：'+str(exc)[:140])
 else:base=body_detect(snapshot,model,body_path,overlay,**args)
 base=pose_data.validate(base);base['image']['original_path']=str(Path(image).resolve());worker.write_json(body_path,base)
 review=base.get('foot_review_sides',())
 if review:
  stage('局部复核有分歧的脚部')
  try:
   try:from . import worker_feet
   except ImportError:import worker_feet
   base,foot_report=worker_feet.refine_image(base,Path(model).resolve().with_name('rtmw_pose.onnx'),review)
  except Exception as exc:
   foot_report={side:dict(adopted=False,reason=str(exc)[:180]) for side in review}
  base['foot_refinement']=foot_report
 result=copy.deepcopy(base)
 mode='HAMER' if hand_mode=='AUTO' else hand_mode;reports={};session=None;started=time.perf_counter();status='off'
 if hand_mode!='OFF':
  state=hand3d.readiness(Path(model).resolve().parent,mode)
  if not state['ready']:
   reports={s:dict(adopted=False,attempted=False,backend=mode,reason='GPU 手部模型未就绪：'+'、'.join(state['missing'])) for s in ('L','R')};status='unavailable'
  else:
   stage('精修双手 · '+mode+' CUDA')
   try:
    session=(session_factory or hand3d.Session)(Path(model).resolve().parent,screenshot_precision=True)
    session.original_image=base['image'].get('original_path');result,reports=hand3d.refine(base,mode,session)
    for info in reports.values():info['attempted']=True
    status='adopted' if any(v.get('adopted') for v in reports.values()) else 'no_adoptable_result'
   except Exception as exc:
    result=copy.deepcopy(base);reports={s:dict(adopted=False,attempted=True,backend=mode,reason=str(exc)) for s in ('L','R')};status='failed'
   finally:
    if session:session.close()
 for side,info in reports.items():info.update(reference=base['image']['path'],gesture='NONE',fallback=not info.get('adopted'))
 assert all(result.get(k)==base.get(k) for k in ('landmarks','world_landmarks','full_skeleton','body_candidates')),'Hand refinement changed body observations'
 wrists=screenshot_recovery.update_wrists(result,reports) if screenshot_checks else []
 if wrists or base.get('screenshot_roll_review'):
  im=worker.prepare_image(result['image']['path'])
  worker.save_png(worker.draw_overlay(im,result['landmarks'],.35,person_index),overlay)
 result['automatic_hands']=dict(requested=hand_mode,backend=mode if hand_mode!='OFF' else None,status=status,body_preserved=not wrists,hands=reports,seconds=time.perf_counter()-started)
 if reports:result['hand3d_refinement']=dict(backend=mode,hands=reports,body_preserved=not wrists)
 report=dict(backend=mode,hands=reports,automatic=True,status=status,body_preserved=not wrists)
 if base.get('screenshot_roll_review'):report['roll_review']=base['screenshot_roll_review']
 if review:report['feet']=base['foot_refinement']
 worker.write_json(report_path,report);worker.write_json(output,pose_data.validate(result));stage('姿势识别完成')
 return result
def main():
 p=argparse.ArgumentParser()
 for key in ('image','model','output','overlay'):p.add_argument('--'+key,required=True)
 p.add_argument('--backend',choices=['MEDIAPIPE','RTMW','RTMW3D','SAM3D'],default='SAM3D');p.add_argument('--hand-mode',choices=MODES,default='AUTO')
 p.add_argument('--person-index',type=int,default=0);p.add_argument('--max-poses',type=int,default=4);p.add_argument('--confidence',type=float,default=.5);p.add_argument('--native-root-path');p.add_argument('--progress')
 detect(**vars(p.parse_args()));print('SCREENSHOT_PIPELINE_COMPLETE',flush=True)
if __name__=='__main__':main()
