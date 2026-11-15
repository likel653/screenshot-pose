"""Select, record visible video pixels, then optionally run local pose inference."""
import argparse,sys
from pathlib import Path
try:from . import worker,screen_capture,screen_recording,worker_video
except ImportError:import worker,screen_capture,screen_recording,worker_video

def run(output,duration=5.,record_fps=15,identify=False,model=None,backend='SAM3D',sample_fps=12.,confidence=.5,native_root_path=None,select_fn=None,record_fn=None,detect_fn=None,continuous=True,sampling='FIXED',max_sample_fps=30.):
 out=Path(output).resolve()
 if out.exists():raise ValueError('录制输出目录必须是新目录')
 screen_recording.check_options((0,0,64,64),duration,record_fps)
 out.mkdir(parents=True)
 def status(message,**values):return worker.write_progress(out/'progress.json',dict(message=message,**values))
 status('请框选视频画面区域，Esc 可取消')
 box=(select_fn or screen_capture.select_region)()
 if box is None:
  worker.write_json(out/'result.json',dict(cancelled=True));status('已取消选区');return None
 screen_recording.check_options(box,duration,record_fps)
 status('选区已就绪，请点击录制窗口中的“开始录制”')
 def progress(stats):return status('正在录制 · 剩余 %.1f 秒'%(max(0,duration-stats['elapsed'])),**stats)
 try:result=(record_fn or screen_recording.capture_video)(box,out/'screen.avi',duration,record_fps,progress)
 except screen_recording.RecordingCancelled:result=None
 if result is None:
  worker.write_json(out/'result.json',dict(cancelled=True));status('已取消录制');return None
 worker.write_json(out/'record.json',result)
 final=dict(cancelled=False,video='screen.avi',record=result,sequence=None)
 # Commit the recording first. An inference error must not lose a good video.
 worker.write_json(out/'result.json',final)
 if identify:
  try:
   status('录制完成，开始识别视频')
   (detect_fn or worker_video.run)(out/'screen.avi',out/'poses',model,backend=backend,sample_fps=sample_fps,confidence=confidence,
      native_root_path=native_root_path,progress_path=out/'progress.json',continuous=continuous,sampling=sampling,max_sample_fps=max_sample_fps)
   final['sequence']='poses/sequence.json';worker.write_json(out/'result.json',final);status('屏幕视频录制和识别完成')
  except Exception as exc:
   final['inference_error']=str(exc)[:240];worker.write_json(out/'result.json',final)
   status('录制已保存，识别失败：'+str(exc)[:180])
 else:status('录制已保存，可继续识别或重新选择区域')
 return final

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--output',required=True);p.add_argument('--duration',type=float,default=5.);p.add_argument('--record-fps',type=int,default=15)
 p.add_argument('--no-continuity',dest='continuous',action='store_false')
 p.add_argument('--sampling',choices=('FIXED','ADAPTIVE'),default='FIXED');p.add_argument('--max-sample-fps',type=float,default=30.)
 p.add_argument('--identify',action='store_true');p.add_argument('--model');p.add_argument('--backend',default='SAM3D')
 p.add_argument('--sample-fps',type=float,default=12.);p.add_argument('--confidence',type=float,default=.5);p.add_argument('--native-root-path')
 try:run(**vars(p.parse_args()));return 0
 except Exception as exc:print('屏幕录制失败：'+str(exc),file=sys.stderr);return 1
if __name__=='__main__':raise SystemExit(main())
