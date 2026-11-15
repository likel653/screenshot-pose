"""Decode locally, recognize sampled frames, commit a reusable video cache."""
import argparse, json, os, sys, tempfile, time
from pathlib import Path
try:
    from . import worker, video_data,video_sampling
except ImportError:
    import worker, video_data,video_sampling

class Tracker:
    def __init__(self, model, confidence):
        os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'screenshot_pose_mpl'))
        import mediapipe as mp
        self.mp=mp
        opts=mp.tasks.vision.PoseLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_buffer=Path(model).read_bytes()),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,num_poses=2,min_pose_detection_confidence=confidence,
            min_pose_presence_confidence=confidence,min_tracking_confidence=confidence)
        self.engine=mp.tasks.vision.PoseLandmarker.create_from_options(opts)
    def detect(self,image,model,output,overlay,time,confidence,**kwargs):
        import numpy as np
        im=worker.prepare_image(image);mp=self.mp
        result=self.engine.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(np.asarray(im))),round(time*1000))
        data=worker.make_document(result,Path(image),Path(image),im.size,0)
        worker.save_png(worker.draw_overlay(im,data['landmarks'],confidence),overlay);worker.write_json(output,data)
        return data
    def close(self): self.engine.close()

def run(video, output, model, backend='SAM3D', start=0., end=0., sample_fps=12., confidence=.5, native_root_path=None, detect_fn=None,resident=True,progress_path=None,continuous=True,sampling='FIXED',max_sample_fps=30.,hand3d_backend='LOCAL'):
    import cv2
    source=Path(video).resolve();out=Path(output).resolve()
    if not source.is_file(): raise ValueError('视频文件不存在')
    if backend not in ('MEDIAPIPE','SAM3D'): raise ValueError('视频识别模式无效')
    if hand3d_backend not in ('LOCAL','WILOR','HAMER'):raise ValueError('视频手部模型无效')
    if hand3d_backend!='LOCAL':
        try:from . import hand3d
        except ImportError:import hand3d
        state=hand3d.readiness(Path(model).resolve().parent,hand3d_backend)
        if not state['ready']:raise ValueError(hand3d_backend+' 缺少：'+'、'.join(state['missing']))
    worker.validate_options(0,2,confidence)
    if not Path(model).is_file(): raise ValueError('本地模型未准备好')
    # Caller allocates a new directory; never overwrite a completed cache.
    if out.exists(): raise ValueError('输出目录已存在，请使用新目录')
    out.mkdir(parents=True)
    cap=cv2.VideoCapture(str(source));tracker=None;items=[];engines=None;hand_session=None;started=time.perf_counter()
    def progress(**values): worker.write_progress(Path(progress_path) if progress_path else out/'progress.json',values)
    try:
        if not cap.isOpened(): raise ValueError('无法解码视频，请使用 MP4 / MOV / AVI 等本地文件')
        fps=float(cap.get(cv2.CAP_PROP_FPS));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        plan=video_data.sample_plan(fps,count,start,end,sample_fps)
        sampling_report=dict(method='fixed',base_samples=len(plan),added_samples=0)
        if sampling not in ('FIXED','ADAPTIVE'):raise ValueError('未知视频采样模式')
        if sampling=='ADAPTIVE' and sample_fps>0 and max_sample_fps>sample_fps:
            plan,sampling_report=video_sampling.analyze(source,plan,fps,max_sample_fps,progress)
        progress(done=0,total=len(plan),message='加载识别模型')
        if detect_fn is None:
            if backend=='MEDIAPIPE': tracker=Tracker(model,confidence);detect_fn=tracker.detect
            else:
                try: from .worker_sam import detect
                except ImportError: from worker_sam import detect
                detect_fn=detect
                if resident:
                    try:from .inference_session import Engines
                    except ImportError:from inference_session import Engines
                    engines=Engines()
        continuity=None
        if continuous and backend=='SAM3D' and detect_fn.__module__.endswith('worker_sam'):
            try:from .video_tracking import Continuity
            except ImportError:from video_tracking import Continuity
            continuity=Continuity()
        cap.set(cv2.CAP_PROP_POS_FRAMES,plan[0]);current=plan[0]
        from PIL import Image
        for index,source_frame in enumerate(plan):
            while current<source_frame:
                if not cap.grab(): raise ValueError('视频解码提前终止')
                current+=1
            ok,bgr=cap.read();current+=1
            if not ok: raise ValueError('视频帧解码失败：'+str(source_frame))
            folder=out/('%06d'%index);folder.mkdir()
            image=folder/'image.png';pose=folder/'pose.json';overlay=folder/'overlay.png'
            rgb=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));rgb.thumbnail((1600,1600));worker.save_png(rgb,image)
            t=source_frame/fps;item=dict(source_frame=source_frame,time=t,status='ok',pose=pose.relative_to(out).as_posix())
            progress(done=index,total=len(plan),message='识别第 %d / %d 帧'%(index+1,len(plan)))
            try:
                extra={'time':t} if backend=='MEDIAPIPE' else {'native_root_path':native_root_path}
                if engines:extra['engines']=engines
                if continuity:continuity.begin(rgb,t);extra['continuity']=continuity
                frame_start=time.perf_counter()
                data=detect_fn(image,model,pose,overlay,confidence=confidence,**extra)
                if data.get('pose_count',1)>1: raise ValueError('视频 %.2f 秒（源第 %d 帧）检测到多个人物，请框选单人区域或裁短范围后重试，避免换人'%(t,source_frame+1))
                if hand3d_backend!='LOCAL':
                    if hand_session is None:hand_session=hand3d.Session(Path(model).resolve().parent)
                    data,hand_report=hand3d.refine(data,hand3d_backend,hand_session)
                    worker.write_json(pose,data)
                    item['hand3d']=dict(backend=hand3d_backend,hands=hand_report)
                    # Existing video hand/wrist arbitration consumes these raw
                    # observations during baking; no extra smoothing here.
                item['inference_seconds']=time.perf_counter()-frame_start
            except Exception as exc:
                if not any(s in str(exc) for s in ('没有识别到人物','未检测到人物','未找到所选人物','高精度模型未检测到人物')): raise
                if continuity:continuity.missing()
                item.update(status='missing',error=str(exc)[:180]);item.pop('pose',None)
            items.append(item)
        manifest=dict(schema=video_data.SCHEMA,source_video=str(source),source_fps=fps,source_frame_count=count,
            backend=backend,start_seconds=start,end_seconds=min(end,count/fps) if end else count/fps,
            sample_fps=sample_fps,sampling=sampling_report,frames=items,root_motion='estimated_during_bake',timing='constant_fps_frame_index',hand3d_backend=hand3d_backend,
            tracking=dict(enabled=bool(continuity),events=continuity.events if continuity else []),
            runtime=dict(total_seconds=time.perf_counter()-started,persistent=bool(engines),engines_created=engines.created if engines else {}))
        if hand3d_backend!='LOCAL':
            hand_rows=[v for row in items for v in row.get('hand3d',{}).get('hands',{}).values()]
            manifest['hand3d_summary']=dict(backend=hand3d_backend,attempted=len(hand_rows),
                adopted=sum(bool(v.get('adopted')) for v in hand_rows),
                rejected=sum(not v.get('adopted') for v in hand_rows),
                rejected_hands_keep_original_observations=True)
        worker.write_json(out/'sequence.json',manifest)
        try: video_data.read(out/'sequence.json')
        except Exception:
            (out/'sequence.json').unlink(missing_ok=True);raise
        progress(done=len(plan),total=len(plan),message='视频识别完成',missing=sum(f['status']=='missing' for f in items))
        return manifest
    except Exception as exc:
        progress(done=len(items),total=len(plan) if 'plan' in locals() else 0,message='失败：'+str(exc)[:180]);raise
    finally:
        cap.release()
        if tracker: tracker.close()
        if engines:engines.close()
        if hand_session:hand_session.close()

def main():
    p=argparse.ArgumentParser()
    for name in ('video','output','model'):p.add_argument('--'+name,required=True)
    p.add_argument('--backend',default='SAM3D');p.add_argument('--native-root-path')
    p.add_argument('--hand3d-backend',choices=('LOCAL','WILOR','HAMER'),default='LOCAL')
    p.add_argument('--no-continuity',dest='continuous',action='store_false')
    p.add_argument('--sampling',choices=('FIXED','ADAPTIVE'),default='FIXED');p.add_argument('--max-sample-fps',type=float,default=30.)
    p.add_argument('--one-shot',dest='resident',action='store_false',help='诊断用：每帧重新加载，与旧流程对比')
    for name,default in (('start',0.),('end',0.),('sample-fps',12.),('confidence',.5)):p.add_argument('--'+name,type=float,default=default)
    try:run(**vars(p.parse_args()));return 0
    except Exception as exc:print('视频识别失败：'+str(exc),file=sys.stderr,flush=True);return 1

if __name__=='__main__':raise SystemExit(main())
