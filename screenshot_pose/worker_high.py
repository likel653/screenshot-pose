"""RTMW 133-point local inference; MediaPipe supplies optional estimated depth."""
import argparse,copy,json,sys
from contextlib import nullcontext
from pathlib import Path
try:
    from . import worker,pose_data
except ImportError:
    import worker,pose_data
# COCO WholeBody -> legacy body interchange. Full 133 points are retained too.
MAP={0:0,1:1,2:1,3:1,4:2,5:2,6:2,7:3,8:4,9:77,10:71,
     11:5,12:6,13:7,14:8,15:9,16:10,17:111,18:132,19:99,20:120,21:95,22:116,
     23:11,24:12,25:13,26:14,27:15,28:16,29:19,30:22,31:17,32:20}

def convert(points,scores,size,base=None):
    import numpy as np
    points=np.asarray(points);scores=np.asarray(scores)
    if points.shape!=(133,2) or scores.shape!=(133,) or not np.isfinite(points).all() or not np.isfinite(scores).all():
        raise worker.WorkerError('RTMW 返回的 133 点数据无效')
    w,h=size
    data=copy.deepcopy(base) if base else pose_data.template_pose(width=w,height=h)
    for i,j in MAP.items():
        p=data['landmarks'][i];p.update(x=float(points[j,0]/w),y=float(points[j,1]/h),visibility=float(np.clip(scores[j],0,1)),presence=float(np.clip(scores[j],0,1)))
        if not base:data['world_landmarks'][i].update(x=p['x']*2,y=p['y']*2*h/w,z=0.)
    data.update(source='rtmw_wholebody',suggested_mode='HYBRID',depth_source='mediapipe_estimate' if base else 'flat_fallback',
                wholebody=[dict(x=float(p[0]/w),y=float(p[1]/h),confidence=float(np.clip(s,0,1))) for p,s in zip(points,scores)])
    return data

def detect(image,model,output,overlay,person_index=0,max_poses=4,confidence=.5,engines=None,person_box=None,continuity=None):
    import numpy as np
    from rtmlib import RTMPose
    try:from .person_detector import make_detector
    except ImportError:from person_detector import make_detector
    worker.validate_options(person_index,max_poses,confidence)
    root=Path(model).resolve().parent;output=Path(output).resolve();original=Path(image).resolve();overlay=Path(overlay).resolve()
    for name in ('rtmw_detector.onnx','rtmw_pose.onnx'):
        if not (root/name).is_file():raise worker.WorkerError('请先点击“准备高精度模型”下载 RTMW')
    processed=output.with_name(output.stem+'.input.png')
    worker._check_paths(original,Path(model).resolve(),output,overlay,processed)
    if any(p==root/n for p in (output,overlay,processed) for n in ('rtmw_detector.onnx','rtmw_pose.onnx')):raise worker.WorkerError('输出不能覆盖模型')
    im=worker.prepare_image(original);rgb=np.asarray(im);bgr=np.ascontiguousarray(rgb[:,:,::-1])
    create=lambda:make_detector(str(root/'rtmw_detector.onnx'),score_thr=confidence,backend='onnxruntime',device='cpu')
    detector=engines.get(('yolox_cpu',str(root),confidence),create) if engines else create()
    boxes=detector(bgr) if person_box is None else [person_box]
    if len(boxes)==0:raise worker.WorkerError('高精度模型未检测到人物，请裁剪图片或降低检测阈值')
    # Stable largest-person-first order; never silently infer over an empty detection.
    boxes=sorted(boxes,key=lambda b:float((b[2]-b[0])*(b[3]-b[1])),reverse=True)[:max_poses]
    if person_index>=len(boxes):raise worker.WorkerError(f'只检测到 {len(boxes)} 人')
    create=lambda:RTMPose(str(root/'rtmw_pose.onnx'),model_input_size=(288,384),backend='onnxruntime',device='cpu',to_openpose=False)
    pose=engines.get(('rtmw_cpu',str(root)),create) if engines else create()
    points,scores=pose(bgr,bboxes=[boxes[person_index]])
    base=None;notes=[]
    try:
        import mediapipe as mp
        opts=mp.tasks.vision.PoseLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_buffer=Path(model).read_bytes()),num_poses=max_poses,min_pose_detection_confidence=confidence,min_pose_presence_confidence=confidence)
        create=lambda:mp.tasks.vision.PoseLandmarker.create_from_options(opts)
        manager=nullcontext(engines.get(('pose_image_cpu',str(root),confidence,max_poses),create)) if engines else create()
        with manager as engine:
            result=engine.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(rgb)))
        candidates=[worker.make_document(result,processed,original,im.size,i) for i in range(len(result.pose_landmarks))]
        def distance(d):
            return sum((d['landmarks'][i]['x']-points[0,MAP[i],0]/im.width)**2+(d['landmarks'][i]['y']-points[0,MAP[i],1]/im.height)**2 for i in (11,12,23,24))/4
        if candidates:
            candidate=min(candidates,key=distance)
            if distance(candidate)<.025:base=candidate
    except (ValueError,RuntimeError,OSError,ImportError) as exc:
        notes.append('深度估计不可用：'+str(exc)[:120])
    data=convert(points[0],scores[0],im.size,base)
    if base is None:notes.append('未匹配到可靠三维深度，使用平面初始姿势；请校正前后深度')
    data.update(image=dict(path=str(processed),original_path=str(original),width=im.width,height=im.height),person_index=person_index,pose_count=len(boxes),backend_warnings=notes)
    worker.save_png(im,processed);worker.save_png(worker.draw_overlay(im,data['landmarks'],.35,person_index),overlay);worker.write_json(output,data)
    return data
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('image','model','output','overlay'):p.add_argument('--'+name,required=True)
    p.add_argument('--person-index',type=int,default=0);p.add_argument('--max-poses',type=int,default=4);p.add_argument('--confidence',type=float,default=.5)
    try:
        a=p.parse_args();d=detect(**vars(a));print(json.dumps({'ok':True,'backend':d['source'],'depth':d['depth_source']}))
    except Exception as exc:print('识别失败：'+str(exc),file=sys.stderr);sys.exit(1)
