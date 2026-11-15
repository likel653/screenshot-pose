"""Cheap optical-flow prepass adds real observations where screen motion is fast."""
import math

def refine(base,signals,fps,max_fps=30.,threshold=.10):
    if not base:return []
    if not math.isfinite(max_fps) or max_fps<=0:raise ValueError('最高采样率无效')
    limit=min(fps,max_fps);spacing=max(1,int(math.floor(fps/limit)));selected=set(base)
    # Each one-second bucket has an explicit inference budget; base observations stay.
    counts={}
    for frame in base:counts[int(frame/fps)]=counts.get(int(frame/fps),0)+1
    candidates=sorted(((score,frame) for frame,score in signals.items() if score>=threshold and base[0]<=frame<=base[-1]),reverse=True)
    for score,frame in candidates:
        if counts.get(int(frame/fps),0)>=math.ceil(limit):continue
        if any(abs(frame-j)<spacing for j in range(max(base[0],frame-spacing+1),min(base[-1],frame+spacing-1)+1) if j in selected):continue
        selected.add(frame);counts[int(frame/fps)]=counts.get(int(frame/fps),0)+1
    return sorted(selected)

def analyze(path,base,fps,max_fps,progress=None):
    import cv2,numpy as np
    cap=cv2.VideoCapture(str(path));scores={};previous=None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES,base[0])
        for frame in range(base[0],base[-1]+1):
            ok,image=cap.read()
            if not ok:raise ValueError('自适应采样预扫描解码失败')
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);gray=cv2.resize(gray,(160,120),interpolation=cv2.INTER_AREA)
            if previous is not None:
                flow=cv2.calcOpticalFlowFarneback(previous,gray,None,.5,2,11,2,5,1.1,0)
                speed=np.linalg.norm(flow[12:108,16:144],axis=2)
                scores[frame]=float(np.quantile(speed,.8)*fps/200.)
            previous=gray
            if progress and (frame-base[0])%120==0:progress(message='运动采样预扫描 %d / %d'%(frame-base[0]+1,base[-1]-base[0]+1),done=0,total=len(base))
    finally:cap.release()
    refined=refine(base,scores,fps,max(max_fps,len(base)/max(1.,(base[-1]-base[0]+1)/fps)))
    return refined,dict(method='image_optical_flow_budgeted',base_samples=len(base),added_samples=len(refined)-len(base),
        maximum_fps=max_fps,normalized_screen_speed_threshold=.10,motion_is_not_pose_error=True)
