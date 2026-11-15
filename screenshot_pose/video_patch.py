"""Local correction offsets with unchanged interval boundaries."""
import math
from .video_stability import unit,slerp
from .video_temporal import multiply

IDENTITY=[1.,0.,0.,0.]
def delta(base,corrected):
    q=unit(base)
    return unit(multiply(unit(corrected),[q[0],-q[1],-q[2],-q[3]]))

def validate(start,end,anchors):
    if not isinstance(start,int) or not isinstance(end,int) or end-start<2:raise ValueError('区间至少跨三帧，首尾用于衔接原动作')
    if end-start>2000:raise ValueError('单次局部修正最多 2000 帧，请分段处理')
    if not anchors:raise ValueError('请先记录至少一个修正点')
    if any(not start<int(frame)<end for frame in anchors):raise ValueError('修正点必须位于区间内部；起止帧保留原姿势')

def offset(frame,name,start,end,anchors):
    if frame<=start or frame>=end:return IDENTITY[:],[0.,0.,0.]
    keys=[(float(start),dict(rotation=IDENTITY,location=[0.,0.,0.]))]
    keys.extend((float(k),v[name]) for k,v in sorted(anchors.items(),key=lambda p:float(p[0])))
    keys.append((float(end),dict(rotation=IDENTITY,location=[0.,0.,0.])))
    for (left,a),(right,b) in zip(keys,keys[1:]):
        if left<=frame<=right:
            t=(frame-left)/(right-left);t=t*t*(3-2*t)
            return slerp(a['rotation'],b['rotation'],t),[x+(y-x)*t for x,y in zip(a['location'],b['location'])]
    raise ValueError('修正区间无效')

def apply(frame,name,rotation,location,start,end,anchors):
    q,p=offset(frame,name,start,end,anchors)
    return unit(multiply(q,unit(rotation))),[a+b for a,b in zip(location,p)]
