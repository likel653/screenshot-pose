"""Video timing, cache validation and rotation smoothing; no Blender imports."""
import json, math
from pathlib import Path

SCHEMA = 'screenshot_pose/video_v1'
MAX_SAMPLES = 10000

def reference_plan(data,mapped,generated_fps=None,start_frame=None):
    """Keep all source observations, including missing samples and shared output frames."""
    if not mapped or any(b['frame']<a['frame'] or b['time']<=a['time'] for a,b in zip(mapped,mapped[1:])):
        raise ValueError('动作帧或时间对照无效')
    if generated_fps is None:
        mt=sum(r['time'] for r in mapped)/len(mapped);mf=sum(r['frame'] for r in mapped)/len(mapped)
        variance=sum((r['time']-mt)**2 for r in mapped)
        generated_fps=sum((r['time']-mt)*(r['frame']-mf) for r in mapped)/variance if variance>1e-9 else 0.
    if not isinstance(generated_fps,(int,float)) or not math.isfinite(generated_fps) or generated_fps<=0:raise ValueError('无法确定原动画帧率')
    origin=data.get('start_seconds',data['frames'][0]['time'])
    if start_frame is None:start_frame=mapped[0]['frame']-round((mapped[0]['time']-origin)*generated_fps)
    exact={r['time']:r['frame'] for r in mapped};rows=[]
    for i,item in enumerate(data['frames']):
        relative=(Path(item['pose']).parent/'image.png').as_posix() if item['status']=='ok' else '%06d/image.png'%i
        rows.append(dict(time=item['time'],frame=exact.get(item['time'],start_frame+round((item['time']-origin)*generated_fps)),
            status=item['status'],pose=item.get('pose'),image_relative=relative))
    end=data.get('end_seconds',rows[-1]['time'])
    return dict(rows=rows,fps=generated_fps,origin_time=origin,start_frame=start_frame,
        end_frame=max(rows[-1]['frame'],start_frame+math.ceil((end-origin)*generated_fps-1e-8)-1))

def sample_plan(fps, count, start=0., end=0., sample_fps=12.):
    if not all(math.isfinite(v) for v in (fps,start,end,sample_fps)) or not 0 < fps <= 1000 or not 0 < count <= 100000000:
        raise ValueError('视频帧率或帧数无效')
    if start < 0 or end < 0 or sample_fps < 0:
        raise ValueError('时间和识别帧率不能为负数')
    duration = count / fps
    stop = min(end, duration) if end else duration
    if start >= stop: raise ValueError('开始时间必须早于结束时间和视频结尾')
    first = max(0, math.ceil(start * fps - 1e-6))
    last = min(int(count)-1, math.ceil(stop * fps - 1e-6)-1)
    rate = min(sample_fps, fps) if sample_fps else fps
    total = math.ceil((last-first+1) / fps * rate)
    if total > MAX_SAMPLES: raise ValueError('识别帧超过 10000，请缩短范围或降低识别帧率')
    return sorted(set(min(last, first + round(i*fps/rate)) for i in range(total)))

def child(root, name):
    if not isinstance(name,str) or not name or Path(name).is_absolute(): raise ValueError('视频缓存必须使用相对路径')
    path=(root/name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file(): raise ValueError('视频缓存丢失或路径越界：'+name)
    return path

def read(path):
    path=Path(path).resolve()
    if path.stat().st_size>16*1024*1024: raise ValueError('视频索引过大')
    data=json.loads(path.read_text(encoding='utf8'))
    if data.get('schema')!=SCHEMA: raise ValueError('不是本插件的视频缓存')
    items=data.get('frames');previous=-1.
    if not isinstance(items,list) or not 1 <= len(items) <= MAX_SAMPLES: raise ValueError('视频缓存帧数无效')
    for item in items:
        t=item.get('time');status=item.get('status')
        if isinstance(t,bool) or not isinstance(t,(int,float)) or not math.isfinite(t) or t < 0 or t<=previous:
            raise ValueError('视频时间戳必须递增')
        previous=t
        if status not in ('ok','missing'): raise ValueError('视频帧状态无效')
        if status=='ok': child(path.parent,item.get('pose'))
    if sum(i['status']=='ok' for i in items)<2: raise ValueError('至少需要两帧有效姿势才能生成动画')
    return data

def timed_frames(data, scene_fps, start_frame, max_gap=.3):
    valid=[f for f in data['frames'] if f['status']=='ok']
    if len(valid)<2: raise ValueError('有效姿势不足两帧')
    # Only missing samples count as gaps; a deliberately low sample rate is valid.
    pending=[];previous=None
    for item in data['frames']:
        if item['status']=='missing': pending.append(item);continue
        if previous is not None and pending and item['time']-previous['time'] > max_gap+1e-6:
            raise ValueError('存在超过 %.2f 秒的识别中断，请裁短视频或换用其他识别模式'%max_gap)
        previous=item;pending=[]
    origin=float(data.get('start_seconds',data['frames'][0]['time']))
    if not math.isfinite(origin) or origin<0 or origin>data['frames'][0]['time']+1e-6: raise ValueError('视频起始时间无效')
    result={}
    for item in valid:
        frame=int(start_frame)+round((item['time']-origin)*scene_fps)
        result[frame]=item
    if len(result)<2: raise ValueError('视频范围过短，无法落在两个不同动画帧上')
    return sorted(result.items())

def smooth_quaternion(center, neighbors):
    """Hemisphere-aligned normalized average; never average q with -q."""
    values=[0.]*4;weight=0.
    for q,w in neighbors:
        sign=1. if sum(a*b for a,b in zip(center,q))>=0 else -1.
        for j in range(4): values[j]+=q[j]*w*sign
        weight+=w
    norm=math.sqrt(sum(v*v for v in values))
    return [v/norm for v in values] if norm>1e-10 and weight else list(center)
