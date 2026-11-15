"""Conservative three-frame review; preserve raw predictions and real peaks."""
import math

def unit(q):
    norm=math.sqrt(sum(v*v for v in q))
    if len(q)!=4 or not math.isfinite(norm) or norm<1e-9:raise ValueError('视频旋转无效')
    return [v/norm for v in q]

def angle(a,b):
    a,b=unit(a),unit(b)
    return math.degrees(2*math.acos(min(1.,abs(sum(x*y for x,y in zip(a,b))))))

def slerp(a,b,t):
    a,b=unit(a),unit(b);dot=sum(x*y for x,y in zip(a,b))
    if dot<0:b=[-v for v in b];dot=-dot
    dot=min(1.,dot)
    if dot>.9995:return unit([x*(1-t)+y*t for x,y in zip(a,b)])
    theta=math.acos(dot);denom=math.sin(theta)
    return [(x*math.sin((1-t)*theta)+y*math.sin(t*theta))/denom for x,y in zip(a,b)]

def inspect(times,raw,names,threshold=80.):
    """Repair only isolated low-confidence spikes bracketed by reliable poses.

    High-confidence sharp movement, multi-frame changes, first/last samples,
    and intervals wider than .5 seconds are never automatically repaired.
    """
    corrections={};issues=[]
    for index in range(1,len(times)-1):
        span=times[index+1]-times[index-1]
        if span<=0 or span>.5:continue
        left,center,right=raw(index-1),raw(index),raw(index+1)
        t=(times[index]-times[index-1])/span
        for name in sorted(names):
            a,b,c=left[name],center[name],right[name]
            bridge=slerp(a['rotation'],c['rotation'],t)
            error=angle(b['rotation'],bridge);ends=angle(a['rotation'],c['rotation'])
            if error<threshold or ends>35.:continue
            low=float(b.get('confidence',0.));bracket=min(float(a.get('confidence',0.)),float(c.get('confidence',0.)))
            fix=low<.55 and bracket>=.55
            issues.append(dict(index=index,control=name,angle_deg=round(error,3),confidence=low,
                repaired=fix,reason='低置信度孤立跳变' if fix else '高置信度或相邻证据不足，仅标记'))
            if fix:corrections.setdefault(index,{})[name]=bridge
    # A flipped hypothesis can last two or three samples. Only replace an
    # entirely weak short run with two reliable, mutually consistent anchors.
    weak={};previous=None
    for index,stamp in enumerate(times):
        current=raw(index)
        for name in sorted(names):
            record=current[name]
            if record.get('confidence',0.)<.55:
                if name not in weak and previous is not None and previous[name].get('confidence',0.)>=.55:
                    weak[name]=dict(left=previous[name],time=times[index-1],samples=[])
                run=weak.get(name)
                if run:
                    if stamp-run['time']>.3:run['samples']=None
                    elif run['samples'] is not None:run['samples'].append((index,record))
                continue
            run=weak.pop(name,None)
            if not run or not run['samples'] or len(run['samples'])<2 or stamp-run['time']>.3:continue
            a=run['left']['rotation'];b=record['rotation']
            if angle(a,b)>35.:continue
            candidates=[]
            for j,missing in run['samples']:
                q=slerp(a,b,(times[j]-run['time'])/(stamp-run['time']))
                error=angle(missing['rotation'],q)
                if error<threshold:break
                candidates.append((j,q,error,float(missing.get('confidence',0.))))
            else:
                for j,q,error,confidence in candidates:
                    corrections.setdefault(j,{})[name]=q
                    issues.append(dict(index=j,control=name,angle_deg=round(error,3),confidence=confidence,repaired=True,reason='低置信度短段翻转，可靠前后帧补间'))
        previous=current
    # A short missing local-hand track otherwise resets fingers to baseline.
    # Fill only inside reliable brackets, never extrapolate an edge or a long gap.
    by_key={(i['index'],i['control']):i for i in issues}
    runs={};previous=None
    # Read each solved frame once per pass, rather than once per control.
    for index,stamp in enumerate(times):
        current=raw(index)
        for name in sorted(names):
            record=current[name]
            if record.get('confidence',0.)<=.15:
                if name not in runs and previous is not None:
                    runs[name]=dict(left=previous[name],time=times[index-1],samples=[])
                run=runs.get(name)
                if run:
                    if stamp-run['time']>.3:run['samples']=None
                    elif run['samples'] is not None:run['samples'].append((index,record))
                continue
            run=runs.pop(name,None)
            if not run or not run['samples']:continue
            left,right=run['left'],record;span=stamp-run['time']
            if span<=0 or span>.3 or min(left.get('confidence',0.),right.get('confidence',0.))<.55:continue
            if angle(left['rotation'],right['rotation'])>100.:continue
            for j,missing in run['samples']:
                if name in corrections.get(j,{}):continue
                q=slerp(left['rotation'],right['rotation'],(times[j]-run['time'])/span)
                error=angle(missing['rotation'],q)
                if error<15.:continue
                corrections.setdefault(j,{})[name]=q
                by_key[(j,name)]=dict(index=j,control=name,angle_deg=round(error,3),confidence=float(missing.get('confidence',0.)),
                    repaired=True,reason='短暂局部跟踪中断，可靠前后帧补间')
        previous=current
    return corrections,sorted(by_key.values(),key=lambda i:(i['index'],i['control']))

ROLE_POINTS={'hips':(23,24),'spine':(11,12,23,24),'chest':(11,12),'neck':(0,7,8),'head':(0,7,8)}
for side in ('L','R'):
    off=0 if side=='L' else 1
    for role,points in [('upper_arm',(11,13)),('forearm',(13,15)),('hand',(15,17,19)),('thigh',(23,25)),('shin',(25,27)),('foot',(27,29,31))]:
        ROLE_POINTS[role+'.'+side]=tuple(i+off for i in points)

def confidences(data,mapping,finger_controls,finger_report=None):
    result={};points=data['landmarks'];hands=data.get('orientation',{}).get('hands',{})
    for role,name in mapping.items():
        if role in ROLE_POINTS:
            result[name]=min(min(points[i].get('visibility',1.),points[i].get('presence',1.)) for i in ROLE_POINTS[role])
            if role.startswith('hand.') and data.get('source') in ('sam3d_mhr','rtmw3d_wholebody'):
                result[name]=min(result[name],float(hands.get(role[-1],{}).get('confidence',0.)))
    for side,names in finger_controls.items():
        conf=hands.get(side,{}).get('confidence',0.)
        for name in names:
            result[name]=float(conf) if finger_report is None or name in finger_report.get('touched',[]) else 0.
    return result
