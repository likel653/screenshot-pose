"""Relative image motion and conservative floor-contact hypotheses, not world mocap."""
import math,statistics,bisect

RESETS={'scene_cut','time_gap','missing_person','roi_jump'}

def observation(data):
    points=data['landmarks'];width=data['image']['width'];height=data['image']['height']
    def point(i):
        p=points[i];return [p['x']*width,p['y']*height,min(p.get('visibility',1.),p.get('presence',1.))]
    hips=[point(i) for i in (23,24)];feet={}
    for side,ids in (('L',(27,29,31)),('R',(28,30,32))):
        rows=[point(i) for i in ids];feet[side]=dict(point=[statistics.mean(r[k] for r in rows) for k in (0,1)],confidence=min(r[2] for r in rows))
    lengths=[]
    for ids in ((23,25,27),(24,26,28)):
        rows=[point(i) for i in ids]
        if min(r[2] for r in rows)>=.5:lengths.append(sum(math.dist(a[:2],b[:2]) for a,b in zip(rows,rows[1:])))
    return dict(hips=[statistics.mean(p[k] for p in hips) for k in (0,1)],confidence=min(p[2] for p in hips),
        feet=feet,leg_pixels=statistics.mean(lengths) if lengths else None,tracking=data.get('video_tracking',{}))

def quantile(values,fraction):
    rows=sorted(values);return rows[min(len(rows)-1,int((len(rows)-1)*fraction))]

def plan(times,observations,mode='HEIGHT',gain=1.,min_support=.18,cuts=()):
    """Per-shot scale and persistent low/slow foot runs. Uncertain frames stay free."""
    if mode not in ('IN_PLACE','HEIGHT','SCREEN'):raise ValueError('未知骨盆运动模式')
    cuts=set(cuts)|{i for i,o in enumerate(observations) if i and (times[i]-times[i-1]>.3 or o['tracking'].get('reset') in RESETS)}
    boundaries=[0]+sorted(i for i in cuts if 0<i<len(times))+[len(times)];roots=[[0.,0.,0.] for _ in times];segments=[];scales=[];last=[0.,0.,0.]
    for begin,end in zip(boundaries,boundaries[1:]):
        rows=observations[begin:end];sizes=[r['leg_pixels'] for r in rows if r.get('leg_pixels') and r['leg_pixels']>5]
        if not sizes:continue
        scale=statistics.median(sizes);scales.append(dict(start=begin,end=end-1,pixels_per_leg=scale))
        first=next((r for r in rows if r['confidence']>=.6),None);origin=first['hips'] if first else rows[0]['hips'];offset=list(last)
        for i in range(begin,end):
            o=observations[i]
            if mode!='IN_PLACE' and first and o['confidence']>=.6:
                d=[(o['hips'][0]-origin[0])/scale*gain if mode=='SCREEN' else 0.,0.,-(o['hips'][1]-origin[1])/scale*gain]
                # Camera-relative displacement has a finite, reported operating range.
                if max(abs(x) for x in d)<=2.:
                    candidate=[x+y for x,y in zip(offset,d)]
                    if i==begin or math.dist(candidate,last)<=max(.025,4*(times[i]-times[i-1])):last=candidate
            roots[i]=list(last)
        for side in ('L','R'):
            reliable=[r['feet'][side]['point'][1] for r in rows if r['feet'][side]['confidence']>=.6]
            both=[r['feet'][s]['point'][1] for r in rows for s in ('L','R') if r['feet'][s]['confidence']>=.6]
            if not reliable or not both:continue
            floor=quantile(both,.9);active=[]
            def commit():
                if len(active)>=3 and times[active[-1]]-times[active[0]]>=min_support:
                    segments.append(dict(side=side,start=times[active[0]],end=times[active[-1]],indices=list(active),
                        method='image_floor_and_low_velocity',estimated=True))
            for i in range(begin,end):
                foot=observations[i]['feet'][side];p=foot['point'];speed=0.
                for j in (i-1,i+1):
                    if begin<=j<end:
                        speed=max(speed,math.dist(p,observations[j]['feet'][side]['point'])/scale/abs(times[j]-times[i]))
                tracking=observations[i]['tracking']
                # Missing tracking and ambiguous left/right are insufficient support evidence.
                ambiguous=any('ankles_side_ambiguity' in e for e in tracking.get('events',[]))
                low=p[1]>=floor-scale*(.10 if active else .065)
                stable=speed<(.38 if active else .22)
                if tracking and not ambiguous and foot['confidence']>=.6 and low and stable:active.append(i)
                else:commit();active=[]
            commit()
    return dict(roots=roots,segments=segments,scales=scales,cuts=sorted(cuts),mode=mode,world_trajectory_recovered=False)

def root_at(times,roots,stamp,cuts=()):
    i=max(0,bisect.bisect_right(times,stamp+1e-8)-1)
    if i+1==len(times) or stamp<=times[0] or i+1 in cuts or times[i+1]-times[i]>.3:return list(roots[i])
    u=(stamp-times[i])/(times[i+1]-times[i]);return [a+(b-a)*u for a,b in zip(roots[i],roots[i+1])]

def review_issues(times,mapping,segments,solves,start_frame,origin,fps):
    events=[]
    for segment in segments:
        stamp=(segment['start']+segment['end'])/2
        events.append((stamp,segment['side'],'自动支撑候选 %.2f–%.2f 秒，请回放确认'%(segment['start'],segment['end'])))
    for solve in solves:
        if not solve['accepted']:events.append((solve['time'],solve['side'],'自动支撑目标不可达，已撤回该腿求解；残差 %.4f'%solve['residual']))
    return [dict(index=min(range(len(times)),key=lambda i:abs(times[i]-stamp)),frame=start_frame+round((stamp-origin)*fps),
        control=mapping['foot.'+side],reason=('左脚：' if side=='L' else '右脚：')+reason,repaired=False,estimated=True,automatic_support=True)
        for stamp,side,reason in events if mapping.get('foot.'+side)]
