"""Offline hand continuity. Observations remain intact; estimates are explicit."""
import copy,math
import numpy as np
from . import orientation_math as om,video_stability as rotations

def compact(data):
    skeleton=data.get('full_skeleton',{});roles=skeleton.get('role_map',{})
    return dict(orientation=copy.deepcopy(data.get('orientation',{})),image=data.get('image',{}),
        wholebody=data.get('wholebody',[]),edited_indices=data.get('edited_indices',[]),
        video_tracking=data.get('video_tracking',{}),
        forearms={s:skeleton.get('bones',{}).get(roles.get('forearm.'+s,''),{}).get('pose_rotation') for s in ('L','R')})

def matrix_angle(a,b):
    return math.degrees(math.acos(float(np.clip((np.trace(a.T@b)-1)*.5,-1,1))))

def planar_error(data,side,hand,frame):
    """Re-evaluate an existing IPPE rotation against its same five image points."""
    whole=data.get('wholebody',[]);box=hand.get('box',[])
    if len(whole)!=133 or len(box)!=4:return None
    template=np.zeros((21,3));template[5]=[.035,.065,0];template[9]=[0,.075,0];template[13]=[-.018,.07,0];template[17]=[-.035,.055,0]
    ids=[0,5,9,13,17];start=91 if side=='L' else 112
    size=box[2]-box[0]
    if size<=1e-6:return None
    w,h=data['image']['width'],data['image']['height']
    pixels=(np.array([[whole[start+i]['x']*w,whole[start+i]['y']*h] for i in ids])-box[:2])/size*512
    uv=(pixels-256)/600
    rotation=om.C_DOWN.T@om.proper_rotation(frame)@om.palm_frame(template).T
    points=template[ids]@rotation.T;A=[];b=[]
    for p,(u,v) in zip(points,uv):
        A.extend(([1,0,-u],[0,1,-v]));b.extend((u*p[2]-p[0],v*p[2]-p[1]))
    translation=np.linalg.lstsq(np.asarray(A),np.asarray(b),rcond=None)[0];camera=points+translation
    if np.min(camera[:,2])<=0:return None
    projected=camera[:,:2]/camera[:,2,None]*600+256
    return float(np.sqrt(np.mean(np.sum((projected-pixels)**2,axis=1)))/512)

def plan_palms(documents,times,threshold=.5,cuts=()):
    """Viterbi over observed candidates only; all strong single-frame anchors stay fixed."""
    overrides={};events=[];cuts=set(cuts)
    for side in ('L','R'):
        runs=[];run=[]
        for i,(data,t) in enumerate(zip(documents,times)):
            hand=data.get('orientation',{}).get('hands',{}).get(side)
            if i in cuts or (i and t-times[i-1]>.3):
                if run:runs.append(run);run=[]
            if not hand or hand.get('confidence',0)<threshold:
                if run:runs.append(run);run=[]
                continue
            primary=om.proper_rotation(hand['frame']);options=[primary];errors=[0.]
            eligible=(hand.get('source')=='rtmw_palm_planar_pnp' and not data.get('edited_indices') and 'world' not in hand)
            if eligible:
                first=planar_error(data,side,hand,primary)
                if first is not None:
                    errors=[first]
                    for candidate in hand.get('alternatives',[]):
                        error=planar_error(data,side,hand,candidate)
                        if error is not None and error<=first+.005:
                            options.append(om.proper_rotation(candidate));errors.append(error)
                else:eligible=False
            body=data.get('forearms',{}).get(side)
            body=om.proper_rotation(body) if body is not None else np.eye(3)
            run.append(dict(index=i,options=options,relative=[body.T@q for q in options],errors=errors,eligible=eligible))
        if run:runs.append(run)
        for run in runs:
            costs=[0.]+[.05]* (len(run[0]['options'])-1);history=[]
            for left,right in zip(run,run[1:]):
                current=[];parents=[]
                for j,q in enumerate(right['relative']):
                    values=[costs[k]+min((matrix_angle(p,q)/45)**2,25) for k,p in enumerate(left['relative'])]
                    parent=min(range(len(values)),key=values.__getitem__)
                    current.append(values[parent]+(0 if j==0 else .05)+max(0,right['errors'][j]-right['errors'][0])*50);parents.append(parent)
                costs=current;history.append(parents)
            choice=min(range(len(costs)),key=costs.__getitem__);choices=[choice]
            for parents in reversed(history):choice=parents[choice];choices.append(choice)
            for item,choice in zip(run,reversed(choices)):
                if not item['eligible']:continue
                index=item['index'];hand=copy.deepcopy(documents[index]['orientation']['hands'][side])
                hand['frame']=item['options'][choice].tolist();hand.pop('alternatives',None)
                hand['video_candidate_selection']=dict(candidate=choice+1,method='same_image_fit_and_body_relative_path',estimated=False)
                overrides.setdefault(index,{})[side]=hand
                if choice:events.append(dict(index=index,side=side,candidate=choice+1,reprojection_error=item['errors'][choice],primary_error=item['errors'][0]))
    return overrides,events

def finger_gaps(times,raw,names,max_gap=.6,cuts=(),threshold=.5):
    """Interpolate short bracketed local finger rotations; never alter valid anchors."""
    corrections={};events=[];cuts=set(cuts);values=[{name:record[name] for name in names} for record in (raw(i) for i in range(len(times)))]
    for name in sorted(names):
        left=None;missing=[]
        for i,t in enumerate(times):
            if i in cuts or (i and t-times[i-1]>.3):left=None;missing=[]
            if values[i][name].get('protected',False):left=None;missing=[];continue
            observed=values[i][name].get('observed',values[i][name].get('confidence',0)>=threshold)
            valid=observed and values[i][name].get('confidence',0)>=threshold
            if not valid:
                if not observed:missing.append(i)
                else:left=None;missing=[]
                continue
            if left is not None and missing and t-times[left]<=max_gap+1e-7:
                a=values[left][name]['rotation'];b=values[i][name]['rotation']
                if rotations.angle(a,b)<=100:
                    for j in missing:
                        dependency=values[left][name].get('dependency_state',{})
                        if dependency!=values[i][name].get('dependency_state',{}) or dependency!=values[j][name].get('dependency_state',{}):continue
                        corrections.setdefault(j,{})[name]=rotations.slerp(a,b,(times[j]-times[left])/(t-times[left]))
                        events.append(dict(index=j,control=name,method='bracketed_finger_interpolation',left=left,right=i,estimated=True))
            left=i;missing=[]
    return corrections,events

def carry_fingers(times,raw,names,max_age=.35,cuts=(),reserved=None,threshold=.55):
    """Finite local memory; actual observations and edited poses always take precedence."""
    cuts=set(cuts);reserved=reserved or {};previous={};corrections={};events=[]
    for i,stamp in enumerate(times):
        if i in cuts or (i and stamp-times[i-1]>.3):previous.clear()
        row=raw(i)
        for name in names:
            current=row[name]
            if current.get('protected'):previous.pop(name,None);continue
            if current.get('observed',current.get('confidence',0)>=threshold):
                if current.get('confidence',0)>=threshold:previous[name]=(i,current)
                else:previous.pop(name,None)
                continue
            if name not in previous:continue
            left,anchor=previous[name];age=stamp-times[left]
            if age>max_age+1e-8 or anchor.get('dependency_state',{})!=current.get('dependency_state',{}):previous.pop(name,None);continue
            if name in reserved.get(i,{}):continue
            hold=min(.12,max_age*.5);u=max(0.,min(1.,(age-hold)/max(1e-8,max_age-hold)));weight=1-u*u*(3-2*u)
            target=rotations.slerp(current['rotation'],anchor['rotation'],weight)
            if rotations.angle(current['rotation'],target)<1.:continue
            corrections.setdefault(i,{})[name]=target
            events.append(dict(index=i,control=name,left=left,age_seconds=age,weight=weight,estimated=True,method='finite_finger_memory_decay'))
    return corrections,events

def estimate_issues(estimates,finger_sides,mapping):
    """One navigable issue per hand/sample; retain per-joint evidence separately."""
    groups={}
    for event in estimates:
        side=finger_sides.get(event['control'])
        groups.setdefault((event['index'],side),[]).append(event['control'])
    return [dict(index=index,control=mapping.get('hand.'+side,controls[0]) if side else controls[0],
        controls=sorted(set(controls)),confidence=0.,estimated=True,repaired=True,
        reason=('左手' if side=='L' else '右手' if side=='R' else '手指')+'短暂缺失：'+str(len(set(controls)))+' 节估计衔接')
        for (index,side),controls in groups.items()]
