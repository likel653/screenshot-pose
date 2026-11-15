"""Evidence-bounded wrist repairs in forearm space; confident 3-D gestures stay intact."""
import math
from . import video_stability as q

def evidence(data,side,applied,threshold=.5):
    hand=data.get('orientation',{}).get('hands',{}).get(side,{})
    whole=data.get('wholebody',[]);axes=None;size=None;image_confidence=0.
    if len(whole)==133:
        start=91 if side=='L' else 112;image=data['image'];ids=[0,5,9,17]
        points=[[whole[start+i]['x']*image['width'],whole[start+i]['y']*image['height']] for i in ids]
        confidence=[min(whole[start+i].get('confidence',whole[start+i].get('score',whole[start+i].get('visibility',1.))),whole[start+i].get('presence',1.)) for i in ids]
        across=[points[1][k]-points[3][k] for k in (0,1)];along=[sum(p[k] for p in points[1:])/3-points[0][k] for k in (0,1)]
        if min(math.hypot(*across),math.hypot(*along))>=2.:
            axes=[across,along];size=math.sqrt(math.hypot(*across)*math.hypot(*along));image_confidence=min(confidence)
    shape=None;world=hand.get('world',[])
    if len(world)==21:
        palm=[world[i] for i in (0,5,9,13,17)];scale=sum(math.dist(palm[0],p) for p in (palm[1],palm[2],palm[4]))/3
        if scale>1e-7:
            values=[math.dist(palm[i],palm[j])/scale for i in range(5) for j in range(i+1,5)]
            if min(values)>1e-4:shape=values
    return dict(source=hand.get('source','missing'),confidence=float(hand.get('confidence',0.)),applied=bool(applied),shape=shape,
        reliable_3d=bool(applied and hand.get('source')!='rtmw_palm_planar_pnp' and len(hand.get('world',[]))==21 and hand.get('confidence',0.)>=max(.55,threshold)),
        axes=axes,image_size=size,image_confidence=image_confidence,protected=bool(data.get('edited_indices')))

def observe(rig,mapping,settings,data,orientation_report):
    """Record actual evaluated forearm-relative wrist rotations after the solve."""
    from . import hand_presets,retarget
    rows={}
    for side in ('L','R'):
        hand=rig.pose.bones.get(mapping.get('hand.'+side,''));forearm=rig.pose.bones.get(mapping.get('forearm.'+side,''))
        if not hand or not forearm:continue
        row=evidence(data,side,side in orientation_report.get('hands',{}),settings.orientation_threshold)
        row.update(relative_rotation=list((forearm.matrix.to_quaternion().inverted()@hand.matrix.to_quaternion()).normalized()),
            hand=hand.name,forearm=forearm.name,
            dependency_state={b.name:retarget.snapshot_properties(rig,b) for b in (hand,forearm)})
        row['protected']=bool(row['protected'] or hand_presets.manual_fingers(data,side) or hand_presets.selected(settings,data,side)!='NONE')
        rows[side]=row
    return rows

def angle2(a,b):
    na=math.hypot(*a);nb=math.hypot(*b)
    return math.degrees(math.acos(max(-1.,min(1.,sum(x*y for x,y in zip(a,b))/max(na*nb,1e-9)))))

def image_consistency(rows,left,right,indices):
    samples=[rows[i] for i in (left,right,*indices)]
    if any(not r.get('axes') or r.get('image_confidence',0.)<.5 for r in samples):return None
    reference=samples[0];largest=0.
    for sample in samples[1:]:
        ratio=sample['image_size']/reference['image_size']
        if not .5<=ratio<=2.:return None
        for a,b in zip(reference['axes'],sample['axes']):largest=max(largest,angle2(a,b))
    return largest if largest<=30. else None

def shape_error(a,b):
    return math.sqrt(sum(math.log(x/y)**2 for x,y in zip(a,b))/len(a))

def shape_outlier(rows,index,left,right):
    a,b,c=[rows[i].get('shape') for i in (left,index,right)]
    return bool(a and b and c and shape_error(a,c)<.15 and min(shape_error(a,b),shape_error(c,b))>.35)

def plan(times,records,max_gap=.35,cuts=()):
    """Repair short ambiguous outliers only, with fixed anchors and compatible pixels."""
    cuts=set(cuts);fixes={};events=[];reviews=[]
    for side in ('L','R'):
        rows=[r.get('__wrists__',{}).get(side) for r in records];candidates=[]
        def eligible(index,left,right):
            row=rows[index]
            return bool(row and not row.get('protected') and (not row.get('reliable_3d') or shape_outlier(rows,index,left,right)))
        def anchor(row):return bool(row and not row.get('protected') and row.get('applied') and row.get('confidence',0)>=.55)
        for left in range(len(times)-2):
            if not anchor(rows[left]):continue
            for right in range(left+2,min(left+5,len(times))):
                if times[right]-times[left]>max_gap+1e-7:break
                if any(i in cuts or times[i]-times[i-1]>.3 for i in range(left+1,right+1)):break
                indices=list(range(left+1,right))
                if not anchor(rows[right]):continue
                if not all(eligible(i,left,right) for i in indices):continue
                ends=q.angle(rows[left]['relative_rotation'],rows[right]['relative_rotation'])
                if ends>60:continue
                pixels=image_consistency(rows,left,right,indices)
                if pixels is None:continue
                dependencies=[rows[i].get('dependency_state',{}) for i in (left,right,*indices)]
                if any(state!=dependencies[0] for state in dependencies[1:]):continue
                targets={i:q.slerp(rows[left]['relative_rotation'],rows[right]['relative_rotation'],(times[i]-times[left])/(times[right]-times[left])) for i in indices}
                differences={i:q.angle(rows[i]['relative_rotation'],targets[i]) for i in indices}
                if any(differences[i]<(70 if rows[i].get('applied') else 15) for i in indices):continue
                quality=int(rows[left].get('reliable_3d',False))+int(rows[right].get('reliable_3d',False))
                candidates.append(dict(left=left,right=right,indices=indices,targets=targets,differences=differences,quality=quality,pixels=pixels,ends=ends))
        modified=set();anchors=set()
        for candidate in sorted(candidates,key=lambda c:(-c['quality'],c['ends'],-max(c['differences'].values()),c['left'])):
            left,right=candidate['left'],candidate['right'];indices=candidate['indices']
            if modified.intersection(indices) or anchors.intersection(indices) or left in modified or right in modified:continue
            modified.update(indices);anchors.update((left,right))
            for i,target in candidate['targets'].items():
                fixes.setdefault(i,{})[side]=target
                events.append(dict(index=i,side=side,left=left,right=right,method='forearm_relative_short_outlier_with_image_agreement',
                    estimated=True,source=rows[i]['source'],image_axis_change_degrees=candidate['pixels'],anchor_difference_degrees=candidate['ends'],
                    correction_degrees=candidate['differences'][i],shape_outlier=shape_outlier(rows,i,left,right)))
        for i in range(1,len(times)):
            if not rows[i] or not rows[i-1] or i in cuts or times[i]-times[i-1]>.15:continue
            delta=q.angle(rows[i-1]['relative_rotation'],rows[i]['relative_rotation'])
            if delta>90:reviews.append(dict(index=i,side=side,degrees=delta,reason='局部掌心方向变化较大，请对照原图；快速真实转腕也可能触发'))
    return fixes,events,reviews

def carry(times,records,max_age=.35,cuts=(),reserved=None,blocked_decay=()):
    """A short forearm-relative memory decays to the current body prediction."""
    cuts=set(cuts);reserved=reserved or {};blocked_decay=set(blocked_decay);fixes={};events=[]
    for side in ('L','R'):
        previous=None
        side_events=[];rows=[record.get('__wrists__',{}).get(side) for record in records]
        for i,stamp in enumerate(times):
            if i in cuts or (i and stamp-times[i-1]>.3):previous=None
            row=rows[i]
            if not row or row.get('protected'):previous=None;continue
            if row.get('applied'):
                previous=(i,row) if row.get('reliable_3d') else None
                continue
            if previous is None:continue
            left,anchor=previous;age=stamp-times[left]
            if age>max_age+1e-8:previous=None;continue
            if anchor.get('dependency_state',{})!=row.get('dependency_state',{}):previous=None;continue
            if side in reserved.get(i,{}):continue
            hold=min(.12,max_age*.5);u=max(0.,min(1.,(age-hold)/max(1e-8,max_age-hold)));weight=1-u*u*(3-2*u)
            current=row['relative_rotation'];target=q.slerp(current,anchor['relative_rotation'],weight)
            method='forearm_relative_memory_decay';right=None
            for j in range(i+1,len(times)):
                if j in cuts or times[j]-times[j-1]>.3 or times[j]-times[left]>max_age+1e-8:break
                future=rows[j]
                if not future or future.get('protected') or future.get('dependency_state',{})!=anchor.get('dependency_state',{}):break
                if future.get('applied'):
                    if future.get('confidence',0)>=.55 and q.angle(anchor['relative_rotation'],future['relative_rotation'])<=120:
                        target=q.slerp(anchor['relative_rotation'],future['relative_rotation'],(stamp-times[left])/(times[j]-times[left]));right=j;method='forearm_relative_missing_bracket'
                    break
            if method=='forearm_relative_memory_decay' and (i,side) in blocked_decay:previous=None;continue
            if q.angle(current,target)<1.:continue
            fixes.setdefault(i,{})[side]=target
            side_events.append(dict(index=i,side=side,left=left,right=right,age_seconds=age,weight=weight,estimated=True,
                source='last_reliable_3d_wrist',method=method))
        groups=[];group=[]
        for index in sorted(i for i,values in fixes.items() if side in values):
            if group and index!=group[-1]+1:groups.append(group);group=[]
            group.append(index)
        if group:groups.append(group)
        for group in groups:
            start=max(0,group[0]-1);end=min(len(times)-1,group[-1]+1);old_steps=[];new_steps=[]
            for i in range(start+1,end+1):
                if i in cuts or not rows[i] or not rows[i-1]:continue
                a=reserved.get(i-1,{}).get(side,rows[i-1]['relative_rotation']);b=reserved.get(i,{}).get(side,rows[i]['relative_rotation'])
                old_steps.append(q.angle(a,b));new_steps.append(q.angle(fixes.get(i-1,{}).get(side,a),fixes.get(i,{}).get(side,b)))
            if max(new_steps,default=0)>max(old_steps,default=0)+1e-5 or sum(new_steps)>sum(old_steps)+1e-5:
                for i in group:fixes[i].pop(side)
                for event in side_events:
                    if event['index'] in group:event['quality_guard_rejected']=True
        events.extend(side_events)
    fixes={i:values for i,values in fixes.items() if values}
    return fixes,events

def locked_controls(records,finger_sides):
    """Manual recognition edits outrank every automatic temporal stage."""
    locked=set()
    for i,record in enumerate(records):
        for name in finger_sides:
            if record.get(name,{}).get('protected'):locked.add((i,name))
        for side,row in record.get('__wrists__',{}).items():
            if not row.get('protected'):continue
            if row.get('hand'):locked.add((i,row['hand']))
            locked.update((i,name) for name,s in finger_sides.items() if s==side)
    return locked

def resolve(rig,snapshot,raw,fixes,mapping,settings):
    """Convert relative targets through this actual rig, editing only wrist channels."""
    from mathutils import Quaternion,Vector
    from . import retarget,orientation
    before=retarget.snapshot_pose(rig);corrections={};accepted=[];rejected=[]
    try:
        retarget.restore_pose(rig,snapshot,update=False);rig.animation_data_create();rig.animation_data.action=None
        for index,targets in sorted(fixes.items()):
            record=raw(index)
            for bone in rig.pose.bones:
                if bone.name not in record:continue
                row=record[bone.name];bone.rotation_mode='QUATERNION';bone.rotation_quaternion=row['rotation'];bone.location=row['location'];bone.scale=row['scale']
                for key,value in row.get('properties',{}).items():bone[key]=value
            for key,value in record.get('__object_properties__',{}).items():rig[key]=value
            retarget._update(rig)
            for side,target in targets.items():
                hand=rig.pose.bones.get(mapping.get('hand.'+side,''));forearm=rig.pose.bones.get(mapping.get('forearm.'+side,''))
                if not orientation._safe(hand,rig) or not forearm:rejected.append(dict(index=index,side=side,reason='不可写的手腕映射'));continue
                old=hand.matrix_basis.to_quaternion();wrist=rig.matrix_world@hand.head
                desired=(forearm.matrix.to_quaternion()@Quaternion(target)).normalized()
                neutral=forearm.matrix.to_quaternion()@forearm.bone.matrix_local.to_quaternion().inverted()@hand.bone.matrix_local.to_quaternion()
                axis=hand.head-forearm.head
                if axis.length<1e-7:rejected.append(dict(index=index,side=side,reason='前臂长度退化'));continue
                clamped,limited=orientation._clamp_wrist(desired,neutral,axis.normalized(),settings.wrist_limit)
                if limited and q.angle(list(desired),list(clamped))>.1:rejected.append(dict(index=index,side=side,reason='补间超出原手腕弯曲限位'));continue
                if limited:desired=clamped
                for _ in range(3):orientation._absolute(rig,hand,desired)
                error=q.angle(list(hand.matrix.to_quaternion()),list(desired));movement=(rig.matrix_world@hand.head-wrist).length
                if error>.1 or movement>1e-5:
                    retarget._set_rotation(hand,old);retarget._update(rig);rejected.append(dict(index=index,side=side,reason='绑定未按预期跟随',error_degrees=error,wrist_movement=movement));continue
                corrections.setdefault(index,{})[hand.name]=list(hand.matrix_basis.to_quaternion())
                accepted.append(dict(index=index,side=side,control=hand.name,error_degrees=error,wrist_movement=movement))
    finally:retarget.restore_pose(rig,before)
    return corrections,accepted,rejected
