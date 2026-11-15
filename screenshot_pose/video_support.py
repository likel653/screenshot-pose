"""Bake relative pelvis translation and inferred stance to FK without constraints."""
import math
from mathutils import Vector
from . import contact_solver as cs,contact_targets as ct,contact_math,retarget,video_motion,video_temporal,video_resample

class Solver:
    def __init__(self,rig,mapping,settings,times,observations,cuts=(),calibration=None):
        self.rig=rig;self.s=settings;self.times=times;self.mapping=mapping;self.root=None;self.chains={};self.anchors={};self.warnings=[];self.records=[]
        self.mode=getattr(settings,'video_root_motion','HEIGHT');self.enabled=getattr(settings,'video_auto_support',True)
        self.plan=video_motion.plan(times,observations,self.mode,getattr(settings,'video_root_gain',1.),cuts=cuts)
        root_records=[{'root':dict(rotation=[1.,0.,0.,0.],location=p,confidence=o['confidence'])} for p,o in zip(self.plan['roots'],observations)]
        root_raw=lambda i:root_records[i]
        window=getattr(settings,'video_smooth',.12)
        protected={(i,'root') for i in range(len(times)) if any(abs(times[i]-times[c])<=window for c in self.plan['cuts'])}
        smooth=video_temporal.Smoother(times,root_raw,lambda i,n:[1.,0.,0.,0.],window,protected=protected)
        self.root_resampler=video_resample.Resampler(times,root_raw,smooth.sample,'HERMITE',self.plan['cuts'])
        self.calibration=calibration or {};self.touched=set()
        lengths=[]
        for side in ('L','R'):
            bones=[rig.data.bones.get(mapping.get(r+'.'+side,'')) for r in ('thigh','shin','foot')]
            if all(bones):lengths.append(sum((rig.matrix_world.to_3x3()@(b.head_local-a.head_local)).length for a,b in zip(bones,bones[1:])))
        self.scale=sum(lengths)/len(lengths) if lengths else 0.
        if self.mode!='IN_PLACE' or self.enabled:
            try:
                if self.scale<=1e-6:raise ValueError('缺少可靠的目标腿长尺度，请检查大腿、小腿与脚映射')
                scale=rig.matrix_world.to_scale()
                if min(scale)<=0 or max(scale)/min(scale)>1.001:raise ValueError('需要正向均匀骨架对象缩放')
                self.root=cs.root_control(rig,mapping,settings.root_bone);self.touched.add(self.root.name)
            except ValueError as exc:self.warnings.append('视频骨盆与支撑未启用：'+str(exc));self.enabled=False;self.mode='IN_PLACE'
        if self.enabled:
            for side in ('L','R'):
                try:
                    self.chains[side]=ct.chain(rig,mapping,'FOOT',side)
                    if side not in self.calibration:raise ValueError('缺少鞋底校准')
                except ValueError as exc:self.warnings.append(side+'脚自动支撑未启用：'+str(exc));self.chains.pop(side,None)
            self.plan['segments']=[r for r in self.plan['segments'] if r['side'] in self.chains]
            for bones in self.chains.values():self.touched.update(b.name for b in bones)
        else:self.plan['segments']=[]
    def apply(self,stamp):
        rig=self.rig;s=self.s;retarget._update(rig)
        requested=Vector((0,0,0));before=rig.matrix_world@self.root.head if self.root else None
        if self.root and self.mode!='IN_PLACE':
            requested=Vector(self.root_resampler.at(stamp,'root')[1])*self.scale
            if s.mirror:requested.x=-requested.x
            c=math.cos(s.yaw);si=math.sin(s.yaw);requested.x,requested.y=c*requested.x-si*requested.y,si*requested.x+c*requested.y
            cs._translate_world(rig,self.root,requested)
            actual=(rig.matrix_world@self.root.head)-before
            if (actual-requested).length>max(1e-5,self.scale*.001):raise ValueError('骨盆控制骨未按预期移动，请检查映射、位置约束或驱动')
        jobs=[]
        for index,segment in enumerate(self.plan['segments']):
            if not segment['start']-1e-8<=stamp<=segment['end']+1e-8:continue
            side=segment['side'];bones=self.chains[side];end=bones[-1]
            if index not in self.anchors:
                world=rig.matrix_world@end.matrix;target=world.translation.copy()
                sole=min((world@Vector(p)).z for p in self.calibration[side]['anchors'])
                target.z+=s.floor_height-sole
                self.anchors[index]=(target,world.to_quaternion())
            target,rotation=self.anchors[index]
            fade=.07;amount=min(1.,max(0.,(stamp-segment['start']+.02)/fade),max(0.,(segment['end']-stamp+.02)/fade));amount=amount*amount*(3-2*amount)
            jobs.append(dict(side=side,bones=bones,target=target,rotation=rotation,amount=amount,pole=rig.matrix_world@bones[1].head))
        correction=Vector((0,0,0))
        root_prior=self.root.location.copy() if self.root else None
        if jobs and self.root:
            for j in jobs:correction+=(j['target']-(rig.matrix_world@j['bones'][-1].head))*j['amount']
            correction/=1.5+sum(j['amount'] for j in jobs)
            # Reachability is a separate constraint from the weighted image-motion prior.
            legs=[]
            for j in jobs:
                h,k,e=[rig.matrix_world@b.head for b in j['bones']]
                target=e.lerp(j['target'],j['amount'])
                legs.append((h+correction,target,(k-h).length,(e-k).length))
            shift,reachable=contact_math.pelvis_shift(legs)
            if reachable:correction.z+=shift
            limit=self.scale*getattr(s,'video_support_limit',.25)
            if correction.length>limit:correction*=limit/correction.length
            cs._translate_world(rig,self.root,correction)
        accepted_count=0
        for j in jobs:
            bones=j['bones'];old={b.name:b.matrix_basis.to_quaternion() for b in bones}
            target=(rig.matrix_world@bones[-1].head).lerp(j['target'],j['amount'])
            rotation=(rig.matrix_world.to_quaternion().inverted()@j['rotation'])
            current=bones[-1].matrix.to_quaternion();rotation=current.slerp(rotation,j['amount'])
            error=ct.solve_chain(rig,bones,target,j['pole']+correction,rotation)
            accepted=error<=max(.003,self.scale*.015)
            if not accepted:
                for b in bones:retarget._set_rotation(b,old[b.name])
                retarget._update(rig)
            else:accepted_count+=1
            self.records.append(dict(time=stamp,side=j['side'],weight=j['amount'],residual=error,accepted=accepted,target=list(target)))
        if jobs and not accepted_count and self.root:
            self.root.location=root_prior;correction=Vector((0,0,0));retarget._update(rig)
        return dict(root_requested=list(requested),root_support_correction=list(correction),root_world=list(rig.matrix_world@self.root.head) if self.root else None)
    def report(self):
        return dict(mode=self.mode,enabled=bool(self.root and self.mode!='IN_PLACE'),root=self.root.name if self.root else '',
            units_per_leg=self.scale,world_trajectory_recovered=False,source='relative_image_pelvis',
            support_enabled=self.enabled,segments=self.plan['segments'],solves=self.records,warnings=self.warnings,
            accepted=sum(r['accepted'] for r in self.records),rejected=sum(not r['accepted'] for r in self.records))
