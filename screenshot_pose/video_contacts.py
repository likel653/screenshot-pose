"""Explicit interval locks baked to an isolated action; no permanent constraints."""
import json
import bpy
from mathutils import Vector,Quaternion,Matrix
from . import video_review as review,contact_targets as ct,contact_solver as cs,contact_math,retarget,orientation

def weight(frame,start,end,fade):
    if start<=frame<=end:return 1.
    distance=start-frame if frame<start else frame-end
    t=max(0.,1-distance/fade);return t*t*(3-2*t)

class ContactSession(review.EditSession):
    contact_session=True
    def __init__(self,context):
        s=context.scene.screenshot_pose
        start,end,fade=s.video_lock_start,s.video_lock_end,s.video_lock_fade
        if end<start or end-start+2*fade>2000:raise ValueError('接触区间无效或含过渡超过 2000 帧')
        if not start<=context.scene.frame_current<=end:raise ValueError('先定位到接触区间内的参考帧，再生成预览')
        rig,action=review.action_for(s);mapping=review.addon().pose_tools.mapping_for(s)
        endpoints=['FOOT_L','FOOT_R'] if s.video_lock_part=='BOTH_FEET' else [s.video_lock_part]
        names=set();jobs=[]
        scale=rig.matrix_world.to_scale()
        if min(scale)<=0 or max(scale)/min(scale)>1.001:raise ValueError('接触锁定需要角色骨架正向均匀缩放')
        retarget._update(rig)
        for endpoint in endpoints:
            kind,side=endpoint.split('_');bones=ct.chain(rig,mapping,kind,side)
            # Generated video uses FK; do not change keyed rig switches implicitly.
            for bone,key,value in retarget._fk_switches(rig,{r:n for r,n in mapping.items() if n in {b.name for b in bones}}):
                if abs(float(bone[key])-value)>1e-5:raise ValueError('接触链当前不是 FK，请使用插件生成的视频动作')
            world=rig.matrix_world@bones[-1].matrix
            jobs.append(dict(kind=kind,side=side,bones=bones,target=world.translation.copy(),world_rotation=world.to_quaternion()))
            names.update(b.name for b in bones)
        root=cs.root_control(rig,mapping,s.root_bone) if s.video_lock_pelvis and any(j['kind']=='FOOT' for j in jobs) else None
        if root:names.add(root.name)
        # Validate all channels before creating a draft, including the optional root.
        for curve in review.addon().animation_preview.curves(action):
            if any(curve.data_path.startswith(rig.pose.bones[n].path_from_id()+'.') for n in names):
                if curve.modifiers or curve.data_path.endswith(('rotation_euler','rotation_axis_angle')) or any(k.interpolation not in ('LINEAR','CONSTANT') for k in curve.keyframe_points):raise ValueError('接触区间仅支持烘焙后的线性四元数动作')
        previous=s.video_edit_scope;s.video_edit_scope='LEG_L' if endpoints[0]=='FOOT_L' else 'LEG_R' if endpoints[0]=='FOOT_R' else 'ARM_'+endpoints[0][-1]
        try:super().__init__(context)
        finally:s.video_edit_scope=previous
        self.jobs=jobs;self.root=root;self.names=sorted(names);self.scope=s.video_lock_part
        self.prefix={n:rig.pose.bones[n].path_from_id() for n in self.names}
        self.defaults={n:dict(rotation=list(rig.pose.bones[n].rotation_quaternion),location=list(rig.pose.bones[n].location)) for n in self.names}
        self.lock_start=start;self.lock_end=end;self.fade=fade;self.start=start-fade;self.end=end+fade;self.limit=s.video_lock_pelvis_limit
        self.reference_frame=context.scene.frame_current;self.state='READY';self.errors=[];self.preview_frame=self.reference_frame
        self.reference_world=[list(row) for row in rig.matrix_world];self.expected_frame=self.reference_frame
    def cancel(self):
        # Solving temporarily detaches the action. Errors in that window still
        # belong to this transaction and must restore its snapshot.
        current=self.rig.animation_data.action if self.rig.animation_data else None
        if not self.closed and self.state=='BUILDING' and current in (None,self.original):
            retarget._assign_action(self.rig,self.draft,self.before.get('action_slot_handle'))
        super().cancel()
    def start_preview(self):
        self.validate();self.preview=self.original.copy();self.preview.name=self.original.name+' · 接触预览';self.preview.use_fake_user=False
        retarget._assign_action(self.rig,self.preview,self.before.get('action_slot_handle'));self.state='BUILDING';self.generator=self.build()
    def tick(self):
        self.validate()
        if self.scene.frame_current!=self.expected_frame:raise ValueError('生成接触时请勿更改时间轴；已撤回')
        return super().tick()
    def build(self):
        rig=self.rig;values={};a=review.addon()
        for frame in range(self.start,self.end+1):
            self.updating=True
            retarget._assign_action(rig,self.original,self.before.get('action_slot_handle'));self.scene.frame_set(frame)
            for n in self.names:rig.pose.bones[n].rotation_mode='QUATERNION'
            retarget._update(rig)
            original={n:(rig.pose.bones[n].rotation_quaternion.copy(),rig.pose.bones[n].location.copy()) for n in self.names}
            rig.animation_data.action=None
            amount=weight(frame,self.lock_start,self.lock_end,self.fade)
            poles={j['side']:rig.matrix_world@j['bones'][1].head for j in self.jobs}
            if amount and self.root:
                legs=[]
                for j in self.jobs:
                    if j['kind']=='FOOT':
                        h,k,e=[rig.matrix_world@b.head for b in j['bones']];legs.append((h,j['target'],(k-h).length,(e-k).length))
                shift,reachable=contact_math.pelvis_shift(legs);shift=max(-self.limit,min(self.limit,shift))
                cs._translate_world(rig,self.root,Vector((0,0,shift)))
                for side in poles:poles[side]+=Vector((0,0,shift))
            for j in self.jobs:
                if amount:
                    q=rig.matrix_world.to_quaternion().inverted()@j['world_rotation']
                    error=ct.solve_chain(rig,j['bones'],j['target'],poles[j['side']],q)
                    if amount==1:self.errors.append(dict(frame=frame,endpoint=j['kind']+'_'+j['side'],error=error))
            values[frame]={}
            for n in self.names:
                bone=rig.pose.bones[n];q,p=original[n]
                corrected=bone.matrix_basis.to_quaternion();corrected.make_compatible(q)
                # Exact original boundary components preserve neighboring interpolation.
                values[frame][n]=dict(rotation_quaternion=list(q.slerp(corrected,amount)) if amount else list(q),location=list(p.lerp(bone.location,amount)))
            retarget._assign_action(rig,self.preview,self.before.get('action_slot_handle'));self.expected_frame=frame;self.updating=False
            self.s.video_review_status='求解接触 %d / %d 帧'%(frame-self.start+1,self.end-self.start+1)
            yield
        # Restore a stable preview state before curve mutation.
        for n in self.names:
            bone=rig.pose.bones[n];bone.rotation_mode='QUATERNION'
            for channel in ('rotation_quaternion','location'):bone.keyframe_insert(data_path=channel,frame=self.start,group=n)
        for curve in a.animation_preview.curves(self.preview):
            match=next(((n,ch) for n in self.names for ch in ('rotation_quaternion','location') if curve.data_path==self.prefix[n]+'.'+ch),None)
            if not match:continue
            n,ch=match;source=self.curves.get((curve.data_path,curve.array_index));outgoing='LINEAR'
            if source:
                prior=[p for p in source.keyframe_points if p.co.x<=self.end]
                if prior:outgoing=prior[-1].interpolation
            for point in list(curve.keyframe_points):
                if self.start<=point.co.x<=self.end:curve.keyframe_points.remove(point,fast=True)
            for frame,record in values.items():
                value=record[n][ch][curve.array_index]
                if frame in (self.start,self.end) and source:value=source.evaluate(frame)
                point=curve.keyframe_points.insert(frame,value,options={'FAST'});point.interpolation=outgoing if frame==self.end else 'LINEAR'
            curve.update()
        self.preview['sp_video_contacts']=json.dumps(dict(start=self.lock_start,end=self.lock_end,fade=self.fade,reference_frame=self.reference_frame,scope=self.scope,errors=self.errors))
        self.preview_hash=review.fingerprint(self.preview);self.state='PREVIEW';self.updating=True;self.scene.frame_set(self.preview_frame);self.updating=False
        worst=max((r['error'] for r in self.errors),default=0.)
        self.s.video_review_status='接触预览完成；最大末端残差 %.4f 场景单位。请播放检查，再确认或撤回'%worst
    def finish(self):
        result=super().finish();result.name=self.original.name+' · 接触修正';return result
