"""Video issue navigation and reversible, scoped animation corrections."""
import bisect,hashlib,importlib,json,math,uuid
from pathlib import Path
import bpy,bpy.utils.previews
from bpy.types import Operator,PropertyGroup,UIList
from bpy.props import StringProperty,IntProperty,BoolProperty,EnumProperty
from . import video_data,video_patch

_EDIT=None
_REVIEW=None
_ICONS=None
_ICON_PATH=''
def addon():return importlib.import_module(__package__)
def active():return _EDIT is not None and not _EDIT.closed
def guard():
    if active():raise ValueError('请先确认或撤回视频局部修正')

class SPVideoIssue(PropertyGroup):
    frame:IntProperty()
    control:StringProperty()
    reason:StringProperty()
    repaired:BoolProperty()
    adjusted:BoolProperty()
    scope_hint:StringProperty()

class SP_UL_video_issues(UIList):
    def draw_item(self,context,layout,data,item,icon,active_data,active_propname,index):
        row=layout.row(align=True)
        row.label(text=str(item.frame),icon='GREASEPENCIL' if item.adjusted else 'CHECKMARK' if item.repaired else 'ERROR')
        row.label(text=item.control)

def action_for(s):
    rig=s.target
    if not rig or rig.type!='ARMATURE':raise ValueError('请先选择角色骨架')
    action=rig.animation_data.action if rig.animation_data else None
    if not action or not action.get('sp_video_source'):raise ValueError('请先生成并使用本插件的视频动作')
    return rig,action

def clear_markers(scene):
    s=scene.screenshot_pose
    for name,frame in json.loads(s.video_review_markers or '[]'):
        marker=scene.timeline_markers.get(name)
        if marker and marker.frame==frame:scene.timeline_markers.remove(marker)
    s.video_review_markers=''

def refresh(scene):
    global _REVIEW
    guard();s=scene.screenshot_pose;rig,action=action_for(s)
    path=Path(bpy.path.abspath(action.get('sp_video_report') or s.video_report))
    if not path.is_file():raise ValueError('未找到此动作的复核记录，请重新生成动作')
    report=json.loads(path.read_text(encoding='utf8'))
    source=Path(bpy.path.abspath(action['sp_video_source'])).resolve()
    if Path(report['source']).resolve()!=source or (not action.get('sp_video_report') and report['action']!=action.name):
        raise ValueError('复核记录与当前动作不匹配，请选择对应动作')
    data=video_data.read(source);frames=report.get('frame_map')
    if not frames:
        timed=video_data.timed_frames(data,scene.render.fps/scene.render.fps_base,int(report['frames'][0]),s.video_max_gap)
        if [f for f,item in timed]!=report['frames']:raise ValueError('旧动作的帧率与当前场景不符，请恢复生成时帧率')
        frames=[dict(frame=f,time=item['time'],pose=item['pose']) for f,item in timed]
    resampling=report.get('resampling',{});plan=video_data.reference_plan(data,frames,resampling.get('fps'),resampling.get('start_frame'))
    references=[]
    for item in plan['rows']:
        image=source.parent/item['image_relative']
        if item['status']=='ok' and not image.is_file():
            pose=video_data.child(source.parent,item['pose'])
            document=json.loads(pose.read_text(encoding='utf8'));image=Path(document['image']['path'])
            if not image.is_absolute():image=pose.parent/image
        references.append(dict(item,image=str(image.resolve()) if image.is_file() else ''))
    clear_markers(scene);s.video_issues.clear()
    edits=json.loads(action.get('sp_video_edits','[]'));marker_records=[];token=uuid.uuid4().hex[:6]
    a=addon();mapping=a.pose_tools.mapping_for(s)
    finger_sides={b.name:side for side in ('L','R') for finger in a.finger_pose.FINGERS for n in (1,2,3) if (b:=a.finger_pose.control(rig,finger,n,side))}
    for issue in report.get('temporal_review',{}).get('issues',[]):
        row=s.video_issues.add();row.frame=int(issue['frame']);row.control=issue['control'];row.reason=issue['reason'];row.repaired=bool(issue['repaired'])
        side=finger_sides.get(row.control)
        if side:row.scope_hint='HAND_'+side
        else:
            side=next((side for side in ('L','R') if mapping.get('hand.'+side)==row.control),None)
            if side:row.scope_hint=('HAND_' if issue.get('controls') else 'WRIST_')+side
        row.adjusted=any(edit['start']<row.frame<edit['end'] and row.control in edit['bones'] for edit in edits)
        marker=scene.timeline_markers.new('SP复核 '+token+' '+str(row.frame)+' '+row.control,frame=row.frame)
        marker_records.append([marker.name,marker.frame])
    s.video_review_markers=json.dumps(marker_records);s.video_issue_index=0
    _REVIEW=dict(scene=scene,rig=rig,action=action,frames=references,numbers=[v['frame'] for v in references],times=[v['time'] for v in references],
        fps=plan['fps'],origin_time=plan['origin_time'],start_frame=plan['start_frame'],end_frame=plan['end_frame'])
    s.video_review_status='已标记 %d 处；勾选表示自动修补，铅笔表示人工调整后待复核'%len(s.video_issues)
    update_reference(scene)

def reference_context(scene):
    if not _REVIEW or _REVIEW['scene']!=scene or scene.screenshot_pose.target!=_REVIEW['rig']:return None
    rig=_REVIEW['rig'];current=rig.animation_data.action if rig.animation_data else None
    if current!=_REVIEW['action'] and not (active() and _EDIT.scene==scene and current in (_EDIT.draft,_EDIT.preview,_EDIT.original)):return None
    return _REVIEW

def update_reference(scene):
    context=reference_context(scene);s=scene.screenshot_pose
    if not context:s.video_reference_path='';return
    frame=scene.frame_current
    if not context['start_frame']<=frame<=context['end_frame']:s.video_reference_path='';s.video_reference_info='';return
    stamp=context['origin_time']+(frame-context['start_frame'])/context['fps'];times=context['times']
    index=min(bisect.bisect_left(times,stamp),len(times)-1)
    if index and abs(times[index-1]-stamp)<abs(times[index]-stamp):index-=1
    row=context['frames'][index];s.video_reference_path=row['image'];s.video_reference_info='原视频 %.3f 秒 · 附近采样对应第 %d 帧'%(row['time'],row['frame'])
    if row['status']=='missing':s.video_reference_info+=' · 该采样未检出人物'

def reference_icon(path):
    global _ICONS,_ICON_PATH
    if not path or not Path(path).is_file():return 0
    if _ICONS is None:_ICONS=bpy.utils.previews.new()
    if path!=_ICON_PATH:
        _ICONS.clear();_ICONS.load('frame',path,'IMAGE');_ICON_PATH=path
    return _ICONS['frame'].icon_id

def scope_names(s,scope):
    a=addon();mapping=a.pose_tools.mapping_for(s);side=scope[-1]
    roles=('neck','head') if scope=='HEAD' else tuple(r+'.'+side for r in (('thigh','shin','foot') if scope.startswith('LEG') else ('upper_arm','forearm','hand') if scope.startswith('ARM') else ('hand',)))
    if any(not mapping.get(r) or mapping[r] not in s.target.pose.bones for r in roles):raise ValueError('此部位骨骼映射不完整')
    names={mapping[r] for r in roles}
    if scope.startswith(('HAND','ARM')):
        for finger in a.finger_pose.FINGERS:
            for segment in (1,2,3):
                bone=a.finger_pose.control(s.target,finger,segment,side)
                if bone:names.add(bone.name)
    return sorted(names)

def curve_map(action):return {(f.data_path,f.array_index):f for f in addon().animation_preview.curves(action)}
def fingerprint(action):
    data=[(f.data_path,f.array_index,f.mute,f.extrapolation,[m.type for m in f.modifiers],[(list(p.co),p.interpolation,list(p.handle_left),list(p.handle_right),p.handle_left_type,p.handle_right_type) for p in f.keyframe_points]) for f in addon().animation_preview.curves(action)]
    return hashlib.sha256(json.dumps(data).encode()).hexdigest()

class EditSession:
    def __init__(self,context):
        a=addon();s=context.scene.screenshot_pose
        guard()
        if a._JOB or s.candidate_state:raise ValueError('请先结束当前任务或姿势候选预览')
        rig,original=action_for(s)
        if hasattr(original,'slots') and len(original.slots)>1:raise ValueError('请先将当前角色动作拆成独立动作再修正')
        if rig.library and not rig.override_library:raise ValueError('链接角色需要先创建库覆盖')
        if context.mode not in ('OBJECT','POSE') or scene_subframe(context.scene):raise ValueError('请在物体或姿态模式的整数帧开始修正')
        if any(not t.mute for t in rig.animation_data.nla_tracks):raise ValueError('当前有 NLA 叠加动作，请先使用生成的视频动作状态再修正')
        self.scene=context.scene;self.rig=rig;self.s=s;self.original=original;self.scope=s.video_edit_scope
        self.names=scope_names(s,self.scope);self.curves=curve_map(original);self.prefix={n:rig.pose.bones[n].path_from_id() for n in self.names}
        for n in self.names:
            for (path,index),curve in self.curves.items():
                if path.startswith(self.prefix[n]+'.'):
                    if path.endswith(('rotation_euler','rotation_axis_angle')):raise ValueError('此部位使用欧拉旋转，请先重新生成插件视频动作')
                    if curve.modifiers or any(p.interpolation not in ('LINEAR','CONSTANT') for p in curve.keyframe_points):raise ValueError('此部位含贝塞尔或曲线修改器，请先烘焙为线性关键帧再修正')
        self.before=a.retarget.snapshot_pose(rig);self.original_hash=fingerprint(original);self.auto_key=self.scene.tool_settings.use_keyframe_insert_auto
        self.closed=False;self.preview=None;self.state='READY';self.anchors={};self.updating=False;self.previous_q={}
        self.draft=original.copy();self.draft.name=original.name+' · 修正工作副本';self.draft.use_fake_user=False;original.use_fake_user=True
        self.scene.tool_settings.use_keyframe_insert_auto=False
        a.retarget._assign_action(rig,self.draft,self.before.get('action_slot_handle'))
        self.defaults={n:dict(rotation=list(rig.pose.bones[n].matrix_basis.to_quaternion()),location=list(rig.pose.bones[n].location)) for n in self.names}
        self.selected=[b.name for b in rig.pose.bones if getattr(b, "select", getattr(b.bone, "select", False))];self.active_bone=rig.data.bones.active.name if rig.data.bones.active else None
        try:self.stage()
        except Exception:self.cancel();raise

    def validate(self):
        if self.closed:raise ValueError('局部修正会话已经结束')
        if bpy.context.scene!=self.scene or self.s.target!=self.rig:raise ValueError('场景或目标角色已改变')
        if self.rig.animation_data.action not in (self.draft,self.preview):raise ValueError('当前动作已改变，请撤回局部修正后重试')
        if self.rig.mode=='EDIT':raise ValueError('请退出骨骼编辑模式')

    def base(self,frame,name,normalize=True):
        values=[]
        for channel,size in (('rotation_quaternion',4),('location',3)):
            default=self.defaults[name]['rotation' if size==4 else 'location']
            values.append([self.curves[(self.prefix[name]+'.'+channel,i)].evaluate(frame) if (self.prefix[name]+'.'+channel,i) in self.curves else default[i] for i in range(size)])
        return video_patch.unit(values[0]) if normalize else values[0],values[1]

    def stage(self):
        self.validate()
        if self.state=='PREVIEW':raise ValueError('请先返回修正点编辑或确认预览')
        if scene_subframe(self.scene):raise ValueError('请切换到整数帧')
        self.updating=True
        try:self.scene.frame_set(self.scene.frame_current)
        finally:self.updating=False
        self.edit_frame=self.scene.frame_current;self.stage_before=addon().retarget.snapshot_pose(self.rig)
        for n in self.names:
            bone=self.rig.pose.bones[n];bone.rotation_mode='QUATERNION';q,p=self.base(self.edit_frame,n)
            if self.edit_frame in self.anchors:
                old=self.anchors[self.edit_frame][n];q=video_patch.multiply(old['rotation'],q);p=[x+y for x,y in zip(p,old['location'])]
            bone.rotation_quaternion=q;bone.location=p
        mapping=addon().pose_tools.mapping_for(self.s);role='head' if self.scope=='HEAD' else ('thigh.' if self.scope.startswith('LEG') else 'upper_arm.' if self.scope.startswith('ARM') else 'hand.')+self.scope[-1]
        primary=mapping[role]
        for b in self.rig.pose.bones:setattr(b if hasattr(b,"select") else b.bone,"select",b.name==primary)
        self.rig.data.bones.active=self.rig.data.bones[primary]
        self.state='EDIT';self.s.video_review_status='第 %d 帧：在视图中调整此部位，然后点击“记录修正点”；请勿手动打关键帧'%self.edit_frame

    def capture(self):
        self.validate()
        if self.state!='EDIT' or self.scene.frame_current!=self.edit_frame:raise ValueError('请先点击“编辑当前帧”')
        anchors={}
        for n in self.names:
            bone=self.rig.pose.bones[n];q,p=self.base(self.edit_frame,n)
            if any(abs(a-b)>1e-5 for a,b in zip(bone.scale,self.stage_before['bones'][n]['scale'])):raise ValueError('区间修正只传播旋转和位移，请先恢复骨骼缩放')
            anchors[n]=dict(rotation=video_patch.delta(q,bone.matrix_basis.to_quaternion()),location=[a-b for a,b in zip(bone.location,p)])
        self.anchors[self.edit_frame]=anchors;self.state='READY'
        self.s.video_review_status='已记录 %d 个修正点：%s；可以换帧继续记录，或预览区间效果'%(len(self.anchors),'、'.join(map(str,sorted(self.anchors))))

    def start_preview(self):
        self.validate()
        if self.state=='EDIT':raise ValueError('请先记录当前修正点')
        if fingerprint(self.original)!=self.original_hash:raise ValueError('原动作已被其他操作修改，请撤回后重试')
        start,end=self.s.video_edit_start,self.s.video_edit_end;video_patch.validate(start,end,self.anchors)
        self.start,self.end=start,end;self.preview_frame=self.scene.frame_current
        old=self.preview;self.preview=self.original.copy();self.preview.name=self.original.name+' · 区间修正预览';self.preview.use_fake_user=False
        addon().retarget._assign_action(self.rig,self.preview,self.before.get('action_slot_handle'))
        if old and old.users==0:bpy.data.actions.remove(old)
        self.state='BUILDING';self.generator=self.build()

    def back(self):
        self.validate()
        if self.state!='PREVIEW':raise ValueError('当前没有可返回的预览')
        old=self.preview;addon().retarget._assign_action(self.rig,self.draft,self.before.get('action_slot_handle'));self.preview=None
        if old.users==0:bpy.data.actions.remove(old)
        self.state='READY';self.scene.frame_set(self.scene.frame_current)
        self.s.video_review_status='已返回修正点编辑，已记录的修正点仍保留'

    def build(self):
        # Create absent transform channels only in our independent Action copy.
        for n in self.names:
            bone=self.rig.pose.bones[n];bone.rotation_mode='QUATERNION';q,p=self.base(self.start,n);bone.rotation_quaternion=q;bone.location=p
            for channel in ('rotation_quaternion','location'):
                if any((self.prefix[n]+'.'+channel,i) not in self.curves for i in range(4 if channel=='rotation_quaternion' else 3)):
                    bone.keyframe_insert(data_path=channel,frame=self.start,group=n)
        curves=curve_map(self.preview);times=set(range(self.start,self.end+1));targets=[]
        for n in self.names:
            for channel,size in (('rotation_quaternion',4),('location',3)):
                for i in range(size):
                    curve=curves[(self.prefix[n]+'.'+channel,i)];source=self.curves.get((curve.data_path,i))
                    times.update(float(p.co.x) for p in curve.keyframe_points if self.start<p.co.x<self.end)
                    outside=[(list(p.co),p.interpolation) for p in curve.keyframe_points if p.co.x<self.start or p.co.x>self.end]
                    outgoing='LINEAR'
                    if source:
                        prior=[p for p in source.keyframe_points if p.co.x<=self.end]
                        if prior:outgoing=prior[-1].interpolation
                    for p in list(curve.keyframe_points):
                        if self.start<=p.co.x<=self.end:curve.keyframe_points.remove(p,fast=True)
                    targets.append((n,channel,i,curve,outgoing,outside))
        for index,frame in enumerate(sorted(times)):
            solved={}
            for n in self.names:
                q,p=self.base(frame,n);q,p=video_patch.apply(frame,n,q,p,self.start,self.end,self.anchors)
                # Keep signs aligned to the original keys at both boundaries.
                if n in self.previous_q and sum(a*b for a,b in zip(q,self.previous_q[n]))<0:q=[-v for v in q]
                elif n not in self.previous_q and sum(a*b for a,b in zip(q,self.base(frame,n)[0]))<0:q=[-v for v in q]
                if frame==self.end and sum(a*b for a,b in zip(q,self.base(frame,n)[0]))<0:raise ValueError('修正点之间形成整圈旋转，请缩短区间或增加中间修正点')
                if frame in (self.start,self.end):q,p=self.base(frame,n,normalize=False)
                self.previous_q[n]=q;solved[n]=dict(rotation_quaternion=q,location=p)
            for n,channel,i,curve,outgoing,outside in targets:
                point=curve.keyframe_points.insert(frame,solved[n][channel][i],options={'FAST'});point.interpolation=outgoing if frame==self.end else 'LINEAR'
            self.s.video_review_status='预览局部修正 %d / %d 帧'%(index+1,len(times))
            yield
        for n,channel,i,curve,outgoing,outside in targets:curve.update()
        self.preview_hash=fingerprint(self.preview)
        self.state='PREVIEW';self.updating=True
        try:self.scene.frame_set(self.preview_frame)
        finally:self.updating=False
        self.s.video_review_status='预览已生成：可播放或拖动时间轴；确认后保留新动作，撤回则恢复原动作'

    def tick(self):
        self.validate()
        try:next(self.generator);return True
        except StopIteration:return False

    def finish(self):
        self.validate()
        if self.state!='PREVIEW':raise ValueError('请先预览区间效果')
        if fingerprint(self.original)!=self.original_hash:raise ValueError('原动作已被其他操作修改，请撤回后重试')
        if fingerprint(self.preview)!=self.preview_hash:raise ValueError('预览动作被另行打帧，请返回修正点重新预览')
        action=self.preview;action.name=self.original.name+' · 局部修正';action.use_fake_user=True;self.original.use_fake_user=True
        edits=json.loads(action.get('sp_video_edits','[]'));edits.append(dict(start=self.start,end=self.end,scope=self.scope,bones=self.names,anchors=self.anchors))
        action['sp_video_edits']=json.dumps(edits);self.scene.tool_settings.use_keyframe_insert_auto=self.auto_key
        self.closed=True
        if self.draft.users==0:bpy.data.actions.remove(self.draft)
        self.s.video_review_status='已保存局部修正为新动作；原动作保留'
        return action

    def cancel(self):
        if self.closed:return
        current=self.rig.animation_data.action if self.rig.animation_data else None;frame=self.scene.frame_current
        self.closed=True;self.scene.tool_settings.use_keyframe_insert_auto=self.auto_key
        if current in (self.draft,self.preview):
            addon().retarget.restore_pose(self.rig,self.before,update=False);self.scene.frame_set(frame)
        self.original.use_fake_user=self.before['action_fake_user']
        for action in (self.draft,self.preview):
            if action and action.users==0:bpy.data.actions.remove(action)
        for b in self.rig.pose.bones:setattr(b if hasattr(b,"select") else b.bone,"select",b.name in self.selected)
        self.rig.data.bones.active=self.rig.data.bones.get(self.active_bone) if self.active_bone else None
        self.s.video_review_status='已撤回未确认的视频局部修正'

def scene_subframe(scene):return abs(scene.frame_subframe)>1e-6
def cancel():
    global _EDIT
    if active():_EDIT.cancel()
    _EDIT=None

@bpy.app.handlers.persistent
def frame_change(scene,*args):
    if active() and _EDIT.scene==scene and not _EDIT.updating and _EDIT.state=='EDIT' and scene.frame_current!=_EDIT.edit_frame:
        cancel();scene.screenshot_pose.video_review_status='换帧前未记录修正点，未确认的修改已撤回'

@bpy.app.handlers.persistent
def after_frame(scene,*args):
    try:
        if hasattr(scene,'screenshot_pose'):update_reference(scene)
    except (ReferenceError,KeyError):pass

@bpy.app.handlers.persistent
def save_pre(*args):
    if active() and _EDIT.state=='BUILDING':addon().stop_job()
    cancel()

def clean():
    global _REVIEW,_ICONS,_ICON_PATH
    cancel();_REVIEW=None
    if _ICONS is not None:bpy.utils.previews.remove(_ICONS);_ICONS=None
    _ICON_PATH=''

class SP_OT_video_review(Operator):
    bl_idname='screenshot_pose.video_review';bl_label='视频复核与局部修正'
    action:EnumProperty(items=[(v,v,'') for v in ('REFRESH','GOTO','PREV','NEXT','REFERENCE','BEGIN','EDIT','CAPTURE','REMOVE','PREVIEW','BACK','CONFIRM','CANCEL','CONTACT_PREVIEW')])
    def execute(self,context):
        global _EDIT
        s=context.scene.screenshot_pose;a=addon()
        try:
            if a._JOB:raise ValueError('请先等待或取消当前任务')
            if self.action=='REFRESH':refresh(context.scene)
            elif self.action in ('GOTO','PREV','NEXT'):
                guard()
                if not reference_context(context.scene):raise ValueError('请先刷新当前动作的问题列表')
                if not len(s.video_issues):raise ValueError('当前没有标记的问题帧')
                if self.action=='PREV':s.video_issue_index=max(0,s.video_issue_index-1)
                elif self.action=='NEXT':s.video_issue_index=min(len(s.video_issues)-1,s.video_issue_index+1)
                s.video_issue_index=min(s.video_issue_index,len(s.video_issues)-1);item=s.video_issues[s.video_issue_index];context.scene.frame_set(item.frame)
                preferred=[(item.scope_hint,'','')] if item.scope_hint else []
                for scope,label,description in preferred+a.pose_tools.SCOPES:
                    try:names=scope_names(s,scope)
                    except ValueError:continue
                    if item.control in names:s.video_edit_scope=scope;break
            elif self.action=='REFERENCE':
                path=Path(s.video_reference_path)
                if not path.is_file() or path.suffix.lower() not in ('.png','.jpg','.jpeg','.webp','.bmp'):raise ValueError('此帧的参考图不可用')
                bpy.ops.wm.path_open(filepath=str(path))
            elif self.action=='CONTACT_PREVIEW':
                from .video_contacts import ContactSession
                _EDIT=ContactSession(context)
                try:_EDIT.start_preview()
                except Exception:cancel();raise
                a._JOB=dict(tick=_EDIT.tick,cancel=cancel,scene=context.scene,purpose='接触区间预览')
                bpy.app.timers.register(a.poll_job,first_interval=.02)
            elif self.action=='BEGIN':
                _EDIT=EditSession(context);s.video_edit_start=context.scene.frame_current-8;s.video_edit_end=context.scene.frame_current+8
            elif self.action=='CANCEL':cancel()
            else:
                if not active():raise ValueError('请先开始局部修正')
                if self.action=='EDIT':_EDIT.stage()
                elif self.action=='CAPTURE':_EDIT.capture()
                elif self.action=='BACK':_EDIT.back()
                elif self.action=='REMOVE':
                    if _EDIT.state!='READY':raise ValueError('请先记录当前修正点')
                    _EDIT.anchors.pop(context.scene.frame_current,None);s.video_review_status='剩余修正点：'+str(sorted(_EDIT.anchors))
                elif self.action=='PREVIEW':
                    _EDIT.previous_q={};_EDIT.start_preview();a._JOB=dict(tick=_EDIT.tick,cancel=cancel,scene=context.scene,purpose='生成局部预览')
                    bpy.app.timers.register(a.poll_job,first_interval=.02)
                elif self.action=='CONFIRM':
                    _EDIT.finish();_EDIT=None
                    try:refresh(context.scene)
                    except (OSError,ValueError,KeyError):pass
                    s.video_review_status='局部修正已确认；铅笔标记的区间请回放复核'
            a._redraw()
        except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}
        return {'FINISHED'}

CLASSES=(SPVideoIssue,SP_UL_video_issues,SP_OT_video_review)
