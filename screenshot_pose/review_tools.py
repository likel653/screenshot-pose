"""Cached review UI data and navigation; no model or FK work during draw."""
import hashlib,importlib,json
from pathlib import Path
import bpy
from bpy.props import EnumProperty,StringProperty
from bpy.types import Operator
from . import review_data

_CACHE={};_FILES={}
def addon():return importlib.import_module(__package__)
def context_stamp(s,digest=None):
    a=addon();options=set(a.pose_library.OPTIONS)|set(a.character_tools.FIELDS)|{'strength','head_strength','hand_strength','orientation_threshold','foot_threshold','contact_enabled','ground_feet','support_l','support_r','hand_contact_l','hand_contact_r'}
    options.update(('floor_height','support_height_l','support_height_r','support_move_pelvis','support_pelvis_limit','keyframe'))
    params={k:getattr(s,k) for k in sorted(options) if k!='keyframe'}
    params['targets']={k:(getattr(s,k).name,[list(row) for row in getattr(s,k).matrix_world]) if getattr(s,k) else None for k in ('support_target_l','support_target_r','knee_target_l','knee_target_r','hand_target_l','hand_target_r')}
    return dict(signature=digest if digest is not None else document(s)[1],target=s.target.name if s.target else '',frame=bpy.context.scene.frame_current,
        settings=hashlib.sha256(json.dumps(params,sort_keys=True).encode()).hexdigest(),mapping=[(r.role,r.bone,r.enabled,bool(s.target and r.bone in s.target.pose.bones)) for r in s.mapping])
def document(s):
    path=Path(bpy.path.abspath(s.json_path)) if s.json_path else None
    if not path:return {},None
    try:
        st=path.stat();key=(str(path.resolve()),st.st_mtime_ns,st.st_size)
        if key not in _FILES:
            if len(_FILES)>=8:_FILES.clear()
            raw=path.read_bytes() if st.st_size<=4*1024*1024 else b''
            data=addon().pose_data.validate(json.loads(raw.decode('utf-8-sig')))
            _FILES[key]=(data,hashlib.sha256(raw).hexdigest())
        return _FILES[key]
    except (OSError,ValueError,KeyError,TypeError):return {},None
def summary(s):
    data,digest=document(s)
    current=context_stamp(s,digest)
    key=(json.dumps(current,sort_keys=True),hash(s.last_result),s.target.as_pointer() if s.target else 0)
    if key not in _CACHE:
        if len(_CACHE)>=8:_CACHE.clear()
        try:result=json.loads(s.last_result) if s.last_result else {}
        except (ValueError,TypeError):result={}
        stamp=result.get('review_context',{})
        fresh=bool(digest and stamp==json.loads(json.dumps(current)))
        rows=review_data.issues(data,result if fresh else {},s.orientation_threshold,s.foot_threshold)
        if s.target:
            for role in addon().retarget.ROLE_KEYS:
                row=next((r for r in s.mapping if r.role==role),None)
                if row and row.enabled and row.bone in s.target.pose.bones:continue
                base,_,side=role.partition('.');part=('LEG_'+side if base in ('thigh','shin','foot') else 'HAND_'+side if base=='hand' else 'ARM_'+side if base in ('upper_arm','forearm') else 'HEAD' if base in ('head','neck') else 'BODY')
                rows.append(dict(id='map_'+role,part=part,title=review_data.LABELS[part],message=addon().label(role)+'未启用或缺少控制骨，需检查映射。',fix='MAPPING',severity='REVIEW'))
        _CACHE[key]=dict(rows=rows,data=data,fresh=fresh,has_result=bool(result))
    return _CACHE[key]
def clear():_CACHE.clear();_FILES.clear()

class SP_OT_review_focus(Operator):
    bl_idname='screenshot_pose.review_focus';bl_label='定位疑点'
    part:EnumProperty(items=review_data.PARTS,default='BODY')
    fix:StringProperty(default='LOCAL')
    def execute(self,context):
        a=addon();s=context.scene.screenshot_pose
        if a._JOB or s.candidate_state or a.video_review.active():self.report({'INFO'},'请先结束当前任务或预览');return {'CANCELLED'}
        s.ui_section='CORRECT';s.ui_part=self.part
        if self.part!='BODY':s.local_scope=self.part
        if self.part.startswith(('HAND_','ARM_')):s.hand_edit_side=self.part[-1]
        if self.fix=='MAPPING':s.ui_section='ADVANCED'
        elif self.fix=='CALIBRATION':s.ui_section='ADVANCED'
        elif self.fix in ('CANDIDATES','FIT'):s.ui_part='BODY'
        s.status='已定位'+review_data.LABELS[self.part]+'；请使用下方对应工具复核'
        panel={'MAPPING':'SP_PT_mapping','CALIBRATION':'SP_PT_character','IMAGE':'SP_PT_tools','HAND':'SP_PT_hand_edit','FEET':'SP_PT_feet','CANDIDATES':'SP_PT_candidates','FIT':'SP_PT_fit','LOCAL':'SP_PT_orientation'}.get(self.fix)
        if self.fix=='IMAGE':s.ui_section='ADVANCED'
        if self.fix=='FEET' and not self.part.startswith('LEG_'):panel='SP_PT_fit';s.ui_part='BODY'
        if self.fix=='HAND' and not self.part.startswith(('HAND_','ARM_')):panel='SP_PT_hand_models';s.ui_section='ADVANCED'
        if panel and context.area and context.area.type=='VIEW_3D':bpy.ops.wm.call_panel(name=panel,keep_open=True)
        return {'FINISHED'}

class SP_OT_review_refresh(Operator):
    bl_idname='screenshot_pose.review_refresh';bl_label='刷新疑点'
    def execute(self,context):
        clear();info=summary(context.scene.screenshot_pose);context.scene.screenshot_pose.status=f'已更新 {len(info["rows"])} 条复核提示；不会修改人物姿势';return {'FINISHED'}

class SP_OT_workflow_apply(Operator):
    bl_idname='screenshot_pose.workflow_apply';bl_label='预览或确认姿势';bl_options={'REGISTER','UNDO'}
    action:EnumProperty(items=[('PREVIEW','预览姿势','不写入关键帧'),('KEYFRAME','确认并打帧','生成独立动作，保留原动画')])
    def execute(self,context):
        a=addon();s=context.scene.screenshot_pose;old=s.keyframe
        try:
            if a._JOB or s.candidate_state or a.video_review.active():raise ValueError('请先结束当前任务或预览')
            s.keyframe=self.action=='KEYFRAME';a.apply_current(context,s)
            result=a.pose_tools.last_result(s);result['review_context']=context_stamp(s)
            s.last_result=json.dumps(result,ensure_ascii=False)
        except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}
        finally:s.keyframe=old
        return {'FINISHED'}

OPERATORS=(SP_OT_review_focus,SP_OT_review_refresh,SP_OT_workflow_apply)
