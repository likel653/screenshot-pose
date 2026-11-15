"""Compact native Blender workflow with optional detail panels."""
import importlib,json,textwrap
from pathlib import Path
import bpy
from bpy.types import Panel

def addon():return importlib.import_module(__package__)
def exists(path):return bool(path and Path(bpy.path.abspath(path)).is_file())
def wrap(layout,context,text,icon='NONE'):
 region=context.region
 region_width=getattr(region,'width',300) if getattr(region,'type',None)=='UI' else 300
 width=max(10,int((region_width-55)/14))
 for i,line in enumerate(textwrap.wrap(text.rstrip('。'),width=width,break_long_words=True,break_on_hyphens=False) or ['']):
  layout.label(text=line,icon=icon if i==0 else 'NONE')
def column(layout):
 c=layout.column(align=False);c.use_property_split=False;c.use_property_decorate=False
 c.enabled=addon()._JOB is None and not addon().video_review.active() and not bool(bpy.context.scene.screenshot_pose.candidate_state)
 return c
def ready(s):
 a=addon();root=a.runtime_dir()
 files=[a.runtime_python(s.backend),root/'pose_landmarker_full.task']
 if s.backend in ('RTMW','RTMW3D','SAM3D'):files.append(root/'rtmw_detector.onnx')
 if s.backend in ('RTMW','SAM3D'):files.append(root/'rtmw_pose.onnx')
 if s.backend in ('RTMW3D','SAM3D'):files.append(root/'rtmw3d_pose.onnx')
 if s.backend=='SAM3D':
  args=a.sam_root_args();native=Path(args[1]) if args else a.HERE.parent
  files.extend(native/'native_build/vulkan/bin'/n for n in ('sam3d-body-infer.exe','ggml-vulkan.dll'))
  files.extend(root/'sam3d_gguf'/n for n in ('body-dinov3-f32.gguf','body-pose-branch-f32.gguf','mhr-lod1-f32.gguf'))
 return all(p.is_file() for p in files)

def hand_choices(c,s):
 choices=addon().hand_tools.comparison_choices(s)
 if not choices:wrap(c,bpy.context,'先在“设置 → 手部模型与结果选择”生成模型对比。');return
 if exists(s.hand3d_comparison):c.operator('screenshot_pose.hand3d_comparison',icon='IMAGE_DATA')
 c.prop(s,'hand_adopt_side');c.prop(s,'hand_adopt_component')
 for backend in choices:
  label={'WILOR':'WiLoR','HAMER':'HaMeR','HAMER_TEXTURE':'增强 HaMeR'}[backend]
  c.operator('screenshot_pose.hand3d_adopt',text='采用 '+label).backend=backend
 data,_=addon().review_tools.document(s);hd=data.get('orientation',{}).get('hands',{}).get(s.hand_edit_side,{})
 selection=hd.get('component_selection')
 if selection:
  wrap(c,bpy.context,'当前掌向：'+str(selection.get('palm_source')))
  wrap(c,bpy.context,'当前手指：'+str(selection.get('finger_source')))
 wrap(c,bpy.context,'拆分选择会保留另一部分，并对齐掌内形状。不会覆盖人工校正，也不会自动应用角色。')

class SP_PT_main(Panel):
 bl_label='截图转姿势'
 bl_idname='SP_PT_main';bl_space_type='VIEW_3D';bl_region_type='UI';bl_category='Screenshot Pose'
 def draw(self,context):
  a=addon();s=context.scene.screenshot_pose;l=self.layout;l.use_property_decorate=False
  busy=a._JOB is not None;has_pose=exists(s.json_path);is_ready=ready(s)
  box=l.box();box.label(text='目标角色',icon='ARMATURE_DATA')
  c=column(box);c.prop(s,'target',text='')
  if s.target:
   mapped=sum(bool(r.bone) and r.enabled for r in s.mapping)
   box.label(text=f'已匹配 {mapped} / {len(a.retarget.ROLE_KEYS)} 个部位',icon='CHECKMARK' if mapped==len(a.retarget.ROLE_KEYS) else 'INFO')
  tabs=l.column(align=True)
  for sections in (('GENERATE','CORRECT','ADVANCED'),('LIBRARY','VIDEO')):
   row=tabs.row(align=True)
   for section in sections:row.prop_enum(s,'ui_section',section)
  status=l.box()
  wrap(status,context,s.status or '选择角色后，框选截图或载入参考图。','TIME' if busy else 'ERROR' if s.status.startswith('失败') else 'INFO')
  if busy:status.operator('screenshot_pose.cancel',text='取消当前任务',icon='CANCEL')
  if s.candidate_state:
   if a.direct_tools.active(s):status.operator('screenshot_pose.direct_action',text='取消局部拖动并恢复',icon='LOOP_BACK').action='CANCEL'
   elif a.hand_preview.active(s):status.operator('screenshot_pose.palm_preview',text='取消掌向预览并恢复',icon='LOOP_BACK').action='CANCEL'
   elif json.loads(s.candidate_state).get('kind')=='FOOT_DIRECTION':status.operator('screenshot_pose.foot_preview',text='取消脚向预览并恢复',icon='LOOP_BACK').action='CANCEL'
   elif json.loads(s.candidate_state).get('kind')=='BEFORE_AFTER':status.operator('screenshot_pose.pose_compare',text='返回当前姿势',icon='LOOP_BACK')
   else:status.operator('screenshot_pose.candidates',text='取消候选预览并恢复',icon='LOOP_BACK').action='CANCEL'
  if s.ui_section=='CORRECT':
   c=column(l);c.prop(s,'ui_part',text='修正部位')
   wrap(c,context,'左右均指角色自身的左右。局部更新保留其他部位。')
   if s.ui_part!='BODY':
    row=c.row();row.enabled=bool(s.target and has_pose and a.preview.root_for(s));row.operator('screenshot_pose.apply_local',text='仅更新'+a.review_tools.review_data.LABELS[s.ui_part],icon='CHECKMARK').scope=s.ui_part
   row=c.row(align=True);row.enabled=bool(s.target and has_pose);row.operator('screenshot_pose.show_result',text='详细结果',icon='INFO');row.operator('screenshot_pose.transfer_diagnosis',text='骨架传递诊断',icon='VIEWZOOM')
  if s.ui_section!='GENERATE':return
  box=l.box();box.label(text='01  获取参考图',icon='IMAGE_DATA');c=column(box)
  c.prop(s,'backend',text='模型')
  c.prop(s,'screenshot_hands')
  if s.screenshot_hands!='OFF':wrap(c,context,'身体识别后自动精修双手；未就绪或失败时保留已有观测，不自动下载。')
  if not is_ready:wrap(box,context,'当前模型尚未就绪，请到“设置 → 模型与环境”。','ERROR')
  row=c.row();row.enabled=is_ready;row.scale_y=1.5
  row.operator('screenshot_pose.clipboard',text='框选截图并识别',icon='RESTRICT_VIEW_OFF')
  box.label(text='立即拖框 · 原亮度 · Esc 取消',icon='INFO')
  c.prop(s,'image_path',text='图片')
  row=c.row();row.enabled=is_ready and exists(s.image_path)
  row.operator('screenshot_pose.detect',text='识别所选图片',icon='VIEWZOOM')
  if has_pose:
   data,_=a.review_tools.document(s);reference=data.get('image',{}).get('path','')
   if reference and exists(reference):
    try:c.template_icon(icon_value=a.pose_library.icon(reference),scale=6.)
    except (OSError,ValueError,RuntimeError):pass
    wrap(c,context,'已识别：'+Path(reference).name)
   row=c.row();row.enabled=not busy and not s.candidate_state
   row.operator('screenshot_pose.library_save',text='加入本地姿势库',icon='ADD')
   row=c.row(align=True);row.operator('screenshot_pose.open_overlay',text='识别预览',icon='IMAGE_DATA');row.operator('screenshot_pose.edit_image',text='校正关节',icon='GREASEPENCIL')
  box=l.box();box.label(text='02  预览与应用',icon='POSE_HLT');c=column(box)
  c.prop(s,'strength',text='应用强度',slider=True)
  c.label(text=f'当前帧：{context.scene.frame_current}')
  if s.target and 'sp_animation_preview' in s.target:wrap(c,context,'临时姿势正在显示：原动画保留，换帧或恢复原姿势时退出预览。','INFO')
  row=c.row();row.scale_y=1.6;row.enabled=bool(s.target and has_pose and a.preview.root_for(s))
  row.operator('screenshot_pose.workflow_apply',text='预览姿势',icon='HIDE_OFF').action='PREVIEW'
  row=c.row();row.scale_y=1.4;row.enabled=bool(s.target and has_pose and a.preview.root_for(s))
  row.operator('screenshot_pose.workflow_apply',text='确认姿势并打帧',icon='KEY_HLT').action='KEYFRAME'
  wrap(c,context,'预览不写关键帧；确认时生成独立动作，保留原动画。')
  c.prop(s,'keyframe',text='局部修正 / 候选确认时打帧')
  if not s.target:box.label(text='先选择角色骨架',icon='INFO')
  elif not has_pose:box.label(text='识别图片或导入姿势后即可应用',icon='INFO')
  elif not a.preview.root_for(s):wrap(box,context,'关节点已移除，请在“关节编辑与数据”中重建。','INFO')
  row=c.row(align=True);r=row.row();r.enabled=bool(s.target and s.target.get('screenshot_pose_backup'))
  r.operator('screenshot_pose.restore',text='恢复原姿势',icon='LOOP_BACK')
  r=row.row();r.enabled=bool(s.last_result or s.quality_warnings);r.operator('screenshot_pose.show_result',text='查看结果',icon='INFO')
  row=c.row();row.enabled=bool(s.target and has_pose and s.last_result)
  row.operator('screenshot_pose.transfer_diagnosis',text='源姿势与角色诊断',icon='VIEWZOOM')

class SPDetailPanel:
 bl_space_type='VIEW_3D';bl_region_type='UI';bl_category='Screenshot Pose';bl_parent_id='SP_PT_main';bl_options={'DEFAULT_CLOSED'}
 @classmethod
 def poll(cls,context):
  s=context.scene.screenshot_pose
  section=_SECTIONS.get(cls.bl_idname,'CORRECT')
  if s.ui_section not in section:return False
  parts=_PART_FILTER.get(cls.bl_idname)
  return not parts or s.ui_part in parts

class SP_PT_issues(SPDetailPanel,Panel):
 bl_label='需要复核的部位';bl_idname='SP_PT_issues';bl_options=set()
 def draw(self,context):
  a=addon();s=context.scene.screenshot_pose;c=self.layout.column();info=a.review_tools.summary(s)
  c.operator('screenshot_pose.review_refresh',text='刷新复核提示',icon='FILE_REFRESH')
  if not info['data']:wrap(c,context,'识别参考图后显示具体复核依据。');return
  if info['has_result'] and not info['fresh']:wrap(c,context,'识别数据已变化或上次应用不匹配；不沿用旧的传递诊断。','INFO')
  elif info['fresh']:wrap(c,context,'传递提示来自最近一次应用；手动改骨后请重新应用或运行角色诊断。','INFO')
  if not info['rows']:wrap(c,context,'当前规则未发现明确疑点，仍需目视检查姿势。','INFO')
  rows=info['rows'] if s.ui_section=='GENERATE' else [r for r in info['rows'] if r['part']==s.ui_part]
  shown=rows[:4] if s.ui_section=='GENERATE' else rows
  for issue in shown:
   box=c.box();box.label(text=issue['title'],icon='INFO' if issue['severity']=='INFO' else 'QUESTION')
   wrap(box,context,issue['message'])
   row=column(box);op=row.operator('screenshot_pose.review_focus',text='定位并修正',icon='FORWARD');op.part=issue['part'];op.fix=issue['fix']
  if len(rows)>len(shown):wrap(c,context,f'另有 {len(rows)-len(shown)} 条提示；在“修正”中选择部位查看。')
  if not rows and info['rows']:wrap(c,context,'此部位未触发明确疑点，仍需目视复核。')
  wrap(c,context,'提示来自检测证据与传递检查，不是真实准确率。')

class SP_PT_library(SPDetailPanel,Panel):
 bl_label='我的本地姿势库';bl_idname='SP_PT_library'
 def draw(self,context):
  a=addon();s=context.scene.screenshot_pose;c=column(self.layout);lib=a.pose_library
  c.prop(s,'library_prompt');c.operator('screenshot_pose.library_save',text='保存当前截图与姿势',icon='ADD')
  row=c.row(align=True);row.prop(s,'library_search',text='搜索');row.operator('screenshot_pose.library_action',text='',icon='FILE_REFRESH').action='REFRESH'
  c.label(text=s.library_status)
  c.template_list('UI_UL_list','sp_library',s,'library_items',s,'library_index',rows=4)
  if 0<=s.library_index<len(s.library_items):
   try:
    key,meta=lib.selected(s);folder=lib.library_store.entry(lib.root(),key)
    c.template_icon(icon_value=lib.icon(lib.library_store.asset(folder,meta['image'])),scale=6.)
    if meta.get('tags'):wrap(c,context,'标签：'+meta['tags'])
    if meta.get('notes'):wrap(c,context,meta['notes'][:240]+('…' if len(meta['notes'])>240 else ''))
    c.operator('screenshot_pose.library_action',text='载入此姿势（不立即应用）',icon='IMPORT').action='LOAD'
    row=c.row(align=True);row.operator('screenshot_pose.library_action',text='改名 / 备注').action='RENAME';row.operator('screenshot_pose.library_action',text='移入回收目录').action='TRASH'
   except (ValueError,OSError,KeyError):wrap(c,context,'条目已变化或素材丢失，请刷新','ERROR')
  prefs=a.preferences()
  if prefs:c.prop(prefs,'pose_library_dir',text='库目录')
  c.operator('screenshot_pose.library_action',text='打开本地库文件夹',icon='FILE_FOLDER').action='OPEN'
  wrap(c,context,'载入后使用当前目标骨架重新适配。场景接触需重设；Blender 内直接改骨骼不属于通用识别数据。')

class SP_PT_video(SPDetailPanel,Panel):
 bl_label='视频转动画';bl_idname='SP_PT_video'
 def draw(self,context):
  a=addon();s=context.scene.screenshot_pose;c=column(self.layout)
  c.prop(s,'video_input',expand=True);c.prop(s,'video_backend',text='模式')
  c.prop(s,'hand3d_video_backend')
  if s.hand3d_video_backend!='LOCAL':wrap(c,context,'需要先在头部 / 手掌面板准备对应三维手部模型。按采样帧推理，失败帧会在报告中标出。')
  if s.video_backend=='SAM3D':c.prop(s,'video_continuous')
  if s.video_input=='SCREEN':
   c.prop(s,'video_record_seconds');c.prop(s,'video_record_fps');c.prop(s,'video_record_identify')
   c.operator('screenshot_pose.video_record',text='框选录制并识别' if s.video_record_identify else '框选屏幕并录制',icon='REC')
   wrap(c,context,'原亮度立即框选；选好区域后点击开始录制。录制期间保持视频可见、窗口位置不变，Esc 可取消。')
   if s.video_capture_summary:wrap(c,context,s.video_capture_summary)
  c.prop(s,'video_path',text='录制文件' if s.video_input=='SCREEN' else '视频')
  if s.video_input=='FILE':
   row=c.row(align=True);row.prop(s,'video_start',text='开始秒');row.prop(s,'video_end',text='结束秒')
  c.prop(s,'video_fps')
  c.prop(s,'video_sampling')
  if s.video_sampling=='ADAPTIVE' and s.video_fps>0:c.prop(s,'video_sampling_max')
  wrap(c,context,'0 = 逐帧识别。录完自动识别时无需再点①。建议先录 2–5 秒。' if s.video_input=='SCREEN' else '0 = 视频每帧；结束秒 0 = 视频结尾。先用短片段测试，SAM 逐帧识别耗时较长。')
  c.operator('screenshot_pose.video_detect',text='① 识别录制视频' if s.video_input=='SCREEN' else '① 识别视频并缓存',icon='SEQUENCE')
  c.prop(s,'video_cache',text='缓存');c.operator('screenshot_pose.video_import',text='载入已有视频缓存',icon='IMPORT')
  c.separator();c.prop(s,'video_start_frame');c.prop(s,'video_temporal_mode');c.prop(s,'video_smooth');c.prop(s,'video_max_gap')
  c.prop(s,'video_interpolation');c.prop(s,'video_root_motion')
  if s.video_root_motion!='IN_PLACE':c.prop(s,'video_root_gain')
  c.prop(s,'video_auto_support')
  if s.video_auto_support:c.prop(s,'floor_height');c.prop(s,'video_support_limit')
  c.prop(s,'video_stabilize');c.prop(s,'video_optimize')
  c.prop(s,'video_hand_continuity')
  if s.video_hand_continuity:c.prop(s,'video_hand_gap')
  c.prop(s,'video_wrist_repair')
  if s.video_wrist_repair:c.prop(s,'video_wrist_gap')
  c.prop(s,'video_local_memory')
  if s.video_local_memory:c.prop(s,'video_local_memory_seconds')
  c.prop(s,'video_native');c.prop(s,'video_fit');c.prop(s,'video_set_range')
  row=c.row();row.enabled=bool(s.target and exists(s.video_cache));row.scale_y=1.4
  row.operator('screenshot_pose.video_bake',text='② 生成到所选角色',icon='ACTION')
  action=s.target.animation_data.action if s.target and s.target.animation_data else None
  if action:
   support=c.box();support.enabled=a._JOB is None and not s.candidate_state
   if action.get(a.rgb_contact.SOURCE):support.operator('screenshot_pose.video_rgb_original',icon='LOOP_BACK')
   else:support.operator('screenshot_pose.video_rgb_contact',icon='CONSTRAINT_BONE')
   wrap(support,context,'可选后处理，生成独立动作。启用时间轴预览范围时，只处理该范围。')
  row=c.row();row.enabled=bool(s.target and 'sp_video_restore' in s.target)
  row.operator('screenshot_pose.video_restore',text='恢复生成前动画',icon='LOOP_BACK')
  if exists(s.video_report):c.operator('screenshot_pose.video_result',text='查看视频复核结果',icon='INFO')
  wrap(c,context,'单人、连续镜头；生成独立动画，保留原动作。识别缓存可重复用于其他角色。')
  wrap(c,context,'骨盆位移来自相对画面估计，移动镜头请用仅高度或原地模式。自动支撑是候选求解，抬脚、跳跃与遮挡需回放复核；不沿用单张图的接触目标。')

class SP_PT_video_review(SPDetailPanel,Panel):
 bl_label='视频复核与局部修正';bl_idname='SP_PT_video_review'
 def draw(self,context):
  a=addon();s=context.scene.screenshot_pose;v=a.video_review;c=self.layout.column();busy=a._JOB is not None;editing=v.active();session=v._EDIT if editing else None
  def button(layout,text,action,icon='NONE'):
   layout.operator('screenshot_pose.video_review',text=text,icon=icon).action=action
  row=c.row();row.enabled=not busy and not editing;button(row,'刷新问题与时间轴标记','REFRESH','FILE_REFRESH')
  if len(s.video_issues):
   c.template_list('SP_UL_video_issues','',s,'video_issues',s,'video_issue_index',rows=4)
   if s.video_issue_index<len(s.video_issues):
    item=s.video_issues[s.video_issue_index];wrap(c,context,item.reason+(' · 已人工调整，请回放复核' if item.adjusted else ''))
   row=c.row(align=True);row.enabled=not busy and not editing;button(row,'上一处','PREV','TRIA_LEFT');button(row,'定位','GOTO','TIME');button(row,'下一处','NEXT','TRIA_RIGHT')
  if s.video_reference_path or s.video_reference_info:
   icon=v.reference_icon(s.video_reference_path)
   if icon:c.template_icon(icon_value=icon,scale=8.)
   wrap(c,context,s.video_reference_info);row=c.row();row.enabled=not busy and exists(s.video_reference_path);button(row,'放大查看对应原图','REFERENCE','IMAGE_DATA')
  if editing and getattr(session,'contact_session',False):
   wrap(c,context,'正在生成或预览接触，请到视频接触面板确认或撤回。');return
  c.separator();row=c.row();row.enabled=not editing and not busy;row.prop(s,'video_edit_scope')
  if not editing:
   row=c.row();row.enabled=not busy and bool(s.target);button(row,'① 开始修正当前帧','BEGIN','GREASEPENCIL')
   wrap(c,context,'在姿态模式调整选定部位，再记录修正点。可以记录多个帧；无需重新识别视频。')
  else:
   row=c.row();row.enabled=not busy and session.state!='EDIT';row.prop(s,'video_edit_start');row.prop(s,'video_edit_end')
   wrap(c,context,'首尾帧保留原姿势；修正点放在区间内部。已记录：'+('、'.join(map(str,sorted(session.anchors))) or '无'))
   if session.state=='EDIT':
    row=c.row();row.enabled=not busy;button(row,'② 记录修正点','CAPTURE','KEY_HLT')
   elif session.state=='READY':
    row=c.row();row.enabled=not busy;button(row,'编辑当前帧','EDIT','GREASEPENCIL');button(row,'删除当前帧修正点','REMOVE','X')
    row=c.row();row.enabled=not busy and bool(session.anchors);button(row,'③ 预览区间效果','PREVIEW','PLAY')
   elif session.state=='PREVIEW':
    row=c.row();row.enabled=not busy;button(row,'返回修正点','BACK','BACK');button(row,'④ 确认保留新动作','CONFIRM','CHECKMARK')
   row=c.row();row.enabled=not busy;button(row,'撤回未确认修改','CANCEL','LOOP_BACK')
   wrap(c,context,'编辑期间自动打帧暂时关闭。请确认后再保存工程；保存会撤回未确认预览。')
  wrap(c,context,s.video_review_status)

class SP_PT_video_contacts(SPDetailPanel,Panel):
 bl_label='视频脚底锁定与手部接触';bl_idname='SP_PT_video_contacts'
 def draw(self,context):
  a=addon();s=context.scene.screenshot_pose;v=a.video_review;c=self.layout.column();active=v.active();busy=a._JOB is not None
  if active and getattr(v._EDIT,'contact_session',False):
   if v._EDIT.state=='PREVIEW':
    row=c.row();row.operator('screenshot_pose.video_review',text='确认接触修正',icon='CHECKMARK').action='CONFIRM'
   row=c.row();row.enabled=not busy;row.operator('screenshot_pose.video_review',text='撤回接触预览',icon='LOOP_BACK').action='CANCEL'
   wrap(c,context,s.video_review_status);return
  c.enabled=not active and not busy;c.prop(s,'video_lock_part')
  row=c.row();row.prop(s,'video_lock_start');row.prop(s,'video_lock_end')
  c.prop(s,'video_lock_fade');c.prop(s,'video_lock_pelvis')
  if s.video_lock_pelvis:c.prop(s,'video_lock_pelvis_limit')
  wrap(c,context,'先把参考帧的手/脚放好并确认局部修正，再在接触区间内定位该帧。锁定当前世界位置和朝向；鞋底和掌心随之保持。')
  c.operator('screenshot_pose.video_review',text='以当前帧为接触点 · 生成预览',icon='CONSTRAINT').action='CONTACT_PREVIEW'
  wrap(c,context,'区间内持续锁定，区间外渐入渐出。默认不移动骨盆；目标不可达会报告残差，不保证接触成功。不自动推断桌面或地面。')

class SP_PT_local(SPDetailPanel,Panel):
 bl_label='局部修正';bl_idname='SP_PT_local'
 def draw(self,context):
  s=context.scene.screenshot_pose;c=column(self.layout)
  c.prop(s,'local_scope',text='更新部位')
  row=c.row();row.enabled=bool(s.target and exists(s.json_path) and addon().preview.root_for(s));row.scale_y=1.35
  row.operator('screenshot_pose.apply_local',text='仅更新这个部位',icon='CHECKMARK')
  wrap(c,context,'先校正对应关节或调整头手方向，再局部更新；不重新识别，不执行全身拟合。','INFO')
  wrap(c,context,'手掌更新固定前臂；腿部接地固定骨盆，只求解选定一侧。')

class SP_PT_candidates(SPDetailPanel,Panel):
 bl_label='姿势候选预览';bl_idname='SP_PT_candidates'
 def draw(self,context):
  s=context.scene.screenshot_pose;c=self.layout.column();c.enabled=addon()._JOB is None
  if addon().direct_tools.active(s):
   wrap(c,context,'正在局部拖动，请先确认或取消。');c.operator('screenshot_pose.direct_action',text='取消拖动',icon='LOOP_BACK').action='CANCEL';return
  if s.candidate_state and json.loads(s.candidate_state).get('kind')=='BEFORE_AFTER':
   wrap(c,context,'正在查看应用前姿势，请先返回当前姿势。');c.operator('screenshot_pose.pose_compare',text='返回当前姿势',icon='LOOP_BACK');return
  if addon().hand_preview.active(s):
   wrap(c,context,'正在预览掌向，请到“手部编辑与掌向预览”确认或取消。');return
  active=bool(s.candidate_state);row=c.row();row.enabled=not active and bool(s.target and exists(s.json_path))
  row.operator('screenshot_pose.candidates',text='准备 / 刷新候选',icon='FILE_REFRESH').action='COLLECT'
  if s.chosen_candidate and not active:
   try:wrap(c,context,'已固定：'+json.loads(s.chosen_candidate)['name'],'CHECKMARK')
   except (ValueError,KeyError):pass
   c.operator('screenshot_pose.candidates',text='恢复自动比较',icon='LOOP_BACK').action='AUTO'
  if s.pose_candidates:
   c.template_list('UI_UL_list','sp_candidates',s,'pose_candidates',s,'candidate_index',rows=min(4,len(s.pose_candidates)))
   if 0<=s.candidate_index<len(s.pose_candidates):c.label(text=s.pose_candidates[s.candidate_index].score,icon='INFO')
   row=c.row(align=True)
   row.operator('screenshot_pose.candidates',text='上一个',icon='TRIA_LEFT').action='PREVIOUS'
   row.operator('screenshot_pose.candidates',text='预览',icon='HIDE_OFF').action='PREVIEW'
   row.operator('screenshot_pose.candidates',text='下一个',icon='TRIA_RIGHT').action='NEXT'
  if active:
   if 0<=s.candidate_active_index<len(s.pose_candidates):wrap(c,context,'正在预览：'+s.pose_candidates[s.candidate_active_index].name,'HIDE_OFF')
   row=c.row(align=True);row.scale_y=1.3
   row.operator('screenshot_pose.candidates',text='确认保留',icon='CHECKMARK').action='CONFIRM'
   row.operator('screenshot_pose.candidates',text='取消恢复',icon='LOOP_BACK').action='CANCEL'
  wrap(c,context,'预览直接显示在角色上，不写关键帧；确认时才按上方设置决定是否打帧。','INFO')
  wrap(c,context,'候选来自已有模型结果。二维误差仅供参考，最终由你选择。')

class SP_PT_orientation(SPDetailPanel,Panel):
 bl_label='头部与手势';bl_idname='SP_PT_orientation'
 def draw(self,context):
  s=context.scene.screenshot_pose;c=column(self.layout)
  if s.ui_part=='HEAD':SP_PT_orientation.draw_head(self,context,c)
  for side,title in [('l','左手'),('r','右手')]:
   if s.ui_part not in ('HAND_'+side.upper(),'ARM_'+side.upper()):continue
   SP_PT_orientation.draw_hand(self,context,c,side,title)
  if s.ui_part=='HEAD':return
  c.prop(s,'auto_finger_curl',text='自动还原手指')
  c.prop(s,'hand_semantics',text='采用可靠的自动手势建议')
  sub=c.column();sub.enabled=s.auto_finger_curl;sub.prop(s,'finger_joints',text='逐节弯曲与张指')
  if 'V' in (s.hand_left,s.hand_right):c.prop(s,'hand_spread',text='V 手势张开角')
  if exists(s.hand_refine_report):c.operator('screenshot_pose.hands_report',text='手部识别 / 回退报告',icon='INFO')
  wrap(c,context,'参考图留空使用主图；精修结果不会自动应用。模型准备与选择请到“设置”。')
 def draw_head(self,context,c):
  s=context.scene.screenshot_pose
  box=c.box();box.label(text='头部',icon='ORIENTATION_GIMBAL');box.prop(s,'orient_head',text='使用识别朝向')
  sub=box.column();sub.enabled=s.orient_head
  sub.prop(s,'head_consensus')
  sub.prop(s,'head_gpu')
  row=box.row();row.enabled=exists(s.json_path)
  row.operator('screenshot_pose.details',text='只重新识别头部候选',icon='VIEWZOOM').head_only=True
  try:
   selection=json.loads(s.last_result).get('orientation',{}).get('head_selection',{}) if s.last_result else {}
   if selection:wrap(box,context,'头向采用：'+selection.get('chosen','保留身体头向'))
  except (ValueError,TypeError):pass
  for prop,label in [('head_yaw','左右转头'),('head_pitch','抬头 / 低头'),('head_roll','左右歪头')]:sub.prop(s,prop,text=label)
  row=box.row();row.enabled=bool(s.target and exists(s.json_path));row.operator('screenshot_pose.apply_local',text='仅更新头部',icon='CHECKMARK').scope='HEAD'
 def draw_hand(self,context,c,side,title):
  s=context.scene.screenshot_pose
  box=c.box();box.label(text=title,icon='HAND');box.prop(s,'orient_hand_'+side,text='使用识别掌向')
  if s.contact_enabled:
   box.prop(s,'hand_contact_'+side,text='接触方式')
   if getattr(s,'hand_contact_'+side)!='FREE':
    box.prop(s,'hand_target_'+side,text='掌心目标')
    row=box.row(align=True);row.operator('screenshot_pose.contact_target',text='创建掌心目标').endpoint='HAND_'+side.upper()
    row.operator('screenshot_pose.contact_apply',text='更新此手接触').scope='HAND_'+side.upper()
    if getattr(s,'hand_contact_'+side)=='GRIP':box.prop(s,'hand_grip_'+side,text='握持收拢',slider=True)
    wrap(box,context,'移动目标定位掌心；旋转目标调整掌向。Y 为指向，Z 为掌面法向。')
  sub=box.column();sub.enabled=getattr(s,'orient_hand_'+side)
  sub.prop(s,'palm_flip_'+side,text='翻转掌心');sub.prop(s,'palm_roll_'+side,text='旋转微调');sub.prop(s,'hand_candidate_'+side,text='掌向候选')
  box.prop(s,'hand_left' if side=='l' else 'hand_right',text='手势预设')
  if getattr(s,'hand_left' if side=='l' else 'hand_right')=='GRIP':box.prop(s,'hand_grip_'+side,text='握持收拢',slider=True)
  box.prop(s,'hand_reference_'+side,text='参考图')
  if getattr(s,'hand_reference_'+side):box.prop(s,'hand_person_'+side,text='图中人物序号')
  row=box.row();row.enabled=exists(s.json_path)
  row.operator('screenshot_pose.hands_refine',text='精修'+title+'参考',icon='VIEWZOOM').side=side.upper()
  row=box.row();row.enabled=bool(s.target and exists(s.json_path));row.operator('screenshot_pose.apply_local',text='仅更新'+title,icon='CHECKMARK').scope='HAND_'+side.upper()
class SP_PT_hand_models(SPDetailPanel,Panel):
 bl_label='手部模型与结果选择';bl_idname='SP_PT_hand_models'
 def draw(self,context):
  s=context.scene.screenshot_pose;c=column(self.layout)
  box=c.box();box.label(text='独立三维手部重建',icon='HAND');box.prop(s,'hand3d_backend')
  if s.hand3d_backend!='LOCAL':
   box.prop(s,'hand3d_precision');box.prop(s,'hand_native_roll')
   from .hand3d import readiness
   for backend in ('WILOR','HAMER') if s.hand3d_backend=='COMPARE' else ('HAMER','HAMER_TEXTURE') if s.hand3d_backend=='COMPARE_TEXTURE' else (s.hand3d_backend,):
    state=readiness(addon().runtime_dir(),backend)
    wrap(box,context,backend+'：'+('文件齐全，待推理验证' if state['ready'] else '缺少 '+ '、'.join(state['missing'])),'CHECKMARK' if state['ready'] else 'INFO')
   box.prop(s,'hand3d_mano');box.operator('screenshot_pose.hand3d_setup',icon='IMPORT')
   if s.hand3d_backend in ('HAMER_TEXTURE','COMPARE_TEXTURE'):
    box.prop(s,'hand_texture_checkpoint');box.operator('screenshot_pose.setup_hand_texture',icon='CHECKMARK')
    wrap(box,context,'增强版权重需单独取得并核验；缺少时不会用原版冒充。')
   wrap(box,context,'选择模型后到“修正 → 左手 / 右手”执行精修。对比保留各模型结果，需手动选择。')
  if exists(s.hand3d_comparison):
   hand_choices(box,s)
  c.prop(s,'hand_semantics',text='采用可靠的自动手势建议')
  c.prop(s,'hand_gpu_2d')
  if s.hand_gpu_2d and not (addon().runtime_dir()/'rtmpose_hand.onnx').is_file():c.operator('screenshot_pose.setup_hand_gpu',text='准备手部 GPU 模型（首次联网）',icon='IMPORT')
  sub=c.column();sub.enabled=s.auto_finger_curl;sub.prop(s,'finger_joints',text='逐节弯曲与张指')
  if 'V' in (s.hand_left,s.hand_right):c.prop(s,'hand_spread',text='V 手势张开角')
  wrap(c,context,'参考图留空使用主图；其他视角须同角色、同姿势，并需要 SAM 完整骨架对齐。精修不会自动应用。')
  wrap(c,context,'手势约束优先于自动手指，当前支持 ARP。握持为近似姿势，不模拟物体接触。','INFO')
  if exists(s.hand_refine_report):c.operator('screenshot_pose.hands_report',text='查看手部识别 / 回退报告',icon='INFO')
  row=c.row(align=True);row.operator('screenshot_pose.reset_orientation',text='重置方向微调',icon='LOOP_BACK')
  row= c.row();row.enabled=exists(s.json_path) and all((addon().runtime_dir()/n).is_file() for n in ('face_landmarker.task','hand_landmarker.task'))
  row.operator('screenshot_pose.details',text='重新识别头部与手掌',icon='VIEWZOOM')
  wrap(c,context,'模型结果载入后，到对应部位执行局部更新。')

class SP_PT_feet(SPDetailPanel,Panel):
 bl_label='脚部与接地';bl_idname='SP_PT_feet'
 def draw(self,context):
  s=context.scene.screenshot_pose;c=column(self.layout)
  foot_active=bool(s.candidate_state and json.loads(s.candidate_state).get('kind')=='FOOT_DIRECTION')
  if foot_active:
   c.enabled=addon()._JOB is None
   wrap(c,context,'正在预览脚向；保持支撑设置不变，确认或取消后继续。')
   for action,label in [('NEXT','下一个脚向'),('CONFIRM','确认脚向'),('CANCEL','取消并恢复')]:c.operator('screenshot_pose.foot_preview',text=label).action=action
   return
  box=c.box();box.prop(s,'foot_side');box.operator('screenshot_pose.foot_refine',text='两尺度局部复核双脚')
  row=box.row(align=True);row.enabled=bool(s.target and exists(s.json_path))
  row.operator('screenshot_pose.foot_preview',text='预览脚向').action='PREVIEW'
  row.operator('screenshot_pose.foot_preview',text='识别 / 角色脚向诊断').action='DIAGRAM'
  box.operator('screenshot_pose.foot_preview',text='恢复此脚 SAM 自动朝向').action='AUTO'
  box.operator('screenshot_pose.foot_preview',text='隐藏脚向诊断').action='HIDE'
  wrap(box,context,'局部复核只细化二维脚点，深度沿用原估计；候选需人工判断。')
  if s.foot_refine_report:
   for side,info in json.loads(s.foot_refine_report).items():wrap(box,context,side+'脚：'+('采用局部观测' if info['adopted'] else info.get('reason','保留原结果')))
  c.prop(s,'contact_enabled',text='独立手脚接触')
  c.prop(s,'foot_pose',text='使用三维脚向与鞋底参考')
  if s.contact_enabled:
   wrap(c,context,'接触目标在图片拟合后生效，需要姿势强度 100%。','INFO')
   for side,title in [('l','左脚'),('r','右脚')]:
    box=c.box();box.label(text=title);box.prop(s,'support_'+side,text='支撑方式')
    if getattr(s,'support_'+side)!='FREE':
     box.prop(s,'support_target_'+side,text='目标物体（可选）')
     if not getattr(s,'support_target_'+side):box.prop(s,'floor_height' if getattr(s,'support_'+side)=='FLOOR' else 'support_height_'+side,text='支撑面高度')
     box.prop(s,'knee_target_'+side,text='膝盖朝向（可选）')
     row=box.row(align=True);row.operator('screenshot_pose.contact_target',text='创建支撑目标').endpoint='FOOT_'+side.upper()
     row.operator('screenshot_pose.contact_apply',text='更新此脚').scope='FOOT_'+side.upper()
   c.prop(s,'support_move_pelvis')
   if s.support_move_pelvis:c.prop(s,'support_pelvis_limit')
   c.operator('screenshot_pose.contact_apply',text='更新全部接触').scope='ALL'
   wrap(c,context,'在“头部与手势”设置扶面或握物；手动编辑过的手指会保留。')
  else:c.prop(s,'ground_feet',text='接地约束')
  if s.ground_feet and not s.contact_enabled:
   c.prop(s,'ground_mode');wrap(c,context,'抬腿 / 踩高处时，仅选择地面上的承重脚。')
   wrap(c,context,'适合站立、深蹲；坐姿和腾空动作请关闭。','INFO')
   c.prop(s,'floor_height',text='地面高度');c.prop(s,'pelvis_offset',text='骨盆高度微调');c.prop(s,'flatten_feet',text='保持脚部水平朝向')
  if s.foot_pose:c.prop(s,'shoe_mesh',text='鞋子网格')
  row=c.row(align=True);row.enabled=bool(s.target and exists(s.json_path))
  row.operator('screenshot_pose.apply_local',text='仅更新左腿').scope='LEG_L';row.operator('screenshot_pose.apply_local',text='仅更新右腿').scope='LEG_R'

class SP_PT_fit(SPDetailPanel,Panel):
 bl_label='身体与原图贴合';bl_idname='SP_PT_fit'
 def draw(self,context):
  s=context.scene.screenshot_pose;c=column(self.layout)
  c.prop(s,'fit_projection',text='贴合原图并匹配相机');c.prop(s,'full_source_rotation',text='保留完整源骨架旋转')
  c.prop(s,'preserve_source_pose')
  if s.preserve_source_pose:wrap(c,context,'完整骨架优先保留源深度与躯干。全身协调用于指定的支撑目标；手动关节点或关闭保护时沿用自由拟合。')
  c.prop(s,'fit_torso')
  c.prop(s,'whole_body')
  if s.whole_body:
   box=c.box();box.prop(s,'whole_body_root_limit');box.prop(s,'whole_body_rotation_limit')
   wrap(box,context,'固定明确的手脚目标，联合调整骨盆与四肢；实际结果无明显改善时保留原姿势。')
   if s.contact_enabled and not s.support_move_pelvis:wrap(box,context,'当前支撑设置固定骨盆位置，仍可协调躯干和四肢。')
   try:
    whole=json.loads(s.last_result).get('fit',{}).get('whole_body',{}) if s.last_result else {}
    if whole:
     wrap(box,context,'全身协调：'+('已采用' if whole.get('accepted') else '保留此前结果'))
     if whole.get('accepted'):wrap(box,context,'骨盆各轴位移：'+', '.join(f'{v:.3f}' for v in whole['root_delta']))
   except (ValueError,TypeError):pass
  camera=c.box();camera.label(text='相机标定与透视（试验）',icon='CAMERA_DATA')
  camera.operator('screenshot_pose.setup_camera',text='准备相机标定模型（首次联网）',icon='IMPORT')
  row=camera.row();row.enabled=exists(s.json_path) and (addon().runtime_dir()/'camera/config.json').is_file()
  row.operator('screenshot_pose.calibrate_camera',text='标定相机用于透视拟合',icon='VIEWZOOM')
  camera.prop(s,'camera_refresh_body')
  row=camera.row();row.enabled=exists(s.json_path)
  row.operator('screenshot_pose.camera_before',text='切回标定前观测',icon='LOOP_BACK')
  camera.prop(s,'fit_model')
  wrap(camera,context,'标定后需手动应用；原有近似可随时切回。相机内参是估计值，纯色背景及夸张透视需复核。')
  if s.full_source_rotation:
   sub=c.column();sub.enabled=not s.preserve_source_pose;sub.prop(s,'auto_full_quality',text='比较图像贴合与关节旋转')
   if s.preserve_source_pose:wrap(c,context,'当前保留完整源姿势；其他方案可到候选预览中手动选择。')
  c.prop(s,'auto_depth_select',text='比较三维深度候选')
  wrap(c,context,'自动比较可能选择其他模型的身体深度，可在应用结果中查看。','INFO')
  if s.fit_projection:
   sub=c.column();sub.enabled=s.ground_feet;sub.prop(s,'coordinate_pelvis')
   if not s.ground_feet:wrap(c,context,'骨盆协调需要同时开启双脚接地。')
   c.prop(s,'fit_regularization',text='保留初始姿态')
   try:
    fit=json.loads(s.last_result).get('fit') if s.last_result else None
    if fit:
     c.label(text=f"投影误差 {fit['before_px']:.1f} → {fit['after_px']:.1f} px")
     if fit.get('torso'):wrap(c,context,'躯干修正：'+('已采用' if fit['torso']['accepted'] else '保留原姿势'))
   except (ValueError,KeyError,TypeError):pass
   row=c.row();row.enabled=bool(s.last_result);row.operator('screenshot_pose.match_camera',text='创建匹配相机',icon='CAMERA_DATA')
  if s.camera_backup:c.operator('screenshot_pose.restore_camera',text='恢复原相机',icon='LOOP_BACK')

class SP_PT_tools(SPDetailPanel,Panel):
 bl_label='关节编辑与数据';bl_idname='SP_PT_tools'
 def draw(self,context):
  s=context.scene.screenshot_pose;c=column(self.layout);has_pose=exists(s.json_path)
  row=c.row(align=True);row.operator('screenshot_pose.import_json',text='导入姿势',icon='IMPORT');r=row.row();r.enabled=has_pose;r.operator('screenshot_pose.export_json',text='导出姿势',icon='EXPORT')
  c.operator('screenshot_pose.template',text='创建手动关节点',icon='EDITMODE_HLT')
  sub=c.column();sub.enabled=has_pose
  sub.operator('screenshot_pose.edit_image',text='在原图上校正关节',icon='GREASEPENCIL');sub.operator('screenshot_pose.rebuild',text='按参数重建关节点',icon='FILE_REFRESH')
  sub=c.column();sub.enabled=bool(addon().preview.root_for(s))
  row=sub.row(align=True);row.operator('screenshot_pose.focus',text='编辑三维关节',icon='PIVOT_MEDIAN');row.operator('screenshot_pose.refresh',text='刷新连线',icon='FILE_REFRESH')
  sub.prop(s,'show_names',text='显示关节名称');sub.operator('screenshot_pose.create_source',text='生成独立源骨架',icon='ARMATURE_DATA')
  wrap(c,context,'三维视图：G 移动关节，G Y 调整深度。')

class SP_PT_mapping(SPDetailPanel,Panel):
 bl_label='角色骨骼映射';bl_idname='SP_PT_mapping'
 def draw(self,context):
  a=addon();s=context.scene.screenshot_pose;c=column(self.layout);c.enabled=c.enabled and bool(s.target)
  c.operator('screenshot_pose.auto_map',text='重新自动匹配',icon='FILE_REFRESH');c.prop(s,'switch_fk',text='自动切换到 FK 控制')
  if s.target:
   for item in s.mapping:
    row=c.row(align=True);row.prop(item,'enabled',text='');row.prop_search(item,'bone',s.target.data,'bones',text=a.label(item.role))
  else:c.label(text='请先选择角色骨架',icon='INFO')

class SP_PT_advanced(SPDetailPanel,Panel):
 bl_label='识别与求解参数';bl_idname='SP_PT_advanced'
 def draw(self,context):
  s=context.scene.screenshot_pose;c=column(self.layout)
  c.label(text='识别与重建',icon='SETTINGS')
  c.prop(s,'fresh_pose_start')
  for p in ('person_index','confidence','threshold','coordinate_mode','depth','mirror','yaw','fit_iterations'):c.prop(s,p)
  c.separator();c.label(text='朝向求解',icon='ORIENTATION_GIMBAL')
  for p in ('head_strength','hand_strength','neck_share','forearm_share','orientation_threshold'):c.prop(s,p)
  c.separator();c.prop(s,'foot_threshold');c.prop(s,'ankle_clearance')
  if s.target:c.prop_search(s,'root_bone',s.target.data,'bones')

class SP_PT_character(SPDetailPanel,Panel):
 bl_label='角色校准与诊断';bl_idname='SP_PT_character'
 def draw(self,context):
  a=addon();s=context.scene.screenshot_pose;c=column(self.layout);c.enabled=c.enabled and bool(s.target)
  c.label(text='01  核对控制骨',icon='ARMATURE_DATA')
  wrap(c,context,'使用角色骨骼映射中的 FK 控制骨。读取静置骨架，无需把角色摆成 T 姿势。')
  c.label(text='02  校准角色方向',icon='ORIENTATION_GIMBAL');c.prop(s,'head_forward')
  if s.target:
   for side,title in [('l','左手'),('r','右手')]:
    box=c.box();box.label(text=title)
    for p in ('index_'+side,'pinky_'+side):box.prop_search(s,p,s.target.data,'bones')
    box.prop(s,'rest_palm_flip_'+side)
   c.prop_search(s,'shoe_mesh',context.scene,'objects',text='鞋子网格')
   c.prop_search(s,'root_bone',s.target.data,'bones')
  c.prop(s,'wrist_limit');wrap(c,context,'食指 / 小指需指定根骨；鞋子留空自动查找。掌面反转只校准角色，不改变原图。')
  c.label(text='03  保存角色配置',icon='FILE_TICK')
  row=c.row(align=True)
  row.operator('screenshot_pose.character_profile',text='保存 / 更新',icon='CHECKMARK').action='SAVE'
  row.operator('screenshot_pose.character_profile',text='读取',icon='FILE_REFRESH').action='LOAD'
  row=c.row(align=True);row.operator('screenshot_pose.profile_export',text='导出配置',icon='EXPORT');row.operator('screenshot_pose.profile_import',text='导入配置',icon='IMPORT')
  c.operator('screenshot_pose.character_profile',text='恢复默认校准参数',icon='LOOP_BACK').action='RESET'
  wrap(c,context,s.profile_status,'INFO');wrap(c,context,'配置随此骨架保存。请保存 .blend；切换角色时自动读取其配置，不保存截图手势微调。')
  c.separator();c.operator('screenshot_pose.character_profile',text='检查识别、映射与约束',icon='VIEWZOOM').action='DIAGNOSE'
  if s.diagnosis:
   try:
    report=json.loads(s.diagnosis)
    wrap(c,context,'以下为上次检查快照；修改参数或姿势后请重新检查。','INFO')
    for item in report['rows']:
     box=c.box();box.label(text=item['part']+' · '+item['category'])
     wrap(box,context,item['message'])
     if item.get('fix'):
      op=box.operator('screenshot_pose.diagnosis_focus',text='打开对应修正工具',icon='FORWARD');op.scope=item['scope'];op.fix=item['fix']
   except (ValueError,KeyError,TypeError):wrap(c,context,'请重新检查诊断。')
  try:comparing=bool(s.candidate_state and json.loads(s.candidate_state).get('kind')=='BEFORE_AFTER')
  except (ValueError,TypeError):comparing=False
  row=self.layout.row();row.enabled=not a._JOB and bool(s.target) and (comparing or (not s.candidate_state and bool(s.target.get('screenshot_pose_backup'))))
  row.operator('screenshot_pose.pose_compare',text='返回当前姿势' if comparing else '查看应用前姿势',icon='LOOP_BACK' if comparing else 'HIDE_OFF')
  wrap(self.layout,context,'前后对比保持当前视角，不写关键帧；返回后完整恢复当前姿势。')

class SP_PT_direct(SPDetailPanel,Panel):
 bl_label='叠图与直接拖动';bl_idname='SP_PT_direct'
 def draw(self,context):
  a=addon();s=context.scene.screenshot_pose;c=column(self.layout);editing=a.direct_tools.active(s)
  row=c.row();row.enabled=bool(s.target and s.last_result);row.operator('screenshot_pose.reference_overlay',text='显示原图 / 进入匹配相机',icon='IMAGE_DATA').action='SHOW'
  row=self.layout.row();row.enabled=not a._JOB;row.prop(s,'reference_opacity',slider=True)
  row=self.layout.row();row.enabled=not a._JOB and a.direct_tools.overlay_visible(s);row.operator('screenshot_pose.reference_overlay',text='隐藏叠图',icon='HIDE_ON').action='HIDE'
  wrap(self.layout,context,'先贴合原图并应用；叠图不进入渲染。')
  c.prop(s,'direct_endpoint');sub=c.column();sub.enabled=s.direct_endpoint.startswith('ankle');sub.prop(s,'direct_lock_height')
  row=c.row();row.enabled=bool(s.target and s.last_result and a.preview.root_for(s));row.operator_context='INVOKE_DEFAULT';row.operator('screenshot_pose.direct_drag',text='开始拖动所选手腕 / 脚踝',icon='ORIENTATION_CURSOR')
  if editing:
   box=self.layout.box();wrap(box,context,'拖动亮色圆点；固定肩 / 骨盆和其他肢体，保留手掌 / 脚的朝向。')
   info=box.row();info.enabled=False;info.prop(s,'direct_residual')
   box.operator('screenshot_pose.direct_action',text='对比拖动前后（B）',icon='HIDE_OFF').action='COMPARE'
   row=box.row(align=True)
   row.operator('screenshot_pose.direct_action',text='确认保留',icon='CHECKMARK').action='CONFIRM';row.operator('screenshot_pose.direct_action',text='取消恢复',icon='LOOP_BACK').action='CANCEL'
  row=c.row();row.enabled=bool(s.direct_undo_state and s.target);row.operator('screenshot_pose.direct_action',text='撤销上次已确认拖动',icon='LOOP_BACK').action='UNDO'
  wrap(self.layout,context,'B 对比 · Enter 确认 · Esc / 右键取消。')
  wrap(self.layout,context,'确认同步关节点并沿用打帧开关。脚踝未锁定高度时可能离地。')

class SP_PT_models(SPDetailPanel,Panel):
 bl_label='模型与环境';bl_idname='SP_PT_models'
 def draw(self,context):
  a=addon();s=context.scene.screenshot_pose;c=column(self.layout)
  c.label(text='当前模型已就绪' if ready(s) else '当前模型尚未就绪',icon='CHECKMARK' if ready(s) else 'ERROR')
  c.operator('screenshot_pose.setup',text='准备基础识别环境',icon='IMPORT')
  c.operator('screenshot_pose.setup_high',text='准备 RTMW 高精度模型',icon='IMPORT')
  c.operator('screenshot_pose.setup_gpu',text='准备 RTMW3D GPU 环境',icon='IMPORT')
  c.operator('screenshot_pose.setup_details',text='准备头部与手部模型',icon='IMPORT')
  c.operator('screenshot_pose.setup_head3d',text='准备独立头向 GPU 模型（可选）',icon='IMPORT')
  wrap(c,context,'首次准备需要联网下载；识别在本机完成。SAM 的运行器与权重目录在插件偏好设置中配置。','INFO')
  if s.last_log:c.operator('screenshot_pose.open_log',text='打开最近任务日志',icon='TEXT')

class SP_PT_multiview(SPDetailPanel,Panel):
 bl_label='多图辅助（可选）';bl_idname='SP_PT_multiview'
 def draw(self,context):
  s=context.scene.screenshot_pose;c=column(self.layout)
  c.prop(s,'multiview_enabled')
  if not s.multiview_enabled:
   wrap(c,context,'同一角色、同一姿势的其他视角，可生成额外候选。默认关闭。');return
  wrap(c,context,'先用 SAM 三维识别主图，再添加最多 3 张辅助图；请勿混用不同动作。','INFO')
  for i,item in enumerate(s.reference_images):
   box=c.box();row=box.row(align=True);row.prop(item,'enabled',text='');row.prop(item,'file_path',text='')
   row.operator('screenshot_pose.reference_remove',text='',icon='X').index=i
   box.prop(item,'person_index',text='图中人物序号')
  row=c.row();row.enabled=len(s.reference_images)<3
  row.operator('screenshot_pose.reference_add',text='添加辅助视角',icon='ADD')
  c.prop(s,'multiview_strength',slider=True)
  row=c.row();row.enabled=exists(s.json_path) and any(r.enabled for r in s.reference_images)
  row.operator('screenshot_pose.multiview',text='生成多图辅助候选',icon='RENDER_STILL')
  if exists(s.multiview_report):c.operator('screenshot_pose.multiview_report',text='查看采用 / 筛除报告',icon='INFO')
  wrap(c,context,'生成后到“姿势候选预览”比较，不自动应用。无法可靠对齐的图片会被筛除；头手局部识别沿用主图。')

class SP_PT_hand_edit(SPDetailPanel,Panel):
 bl_label='手部编辑与掌向预览';bl_idname='SP_PT_hand_edit'
 def draw(self,context):
  a=addon();s=context.scene.screenshot_pose;c=self.layout.column();c.enabled=a._JOB is None
  active=a.hand_preview.active(s)
  if s.candidate_state and not active:
   wrap(c,context,'请先结束全身姿势候选预览。');return
  edit=c.column();edit.enabled=not active
  edit.prop(s,'hand_edit_side',text='对应手')
  row=edit.row();row.enabled=exists(s.json_path)
  row.operator('screenshot_pose.hand_editor',text='放大校正手部 21 点',icon='GREASEPENCIL')
  row=edit.row();row.enabled=exists(s.json_path) and a.hand_preview.can_undo(s);row.operator('screenshot_pose.hand_edit_undo',text='撤回上次此手校正数据',icon='LOOP_BACK')
  wrap(edit,context,'拖动指节、指尖，手腕固定；可调前后深度。保存时可只更新改动的手指。')
  row=edit.row();row.enabled=bool(s.target and exists(s.json_path));row.operator('screenshot_pose.palm_preview',text='准备掌向候选',icon='FILE_REFRESH').action='COLLECT'
  if s.hand_palm_options and s.hand_palm_side==s.hand_edit_side:
   c.template_list('UI_UL_list','sp_palms',s,'hand_palm_options',s,'hand_palm_index',rows=min(4,len(s.hand_palm_options)))
   row=c.row(align=True);row.operator('screenshot_pose.palm_preview',text='上一个',icon='TRIA_LEFT').action='PREVIOUS';row.operator('screenshot_pose.palm_preview',text='临时预览',icon='HIDE_OFF').action='PREVIEW';row.operator('screenshot_pose.palm_preview',text='下一个',icon='TRIA_RIGHT').action='NEXT'
  if active:
   row=c.row(align=True);row.scale_y=1.3;row.operator('screenshot_pose.palm_preview',text='确认保留',icon='CHECKMARK').action='CONFIRM';row.operator('screenshot_pose.palm_preview',text='取消恢复',icon='LOOP_BACK').action='CANCEL'
   wrap(c,context,'临时预览不写关键帧；确认时沿用主界面打帧选项。')
  else:wrap(c,context,'掌向候选含原识别、已有几何解及掌背翻转。它们是比较选项，不是新的识别结果。')

class SP_PT_hand_choice(SPDetailPanel,Panel):
 bl_label='选择此手的模型结果';bl_idname='SP_PT_hand_choice'
 def draw(self,context):hand_choices(column(self.layout),context.scene.screenshot_pose)

_SECTIONS={
 'SP_PT_issues':('GENERATE','CORRECT'),
 'SP_PT_library':('LIBRARY',),
 **{name:('VIDEO',) for name in ('SP_PT_video','SP_PT_video_review','SP_PT_video_contacts')},
 **{name:('ADVANCED',) for name in ('SP_PT_character','SP_PT_mapping','SP_PT_advanced','SP_PT_models','SP_PT_tools','SP_PT_hand_models')},
}
_PART_FILTER={'SP_PT_orientation':('HEAD','HAND_L','HAND_R','ARM_L','ARM_R'),'SP_PT_hand_edit':('HAND_L','HAND_R','ARM_L','ARM_R'),
 'SP_PT_feet':('LEG_L','LEG_R'),'SP_PT_fit':('BODY',),'SP_PT_candidates':('BODY',),'SP_PT_multiview':('BODY',),
 'SP_PT_direct':('BODY','ARM_L','ARM_R','LEG_L','LEG_R'),'SP_PT_local':('HEAD','HAND_L','HAND_R','ARM_L','ARM_R','LEG_L','LEG_R'),
 'SP_PT_hand_choice':('HAND_L','HAND_R','ARM_L','ARM_R')}
PANELS=(SP_PT_main,SP_PT_issues,SP_PT_video,SP_PT_video_review,SP_PT_video_contacts,SP_PT_library,SP_PT_direct,SP_PT_local,SP_PT_character,SP_PT_candidates,SP_PT_multiview,SP_PT_orientation,SP_PT_hand_edit,SP_PT_hand_choice,SP_PT_feet,SP_PT_fit,SP_PT_tools,SP_PT_mapping,SP_PT_advanced,SP_PT_models,SP_PT_hand_models)
