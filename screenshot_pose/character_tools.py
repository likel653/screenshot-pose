"""Per-object calibration profiles and non-mutating pose diagnostics."""
import hashlib,importlib,json,math
from pathlib import Path
import bpy
from bpy.props import EnumProperty,StringProperty
from bpy.types import Operator
from bpy_extras.io_utils import ImportHelper,ExportHelper

KEY='sp_character_profile'
FIELDS=('head_forward','index_l','pinky_l','index_r','pinky_r','rest_palm_flip_l','rest_palm_flip_r',
        'wrist_limit','neck_share','forearm_share','root_bone','ankle_clearance','switch_fk')
def addon():return importlib.import_module(__package__)
def mapping(s):return {r.role:r.bone for r in s.mapping if r.enabled and r.bone}
def fingerprint(rig):
 values=[(b.name,b.parent.name if b.parent else '',[[round(v,6) for v in row] for row in b.matrix_local],round(b.length,6)) for b in rig.data.bones]
 return hashlib.sha256(json.dumps(sorted(values),sort_keys=True).encode()).hexdigest()
def reset(s):
 for name in FIELDS:setattr(s,name,s.bl_rna.properties[name].default)
 s.shoe_mesh=None;s.profile_status='尚未保存此角色配置';s.diagnosis=''
def read(rig):
 if not rig or not rig.get(KEY):return None
 doc=json.loads(rig[KEY])
 if not isinstance(doc,dict) or doc.get('schema')!='screenshot_pose/character_profile_v1':raise ValueError('角色配置版本不兼容')
 return doc
def validate(s,doc):
 if not isinstance(doc,dict):raise ValueError('角色配置必须是 JSON 对象')
 if doc.get('schema')!='screenshot_pose/character_profile_v1':raise ValueError('角色配置格式不兼容')
 if fingerprint(s.target)!=doc.get('rig_signature'):raise ValueError('骨架名称、层级或静置姿态已改变，请重新校准并保存')
 rows=doc.get('mapping');params=doc.get('settings')
 if not isinstance(rows,list) or not isinstance(params,dict):raise ValueError('角色配置缺少映射或参数')
 roles=set();names=[]
 for row in rows:
  if not isinstance(row,dict):raise ValueError('角色映射格式不正确')
  role=row.get('role');bone=row.get('bone');enabled=row.get('enabled')
  if role not in addon().retarget.ROLE_KEYS or role in roles or not isinstance(enabled,bool):raise ValueError('角色配置包含无效或重复部位')
  roles.add(role)
  if not isinstance(bone,str) or (bone and bone not in s.target.pose.bones):raise ValueError('角色配置中的骨骼已不存在')
  if enabled and bone:names.append(bone)
 if len(names)!=len(set(names)):raise ValueError('角色配置包含重复映射')
 if roles!=set(addon().retarget.ROLE_KEYS):raise ValueError('角色配置部位不完整')
 for name in FIELDS:
  if name not in params:raise ValueError('角色配置缺少参数：'+name)
  p=s.bl_rna.properties[name];value=params[name]
  if p.type=='BOOLEAN' and not isinstance(value,bool):raise ValueError('角色配置参数类型不正确')
  if p.type=='STRING' and (not isinstance(value,str) or (value and value not in s.target.pose.bones)):raise ValueError('角色配置参考骨不存在')
  if p.type=='ENUM' and value not in p.enum_items.keys():raise ValueError('角色方向无效')
  if p.type=='FLOAT' and (type(value) not in (int,float) or not math.isfinite(value) or not p.hard_min<=value<=p.hard_max):raise ValueError('角色参数超出范围')
 return doc
def capture(s):
 rig=s.target
 if not rig:raise ValueError('请先选择角色骨架')
 if rig.library and not rig.override_library:raise ValueError('链接骨架只读，请先创建 Library Override')
 doc=dict(schema='screenshot_pose/character_profile_v1',target=rig.name,rig_signature=fingerprint(rig),
          mapping=[dict(role=r.role,bone=r.bone,enabled=r.enabled) for r in s.mapping],
          settings={n:getattr(s,n) for n in FIELDS},shoe_mesh=s.shoe_mesh.name if s.shoe_mesh else '')
 validate(s,doc)
 doc['orientation']=addon().orientation.calibration(rig,mapping(s),s)
 return doc
def save(s):
 doc=capture(s);s.target[KEY]=json.dumps(doc,ensure_ascii=False)
 s.target['sp_orientation_calibration']=json.dumps(doc['orientation'],ensure_ascii=False)
 s.profile_status='角色配置已保存；保存 .blend 后可长期保留'
 return doc
def load(s,doc=None):
 doc=doc or read(s.target)
 if not doc:raise ValueError('此角色还没有保存的配置')
 validate(s,doc) # Validate the entire document before changing any settings.
 s.mapping.clear()
 for row in doc['mapping']:
  r=s.mapping.add();r.role=row['role'];r.bone=row['bone'];r.enabled=row['enabled']
 for name in FIELDS:setattr(s,name,doc['settings'][name])
 shoe=bpy.data.objects.get(doc.get('shoe_mesh',''));s.shoe_mesh=shoe if shoe and shoe.type=='MESH' else None
 s.profile_status='已读取此角色配置';s.diagnosis=''
 return doc
def changed(s):
 reset(s)
 if s.target:
  try:
   if read(s.target):load(s)
  except (ValueError,KeyError,TypeError) as exc:s.profile_status='配置需更新：'+str(exc)
def token(s):
 payload=dict(target=s.target.name if s.target else '',signature=fingerprint(s.target) if s.target else '',
              mapping=[(r.role,r.bone,r.enabled) for r in s.mapping],settings={n:getattr(s,n) for n in FIELDS},
              path=s.json_path,threshold=s.orientation_threshold,last=s.last_result,
              switches=[getattr(s,n) for n in ('orient_head','orient_hand_l','orient_hand_r','finger_joints','foot_pose','ground_feet')])
 try:p=Path(bpy.path.abspath(s.json_path));payload['file']=(p.stat().st_size,p.stat().st_mtime_ns)
 except OSError:pass
 return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
def diagnose(s):
 a=addon();rig=s.target
 if not rig:raise ValueError('请先选择角色骨架')
 m=mapping(s);cal=a.orientation.calibration(rig,m,s);rows=[]
 def add(part,category,message,scope='',fix=''):
  rows.append(dict(part=part,category=category,message=message,scope=scope,fix=fix))
 try:
  doc=read(rig)
  if doc:validate(s,doc);add('角色配置','配置','已保存；当前修改需再次保存')
  else:add('角色配置','配置','尚未保存，可先校准方向并保存')
 except (ValueError,TypeError,KeyError) as exc:add('角色配置','配置',str(exc))
 duplicate={n for n in m.values() if list(m.values()).count(n)>1}
 for role in a.retarget.ROLE_KEYS:
  name=m.get(role);bone=rig.pose.bones.get(name or '')
  if not bone:add(a.label(role),'映射','未启用或未找到控制骨',fix='MAPPING')
  elif name in duplicate:add(a.label(role),'映射','同一控制骨映射了多个部位：'+name,fix='MAPPING')
  elif not a.orientation._safe(bone,rig):add(a.label(role),'约束','旋转受约束或驱动，直接旋转可能被覆盖',fix='MAPPING')
 data={}
 try:data=a.pose_data.read_pose(bpy.path.abspath(s.json_path))
 except (OSError,ValueError,KeyError,TypeError):add('识别数据','识别缺失','没有可读取的姿势数据；先识别或导入')
 orient=data.get('orientation',{})
 for part,obs,enabled,scope in [('头部',orient.get('head'),s.orient_head,'HEAD')]+[(side+'手掌',orient.get('hands',{}).get(side),getattr(s,'orient_hand_'+side.lower()),'HAND_'+side) for side in ('L','R')]:
  if not obs:add(part,'识别缺失','没有局部朝向；不会凭空恢复遮挡信息',scope,'HAND' if scope.startswith('HAND') else 'LOCAL')
  elif obs.get('source')=='manual_hand_landmarks' or obs.get('palm_choice') or obs.get('manual_fingers'):
   add(part,'人工修正','包含人工校正；置信度不代表自动识别准确率',scope,'HAND' if scope.startswith('HAND') else 'LOCAL')
  elif obs.get('confidence',0)<s.orientation_threshold:add(part,'识别质量','低于质量下限，自动局部朝向会保留原姿势',scope,'HAND' if scope.startswith('HAND') else 'LOCAL')
  else:add(part,'可用观察','局部朝向达到质量下限；仍需目视确认',scope,'HAND' if scope.startswith('HAND') else 'LOCAL')
  if not enabled:add(part,'应用设置','当前未启用局部朝向',scope,'LOCAL')
 for side in ('L','R'):
  scope='HAND_'+side;hd=orient.get('hands',{}).get(side,{})
  if side not in cal['hands']:add(side+'手掌','校准','缺少食指 / 小指根骨，无法确定静置掌面',scope,'CALIBRATION')
  world=hd.get('world')
  if not isinstance(world,list) or len(world)!=21:add(side+'手指','识别缺失','没有 21 点三维手部；可用手势预设或手工编辑',scope,'HAND')
  blocked=[]
  for f in a.finger_pose.FINGERS:
   for i in (1,2,3):
    b=a.finger_pose.control(rig,f,i,side)
    if not a.finger_pose.writable(b,rig,f,side):blocked.append(f+str(i))
  if blocked:add(side+'手指','映射 / 约束',f'{len(blocked)} / 15 节缺失或不可直接写入',scope,'MAPPING')
  if a.hand_presets.selected(s,data,side)!='NONE':add(side+'手指','人工约束','已指定手势预设；不代表模型识别结果',scope,'HAND')
  fd=orient.get('feet',{}).get(side)
  add(side+'脚','可用观察' if fd and fd.get('confidence',0)>=s.foot_threshold else '识别缺失','脚部朝向达到质量下限' if fd and fd.get('confidence',0)>=s.foot_threshold else '缺少可靠脚向；接地只能补充地面约束','LEG_'+side,'LOCAL')
 if s.foot_pose or s.ground_feet:
  soles=a.foot_pose.sole_calibration(rig,m,s.shoe_mesh.name if s.shoe_mesh else '')
  for side in ('L','R'):
   add(side+'鞋底','校准',f"已从网格读取 {soles[side]['samples']} 个参考点" if side in soles else '未找到鞋底参考；请指定绑定此角色的鞋子网格','LEG_'+side,'CALIBRATION')
 result=a.pose_tools.last_result(s)
 for section in ('orientation','fingers','feet','local_update'):
  report=result.get(section,{})
  if isinstance(report,dict):
   for warning in report.get('warnings',[]):add('上次应用','应用记录',str(warning))
 animation=rig.animation_data
 if animation and (animation.action or any(not t.mute for t in animation.nla_tracks)):
  add('动画','动画覆盖','角色已有动作；换帧时未打关键帧的临时姿势可能被原动画覆盖')
 return dict(schema='screenshot_pose/diagnosis_v1',target=rig.name,token=token(s),rows=rows)

def guard(s):
 if addon()._JOB or s.candidate_state:raise ValueError('请先结束任务或预览')
 if not s.target:raise ValueError('请先选择角色骨架')

class SP_OT_character_profile(Operator):
 bl_idname='screenshot_pose.character_profile';bl_label='角色校准配置'
 action:EnumProperty(items=[('SAVE','保存',''),('LOAD','读取',''),('RESET','恢复默认参数',''),('DIAGNOSE','诊断','')])
 def execute(self,context):
  try:
   s=context.scene.screenshot_pose;guard(s)
   if self.action=='SAVE':save(s)
   elif self.action=='LOAD':load(s)
   elif self.action=='RESET':reset(s);addon().set_mapping(s);s.profile_status='已恢复默认参数；磁盘中的配置仍可读取'
   else:s.diagnosis=json.dumps(diagnose(s),ensure_ascii=False)
   s.status='诊断已更新；诊断不会修改人物姿势' if self.action=='DIAGNOSE' else s.profile_status
  except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}
  return {'FINISHED'}

class SP_OT_diagnosis_focus(Operator):
 bl_idname='screenshot_pose.diagnosis_focus';bl_label='定位修正部位'
 scope:StringProperty();fix:StringProperty()
 def execute(self,context):
  s=context.scene.screenshot_pose
  s.ui_section='ADVANCED' if self.fix in ('MAPPING','CALIBRATION') else 'CORRECT'
  if self.scope in addon().review_tools.review_data.LABELS:s.ui_part=self.scope
  if self.scope:s.local_scope=self.scope
  if self.scope.startswith('HAND_'):s.hand_edit_side=self.scope[-1]
  s.status='已选择对应局部；展开“'+{'HAND':'手部编辑与掌向预览','MAPPING':'角色骨骼映射','CALIBRATION':'角色校准与诊断','LOCAL':'局部修正'}.get(self.fix,'局部修正')+'”继续修正'
  panel={'HAND':'SP_PT_hand_edit','MAPPING':'SP_PT_mapping','CALIBRATION':'SP_PT_character','LOCAL':'SP_PT_local'}.get(self.fix)
  if panel and context.area and context.area.type=='VIEW_3D':bpy.ops.wm.call_panel(name=panel,keep_open=True)
  return {'FINISHED'}

class SP_OT_profile_export(Operator,ExportHelper):
 bl_idname='screenshot_pose.profile_export';bl_label='导出角色配置'
 filename_ext='.json'
 filter_glob:StringProperty(default='*.json',options={'HIDDEN'})
 def execute(self,context):
  try:
   s=context.scene.screenshot_pose;guard(s);doc=capture(s)
   path=Path(bpy.path.abspath(self.filepath))
   if path.suffix.lower()!='.json':raise ValueError('请选择 .json 文件')
   if any(p and path.resolve()==Path(bpy.path.abspath(p)).resolve() for p in (s.json_path,s.image_path)):raise ValueError('不能覆盖姿势数据或参考图')
   path.write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding='utf-8')
  except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}
  return {'FINISHED'}

class SP_OT_profile_import(Operator,ImportHelper):
 bl_idname='screenshot_pose.profile_import';bl_label='导入角色配置'
 filename_ext='.json'
 filter_glob:StringProperty(default='*.json',options={'HIDDEN'})
 def execute(self,context):
  try:
   s=context.scene.screenshot_pose;guard(s)
   if s.target.library and not s.target.override_library:raise ValueError('链接骨架只读，请先创建 Library Override')
   doc=json.loads(Path(bpy.path.abspath(self.filepath)).read_text(encoding='utf-8'));validate(s,doc);load(s,doc);save(s)
  except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}
  return {'FINISHED'}

class SP_OT_pose_compare(Operator):
 bl_idname='screenshot_pose.pose_compare';bl_label='查看应用前 / 返回当前姿势'
 def execute(self,context):
  a=addon();s=context.scene.screenshot_pose
  try:
   session=a.pose_tools.state(s)
   if session:
    if session.get('kind')!='BEFORE_AFTER':raise ValueError('请先结束当前候选预览')
    a.pose_tools.cancel(s,True);s.status='已返回当前姿势';return {'FINISHED'}
   guard(s);rig=s.target;backup=rig.get('screenshot_pose_backup')
   if not backup:raise ValueError('没有应用前的姿势备份，请先应用一次姿势')
   snapshot=json.loads(backup)
   if snapshot.get('armature')!=rig.name:raise ValueError('备份不属于当前角色')
   session=dict(kind='BEFORE_AFTER',target=rig.name,snapshot=a.retarget.snapshot_pose(rig),backup=backup,
                last_result=s.last_result,frame=context.scene.frame_current,index=-1,
                guides={o.name:list(o.location) for o in a.preview.marker_objects(s)})
   s.candidate_state=json.dumps(session,ensure_ascii=False)
   try:a.retarget.restore_pose(rig,snapshot)
   except Exception:a.pose_tools.cancel(s,True);raise
   s.status='正在查看应用前姿势；点击“返回当前姿势”结束对比，不写关键帧'
  except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}
  return {'FINISHED'}

OPERATORS=(SP_OT_character_profile,SP_OT_diagnosis_focus,SP_OT_profile_export,SP_OT_profile_import,SP_OT_pose_compare)
