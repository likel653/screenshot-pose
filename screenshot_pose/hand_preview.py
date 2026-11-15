"""Transactional palm audition and standalone manual hand editor integration."""
import copy,importlib,json,uuid
from pathlib import Path
import bpy
import numpy as np
from mathutils import Matrix
from bpy.props import EnumProperty
from bpy.types import Operator
from . import orientation_math as om,hand_edit_data

def addon():return importlib.import_module(__package__)
_HISTORY_CACHE={}
def can_undo(s):
 try:
  path=Path(bpy.path.abspath(s.json_path));stat=path.stat();key=(str(path),stat.st_mtime_ns,stat.st_size,s.hand_edit_side)
  if key not in _HISTORY_CACHE:
   if len(_HISTORY_CACHE)>8:_HISTORY_CACHE.clear()
   _HISTORY_CACHE[key]=bool(json.loads(path.read_text(encoding='utf-8')).get('hand_edit_history',{}).get(s.hand_edit_side))
  return _HISTORY_CACHE[key]
 except (OSError,ValueError,TypeError):return False
def active(s):
 try:return bool(s.candidate_state and json.loads(s.candidate_state).get('kind')=='HAND_PALM')
 except ValueError:return False

def options(data,side,fallback):
 hd=data.get('orientation',{}).get('hands',{}).get(side,{})
 frames=[hd['frame']]+hd.get('alternatives',[]) if hd.get('frame') else [fallback]
 result=[]
 for i,frame in enumerate(frames[:4]):
  for angle,label in ((0.,'原掌向'),(np.pi,'翻转掌背'),(np.pi/2,'顺转 90°'),(-np.pi/2,'逆转 90°')) if i==0 else ((0.,'原掌向'),(np.pi,'翻转掌背')):
   c=np.cos(angle);s=np.sin(angle);r=om.proper_rotation(frame)@np.array([[c,0,s],[0,1,0],[-s,0,c]])
   if any(np.linalg.norm(np.array(v['frame'])-r)<1e-6 for v in result):continue
   name=('原识别' if i==0 and hd.get('frame') else '当前角色' if i==0 else f'几何候选 {i+1}')+' · '+label
   result.append(dict(name=name,frame=r.tolist(),side=side,source='manual_comparison_not_new_inference'))
 return result

def palm_document(data,spec):
 result=copy.deepcopy(data);details=result.setdefault('orientation',{});details.setdefault('schema','screenshot_pose/orientation_v1');details.setdefault('camera_axes','X_right_Y_away_Z_up');hd=details.setdefault('hands',{}).setdefault(spec['side'],{})
 old=om.proper_rotation(hd.get('frame',spec['frame']));new=om.proper_rotation(spec['frame'])
 if 'world' in hd:
  p=np.asarray(hd['world'],float)
  if p.shape!=(21,3) or not np.isfinite(p).all():raise ValueError('手部三维点无效')
  hd['world']=((p-p[0])@(new@old.T).T+p[0]).tolist()
 hd.update(frame=new.tolist(),alternatives=[],palm_choice=spec['name'],confidence=1.,confidence_method='user_selected_not_model_probability')
 hd.setdefault('source','manual_current_palm');hd.pop('bone_frames',None)
 return result

def fallback_frame(rig,mapping,s,side=None):
 a=addon();side=side or s.hand_edit_side;cal=a.orientation.calibration(rig,mapping,s)
 if side not in cal['hands']:raise ValueError('当前手缺少掌部校准，请指定食指 / 小指参考骨')
 hand=rig.pose.bones[mapping['hand.'+side]];fit=a.pose_tools.last_result(s).get('fit');camera=Matrix(fit['camera']['rotation']) if fit else Matrix.Identity(3)
 return om.proper_rotation(camera@rig.matrix_world.to_quaternion().to_matrix()@hand.matrix.to_3x3().normalized()@hand.bone.matrix_local.to_3x3().normalized().transposed()@Matrix(cal['hands'][side]['frame'])).tolist()

def character_seed(s,side):
 a=addon()
 if not s.target:return None
 rig=s.target;mapping=a.pose_tools.mapping_for(s);cal=a.orientation.calibration(rig,mapping,s)
 if side not in cal['hands'] or 'hand.'+side not in mapping:return None
 hand=rig.pose.bones[mapping['hand.'+side]];points=[list(hand.head)]
 for finger in a.finger_pose.FINGERS:
  chain=[a.finger_pose.control(rig,finger,n,side) for n in (1,2,3)]
  if not all(chain):return None
  points.extend([list(chain[0].head),list(chain[1].head),list(chain[2].head),list(chain[2].tail)])
 actual=np.asarray(hand.matrix.to_3x3().normalized()@hand.bone.matrix_local.to_3x3().normalized().transposed()@Matrix(cal['hands'][side]['frame']))
 data=a.pose_data.read_pose(bpy.path.abspath(s.json_path));frame=data.get('orientation',{}).get('hands',{}).get(side,{}).get('frame')
 if data.get('orientation',{}).get('hands',{}).get(side,{}).get('source')=='manual_hand_landmarks':return None
 if frame is None:frame=fallback_frame(rig,mapping,s,side)
 frame=om.proper_rotation(frame);p=np.asarray(points);p=(p-p[0])@actual@frame.T;length=np.linalg.norm(p[9])
 if length<1e-8:return None
 p*=.07/length
 return dict(world=p.tolist(),frame=frame.tolist(),source='current_character_pose_normalized_not_image_inference')

def collect(s):
 a=addon()
 if s.candidate_state:raise ValueError('请先确认或取消正在进行的预览')
 rig,mapping,data=a.pose_tools.check(s);specs=options(data,s.hand_edit_side,fallback_frame(rig,mapping,s));s.hand_palm_options.clear()
 for spec in specs:
  row=s.hand_palm_options.add();row.name=spec['name'];row.spec=json.dumps(spec,ensure_ascii=False);row.score='仅预览手腕；指骨局部姿势保留'
 s.hand_palm_side=s.hand_edit_side;s.hand_palm_signature=a.pose_tools.signature(s);s.hand_palm_index=0;s.status=f'已准备 {len(specs)} 个掌向候选；不代表新增模型识别'
 return specs

def session(s):
 if s.candidate_state and not active(s):raise ValueError('全身姿势预览正在进行，请先确认或取消')
 if not s.candidate_state:
  if not s.hand_palm_options or s.hand_palm_side!=s.hand_edit_side or s.hand_palm_signature!=addon().pose_tools.signature(s):collect(s)
  a=addon();rig,_,_=a.pose_tools.check(s)
  if s.strength<=0:raise ValueError('请先提高应用强度')
  state=dict(kind='HAND_PALM',target=rig.name,side=s.hand_edit_side,snapshot=a.retarget.snapshot_pose(rig),backup=rig.get('screenshot_pose_backup'),last_result=s.last_result,frame=bpy.context.scene.frame_current,signature=a.pose_tools.signature(s),index=-1,guides={o.name:list(o.location) for o in a.preview.marker_objects(s)})
  s.candidate_state=json.dumps(state,ensure_ascii=False)
 return json.loads(s.candidate_state)

def restore(s,state):
 a=addon();a.retarget.restore_pose(s.target,state['snapshot']);a.pose_tools.restore_backup(s.target,state['backup']);s.last_result=state['last_result']

def audition(context,s,index=None):
 a=addon();state=session(s)
 try:
  a.pose_tools.validate_session(s,state)
  if s.hand_edit_side!=state['side']:raise ValueError('左右手选择已改变，请取消预览后重新开始')
  index=s.hand_palm_index if index is None else index
  if not 0<=index<len(s.hand_palm_options):raise ValueError('掌向候选序号无效')
  spec=json.loads(s.hand_palm_options[index].spec);data=palm_document(a.pose_data.read_pose(bpy.path.abspath(s.json_path)),spec);restore(s,state)
  result=a.pose_tools.apply_local(context,s,scope='HAND_'+state['side'],fingers=(),data_override=data,preview_session=state,keyframe=False,force_palm=True,palm_reset=True)
  if a.pose_tools.mapping_for(s)['hand.'+state['side']] not in result['touched']:raise ValueError('当前手腕控制器不可写，未改变角色')
  a.pose_tools.restore_backup(s.target,state['backup']);state['index']=index;s.candidate_state=json.dumps(state,ensure_ascii=False);s.hand_palm_index=index;s.hand_palm_active_index=index
  result['palm_preview']=dict(side=state['side'],name=spec['name'],temporary=True);s.last_result=json.dumps(result,ensure_ascii=False);s.status='临时掌向：'+spec['name']+'；确认保留或取消恢复'
  return result
 except Exception:a.pose_tools.cancel(s,True);raise

def adopt_path(s,output,old_signature):
 a=addon();choice=s.chosen_candidate if s.chosen_signature==old_signature else ''
 s.json_path=str(output);s.pose_candidates.clear();s.candidate_signature='';s.chosen_candidate=choice;s.chosen_signature=a.pose_tools.signature(s) if choice else '';s.hand_palm_signature='';s.hand_palm_options.clear()

def confirm(context,s):
 a=addon()
 if not active(s):raise ValueError('请先预览一个掌向候选')
 state=json.loads(s.candidate_state);actions=set(bpy.data.actions)
 try:
  a.pose_tools.validate_session(s,state)
  if state['index']<0:raise ValueError('尚未预览掌向')
  spec=json.loads(s.hand_palm_options[state['index']].spec);data=palm_document(a.pose_data.read_pose(bpy.path.abspath(s.json_path)),spec)
  out=a.runtime_dir()/'hands'/('palm_'+uuid.uuid4().hex+'.json');out.parent.mkdir(parents=True,exist_ok=True)
  # Write first: a disk failure must not leave newly keyed pose changes.
  from .worker import write_json
  write_json(out,data);restore(s,state)
  result=a.pose_tools.apply_local(context,s,scope='HAND_'+state['side'],fingers=(),data_override=data,preview_session=state,keyframe=s.keyframe,force_palm=True,palm_reset=True)
  if a.pose_tools.mapping_for(s)['hand.'+state['side']] not in result['touched']:raise ValueError('当前手腕控制器不可写，未改变角色')
  adopt_path(s,out,state['signature']);side=state['side'].lower();setattr(s,'palm_flip_'+side,False);setattr(s,'palm_roll_'+side,0.);setattr(s,'hand_candidate_'+side,'1');setattr(s,'orient_hand_'+side,True)
  s.candidate_state='';s.hand_palm_active_index=-1;s.status='已保留掌向：'+spec['name']+('，已打关键帧' if s.keyframe else '');return result
 except Exception:
  a.pose_tools.cancel(s,True)
  for action in set(bpy.data.actions)-actions:
   if action.users==0:bpy.data.actions.remove(action)
  raise

def edited_complete(scene,source,signature,target,frame,output):
 a=addon();s=scene.screenshot_pose
 if not Path(output).is_file():s.status='已取消手部关节点编辑，保留原结果';return
 data=a.pose_data.read_pose(output);info=data['hand_edit']
 if str(Path(bpy.path.abspath(s.json_path)).resolve())!=str(Path(source).resolve()) or a.pose_tools.signature(s)!=signature:
  s.status='手部校正已保存，但当前主图已改变，因此未切换数据';return
 adopt_path(s,output,signature)
 if info['apply_on_return'] and s.target and s.target.name==target and scene.frame_current==frame and bpy.context.scene==scene:
  try:report=a.pose_tools.apply_local(bpy.context,s,scope='HAND_'+info['side'],fingers=tuple(info['fingers']),update_palm=False)
  except Exception as exc:s.status='校正数据已保存；局部应用已撤回：'+str(exc)[:150];return
  names=dict(zip(('thumb','index','middle','ring','pinky'),('拇指','食指','中指','无名指','小指')))
  s.status=('已仅更新修改的手指：'+'、'.join(names.get(f,f) for f in info['fingers']) if report['touched'] else '校正数据已保存，但未更新指骨；请检查强度、映射和可写控制链')+('；请查看局部更新提示' if report.get('warnings') else '')
 else:s.status='手部校正已保存，未改变角色；可仅更新对应手'

class SP_OT_hand_editor(Operator):
 bl_idname='screenshot_pose.hand_editor';bl_label='校正手部 21 点'
 side:EnumProperty(items=[('CURRENT','当前手',''),('L','左手',''),('R','右手','')],default='CURRENT')
 def execute(self,context):
  try:
   a=addon();s=context.scene.screenshot_pose
   if a._JOB or s.candidate_state:raise ValueError('请先结束任务或预览')
   source=str(Path(bpy.path.abspath(s.json_path)).resolve());a.pose_data.read_pose(source);signature=a.pose_tools.signature(s);side=s.hand_edit_side if self.side=='CURRENT' else self.side
   out=a.runtime_dir()/'hands'/('edit_'+uuid.uuid4().hex+'.json');out.parent.mkdir(parents=True,exist_ok=True);target=s.target.name if s.target else '';frame=context.scene.frame_current
   python=a.runtime_python('SAM3D')
   if not python.is_file():python=a.runtime_python()
   command=[python,a.HERE/'hand_editor.py','--input',source,'--output',out,'--side',side];seed=character_seed(s,side)
   if seed:
    seed_path=out.with_suffix('.seed.json');seed_path.write_text(json.dumps(seed),encoding='utf-8');command+=['--seed',seed_path]
   a.start_job(context.scene,command,'校正手部 21 点',lambda scene:edited_complete(scene,source,signature,target,frame,out))
  except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}
  return {'FINISHED'}

class SP_OT_palm_preview(Operator):
 bl_idname='screenshot_pose.palm_preview';bl_label='掌向候选预览';bl_options={'REGISTER','UNDO'}
 action:EnumProperty(items=[('COLLECT','准备候选',''),('PREVIEW','预览',''),('NEXT','下一个',''),('PREVIOUS','上一个',''),('CONFIRM','确认',''),('CANCEL','取消','')],default='COLLECT')
 def execute(self,context):
  s=context.scene.screenshot_pose
  try:
   if self.action=='COLLECT':collect(s)
   elif self.action=='CANCEL':
    if not active(s):raise ValueError('当前没有掌向预览')
    addon().pose_tools.cancel(s)
   elif self.action=='CONFIRM':confirm(context,s)
   elif self.action in ('NEXT','PREVIOUS'):
    state=session(s);index=state['index'];audition(context,s,(index+(1 if self.action=='NEXT' else -1))%len(s.hand_palm_options))
   else:audition(context,s)
  except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}
  return {'FINISHED'}

class SP_OT_hand_edit_undo(Operator):
 bl_idname='screenshot_pose.hand_edit_undo';bl_label='撤回上次此手校正数据'
 bl_description='还原此手上次保存前的数据，不自动改变角色；其他手与身体数据保留'
 def execute(self,context):
  try:
   a=addon();s=context.scene.screenshot_pose
   if a._JOB or s.candidate_state:raise ValueError('请先结束任务或预览')
   signature=a.pose_tools.signature(s);data=hand_edit_data.undo(a.pose_data.read_pose(bpy.path.abspath(s.json_path)),s.hand_edit_side)
   out=a.runtime_dir()/'hands'/('undo_'+uuid.uuid4().hex+'.json');out.parent.mkdir(parents=True,exist_ok=True)
   from .worker import write_json
   write_json(out,data);adopt_path(s,out,signature);s.status='已撤回此手上次校正数据；角色保持原样，可仅更新对应手'
  except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}
  return {'FINISHED'}

OPERATORS=(SP_OT_hand_editor,SP_OT_palm_preview,SP_OT_hand_edit_undo)
