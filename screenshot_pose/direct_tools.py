"""Camera reference overlay and transactional wrist/ankle target editing."""
import hashlib,importlib,json,math,time
from pathlib import Path
import bpy
from bpy.props import EnumProperty
from bpy.types import Operator
from mathutils import Matrix,Vector,Quaternion

ENDPOINTS=[('wrist.L','左手腕','固定肩部和其他肢体'),('wrist.R','右手腕','固定肩部和其他肢体'),
           ('ankle.L','左脚踝','固定骨盆和另一条腿'),('ankle.R','右脚踝','固定骨盆和另一条腿')]
_MODALS=[]
_OVERLAY_VIEWS={}
def addon():return importlib.import_module(__package__)
def active(s):
 try:return bool(s.candidate_state and json.loads(s.candidate_state).get('kind')=='DIRECT_EDIT')
 except ValueError:return False
def overlay_slots(camera):
 if not camera or camera.type!='CAMERA':return []
 return [slot for slot in camera.data.background_images if slot.image and slot.image.get('sp_overlay_owner')==camera.name]
def overlay_visible(s):return bool(overlay_slots(bpy.data.objects.get(s.fit_camera)))
def update_overlay(s,context=None):
 for slot in overlay_slots(bpy.data.objects.get(s.fit_camera)):slot.alpha=s.reference_opacity
def hide_overlay(s):
 restore_view_flags(s.fit_camera)
 camera=bpy.data.objects.get(s.fit_camera)
 if not camera or camera.type!='CAMERA':return
 for slot in overlay_slots(camera):
  im=slot.image;camera.data.background_images.remove(slot)
  if im.users==0:bpy.data.images.remove(im)
 if 'sp_overlay_show_before' in camera:
  camera.data.show_background_images=bool(camera['sp_overlay_show_before']);del camera['sp_overlay_show_before']
def restore_view_flags(camera_name=None):
 for window in bpy.context.window_manager.windows:
  for area in window.screen.areas:
   saved=_OVERLAY_VIEWS.get(area.as_pointer())
   if area.type=='VIEW_3D' and saved and (camera_name is None or saved['camera']==camera_name):
    space=area.spaces.active;space.overlay.show_overlays=saved['overlays']
    for name,value in saved['options'].items():setattr(space.overlay,name,value)
    if saved['local']:
     original=bpy.data.objects.get(saved['local_camera'])
     if original:space.camera=original
    del _OVERLAY_VIEWS[area.as_pointer()]
def cleanup_overlays():
 for camera in list(bpy.data.objects):
  if camera.type=='CAMERA' and overlay_slots(camera):
   from types import SimpleNamespace
   hide_overlay(SimpleNamespace(fit_camera=camera.name))
 restore_view_flags();_OVERLAY_VIEWS.clear()
def fit_for(s):
 result=addon().pose_tools.last_result(s).get('fit')
 if not result or not s.target or result.get('target')!=s.target.name:raise ValueError('请先开启“贴合原图”并应用到当前角色')
 if s.mirror or abs(s.yaw)>1e-6 or s.coordinate_mode not in ('IMAGE','HYBRID'):raise ValueError('原图叠加需要图片 / 混合模式，并关闭镜像与朝向修正')
 data=addon().pose_data.read_pose(bpy.path.abspath(s.json_path));image=data['image'];fitted=result['image']
 if any(image[k]!=fitted[k] for k in ('width','height')) or Path(bpy.path.abspath(image['path'])).resolve()!=Path(bpy.path.abspath(fitted['path'])).resolve():raise ValueError('参考图已改变，请重新贴合原图后再试')
 return result
def show_overlay(context,s):
 if addon()._JOB or s.candidate_state:raise ValueError('请先结束任务或预览')
 fit=fit_for(s);path=Path(bpy.path.abspath(fit['image']['path']))
 if not path.is_file():raise ValueError('拟合时的参考图已不存在，请重新识别')
 camera=addon().projection_fit.create_camera(context,s)
 if not overlay_slots(camera):
  image=bpy.data.images.load(str(path),check_existing=True).copy();image.name='SP · 原图叠加';image['sp_overlay_owner']=camera.name
  camera['sp_overlay_show_before']=camera.data.show_background_images
  slot=camera.data.background_images.new();slot.image=image;slot.display_depth='FRONT';slot.frame_method='STRETCH'
 camera.data.show_background_images=True;update_overlay(s)
 if context.area and context.area.type=='VIEW_3D':
  if context.area.as_pointer() not in _OVERLAY_VIEWS:
   options={n:getattr(context.space_data.overlay,n) for n in ('show_bones','show_extras','show_floor','show_axis_x','show_axis_y','show_cursor')}
   _OVERLAY_VIEWS[context.area.as_pointer()]=dict(camera=camera.name,overlays=context.space_data.overlay.show_overlays,options=options,local=context.space_data.use_local_camera,local_camera=context.space_data.camera.name if context.space_data.camera else '')
   if not context.space_data.overlay.show_overlays:
    for name in options:setattr(context.space_data.overlay,name,False)
  context.space_data.overlay.show_overlays=True
  if context.space_data.use_local_camera:context.space_data.camera=camera
  context.space_data.region_3d.view_perspective='CAMERA';context.area.tag_redraw()
 return camera

def _channels(rig,states,names):
 for name in names:
  p=rig.pose.bones[name];state=states[name];p.rotation_mode=state['rotation_mode'];p.matrix_basis=Matrix(state['matrix_basis'])
  for key in ('location','scale','rotation_euler','rotation_quaternion','rotation_axis_angle'):setattr(p,key,state[key])
  for key,value in state['properties'].items():p[key]=value
def action_signature(action):
 if not action:return None
 curves=[]
 for layer in getattr(action,'layers',()):
  for strip in layer.strips:
   for bag in getattr(strip,'channelbags',()):curves.extend(bag.fcurves)
 if not curves:curves=list(getattr(action,'fcurves',()))
 values=[]
 for f in curves:
  modifiers=[]
  for mod in f.modifiers:
   item={}
   for p in mod.bl_rna.properties:
    if p.type in ('BOOLEAN','INT','FLOAT','STRING','ENUM'):
     value=getattr(mod,p.identifier)
     item[p.identifier]=list(value) if getattr(p,'is_array',False) else value
   modifiers.append(item)
  values.append((f.data_path,f.array_index,f.extrapolation,f.mute,modifiers,[(list(k.co),list(k.handle_left),list(k.handle_right),k.interpolation,k.handle_left_type,k.handle_right_type) for k in f.keyframe_points],[list(k.co) for k in f.sampled_points]))
 return hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()
def restore_tracks(rig,session):
 if rig.animation_data:
  for track in rig.animation_data.nla_tracks:
   if track.name in session.get('nla_mutes',{}):track.mute=session['nla_mutes'][track.name]
def begin(context,s):
 a=addon();rig,mapping,data=a.pose_tools.check(s);fit_for(s)
 if s.candidate_state:raise ValueError('请先结束当前预览')
 endpoint=s.direct_endpoint;side=endpoint[-1];arm=endpoint.startswith('wrist')
 roles=[r+'.'+side for r in (('upper_arm','forearm','hand') if arm else ('thigh','shin','foot'))]
 if any(r not in mapping for r in roles):raise ValueError('此肢体缺少完整 FK 映射')
 bones=[rig.pose.bones[mapping[r]] for r in roles]
 if any(not a.orientation._safe(b,rig) for b in bones):raise ValueError('此肢体受旋转约束或驱动，请改选可直接控制的 FK 骨骼')
 snapshot=a.retarget.snapshot_pose(rig)
 state=dict(kind='DIRECT_EDIT',target=rig.name,frame=context.scene.frame_current,signature=a.pose_tools.signature(s),
            snapshot=snapshot,backup=rig.get('screenshot_pose_backup'),last_result=s.last_result,index=-1,
            guides={o.name:list(o.location) for o in a.preview.marker_objects(s)},endpoint=endpoint,
            bones=[b.name for b in bones],roles=roles,properties=[],changed=False,
            nla_mutes={t.name:t.mute for t in rig.animation_data.nla_tracks} if rig.animation_data else {},
            lock_height=not arm and s.direct_lock_height,fit=a.pose_tools.last_result(s)['fit'])
 state['before_action_signature']=action_signature(rig.animation_data.action if rig.animation_data else None)
 s.candidate_state=json.dumps(state,ensure_ascii=False)
 try:
  # Suspend animation only for preview. Drivers stay active. Generic cancellation
  # restores both the original Action and NLA mute states.
  if rig.animation_data:
   if rig.animation_data.action:rig.animation_data.action.use_fake_user=True
   rig.animation_data.action=None
   for t in rig.animation_data.nla_tracks:t.mute=True
  for bone,key,value in a.retarget._fk_switches(rig,{r:mapping[r] for r in roles}):
   if not s.switch_fk and abs(bone[key]-value)>1e-6:raise ValueError('请先开启自动切换 FK 并应用姿势')
   bone[key]=value;state['properties'].append([bone.name,key])
  a.retarget._update(rig)
  p=[rig.matrix_world@b.head for b in bones]
  if min((p[1]-p[0]).length,(p[2]-p[1]).length)<1e-7:raise ValueError('此肢体关节重合，无法拖动')
  state.update(start=[list(v) for v in p],lengths=[(p[1]-p[0]).length,(p[2]-p[1]).length],
               end_rotation=list(bones[-1].matrix.to_quaternion()),working=a.retarget.snapshot_pose(rig)['bones'])
  s.candidate_state=json.dumps(state,ensure_ascii=False);s.direct_residual=0.
  return state
 except Exception:a.pose_tools.cancel(s,True);raise

def validate(s):
 a=addon();state=a.pose_tools.state(s)
 if not state or state.get('kind')!='DIRECT_EDIT':raise ValueError('拖动会话已结束')
 a.pose_tools.validate_session(s,state)
 fit=fit_for(s)
 if fit['camera']!=state['fit']['camera']:raise ValueError('匹配相机已改变，请取消后重新开始')
 return state

def move(context,s,target):
 a=addon();state=validate(s)
 if state.get('comparing'):toggle_compare(s);state=validate(s)
 rig=s.target;target=Vector(target)
 if not all(math.isfinite(v) for v in target):raise ValueError('目标位置无效')
 if state['lock_height']:target.z=state['start'][2][2]
 _channels(rig,state['working'],state['bones'])
 for name,key in state['properties']:rig.pose.bones[name][key]=1.
 a.retarget._update(rig)
 upper,lower,end=[rig.pose.bones[n] for n in state['bones']];anchor=Vector(state['start'][0]);pole=Vector(state['start'][1])
 knee,effective,unreachable=a.contact_solver.solve_two_bone(anchor,target,pole,*state['lengths'])
 for _ in range(3):
  a.projection_fit._aim_link(rig,upper,lower,knee-anchor)
  a.projection_fit._aim_link(rig,lower,end,effective-knee)
 a.retarget._set_rotation(end,a.retarget._to_local_rotation(end,Quaternion(state['end_rotation'])));a.retarget._update(rig)
 after=a.retarget.snapshot_pose(rig);allowed=set(state['bones'])|{n for n,k in state['properties']}
 for name,before in state['working'].items():
  now=after['bones'][name]
  if name not in allowed and (before['properties']!=now['properties'] or max(abs(before['matrix_basis'][i][j]-now['matrix_basis'][i][j]) for i in range(4) for j in range(4))>2e-6):
   a.pose_tools.cancel(s,True);raise ValueError('绑定联动影响了其他控制骨，已撤回本次拖动')
 actual=rig.matrix_world@end.head;residual=(actual-target).length
 anchor_error=(rig.matrix_world@upper.head-anchor).length
 if anchor_error>max(sum(state['lengths'])*1e-4,1e-5):a.pose_tools.cancel(s,True);raise ValueError('肩部 / 骨盆未保持固定，已撤回')
 state.update(changed=True,requested=list(target),actual=list(actual),residual=residual,unreachable=unreachable)
 s.candidate_state=json.dumps(state,ensure_ascii=False);s.direct_residual=residual
 s.status='拖动预览：Enter 确认，Esc / 右键取消'+('；目标超出骨长，已显示最近可达位置' if residual>sum(state['lengths'])*.005 else '')
 return state

def toggle_compare(s):
 a=addon();state=validate(s)
 if not state['changed']:raise ValueError('请先拖动目标，再比较修改前后')
 if state.get('comparing'):
  _channels(s.target,state.pop('comparison_pose'),state['bones']);state['comparing']=False
 else:
  snapshot=a.retarget.snapshot_pose(s.target)['bones'];state['comparison_pose']={n:snapshot[n] for n in state['bones']}
  _channels(s.target,state['working'],state['bones']);state['comparing']=True
 a.retarget._update(s.target);s.candidate_state=json.dumps(state,ensure_ascii=False)
 s.status='正在查看拖动前；按 B 返回拖动预览' if state['comparing'] else '已返回拖动预览；Enter 确认，Esc 取消'

def undo(context,s):
 a=addon()
 if a._JOB or s.candidate_state:raise ValueError('请先结束任务或预览')
 if not s.direct_undo_state or not s.target:raise ValueError('没有可撤销的拖动')
 state=json.loads(s.direct_undo_state);a.pose_tools.validate_session(s,state);rig=s.target;current=a.retarget.snapshot_pose(rig)
 after=state['after']
 if current['action']!=after['action'] or any(v['properties']!=current['bones'][n]['properties'] or max(abs(v['matrix_basis'][i][j]-current['bones'][n]['matrix_basis'][i][j]) for i in range(4) for j in range(4))>2e-5 for n,v in after['bones'].items()):raise ValueError('人物已被其他操作修改，未覆盖当前姿势；请先处理这些新修改')
 if action_signature(rig.animation_data.action if rig.animation_data else None)!=state['after_action_signature']:raise ValueError('动作关键帧已改变，不能直接撤销此拖动')
 original=bpy.data.actions.get(state['snapshot']['action']) if state['snapshot']['action'] else None
 if action_signature(original)!=state['before_action_signature']:raise ValueError('原动作已改变，不能直接返回旧动作')
 if ({t.name:t.mute for t in rig.animation_data.nla_tracks} if rig.animation_data else {})!=state['nla_mutes']:raise ValueError('动画轨道已改变，未覆盖当前设置')
 if {o.name:list(o.location) for o in a.preview.marker_objects(s)}!=state['after_guides']:raise ValueError('关节点已被修改，未覆盖当前校正')
 restore_tracks(rig,state);a.retarget.restore_pose(rig,state['snapshot']);a.pose_tools.restore_backup(rig,state['backup'])
 for o in a.preview.marker_objects(s):
  if o.name in state['guides']:o.location=state['guides'][o.name]
 s.last_result=state['last_result'];s.direct_undo_state='';a.preview.refresh_lines(context,s);s.status='已撤销上次拖动，恢复姿势、动作与对应关节点'

def _sync_guides(s,state,rig):
 a=addon();data=a.pose_data.read_pose(bpy.path.abspath(s.json_path));canonical,_=a.pose_data.to_joints(data,s.coordinate_mode,s.depth)
 aspect=data['image']['height']/data['image']['width'];p=data['landmarks'];left,right=p[11],p[12]
 span=math.hypot(left['x']-right['x'],(left['y']-right['y'])*aspect);v=Vector(canonical['shoulder.L'])-Vector(canonical['shoulder.R']);factor=math.hypot(v.x,v.z)/max(span,1e-8)
 cx=(p[23]['x']+p[24]['x'])/2;cy=(p[23]['y']+p[24]['y'])*aspect/2;cfg=state['fit']['camera'];R=Matrix(cfg['rotation']);side=state['endpoint'][-1]
 joints=('elbow','wrist') if state['endpoint'].startswith('wrist') else ('knee','ankle')
 for joint,bone in zip(joints,state['bones'][1:]):
  from . import camera_data
  x,y=camera_data.project([rig.matrix_world@rig.pose.bones[bone].head],cfg)[0]
  marker=next((o for o in a.preview.marker_objects(s) if o['sp_joint']==joint+'.'+side),None)
  if marker:marker.location.x=(x-cx)*factor;marker.location.z=(cy-y)*factor

def confirm(context,s):
 a=addon();state=validate(s)
 if state.get('comparing'):toggle_compare(s);state=validate(s)
 if not state['changed']:a.pose_tools.cancel(s,True);s.status='未移动目标，保持原姿势';return
 rig=s.target;preview=a.retarget.snapshot_pose(rig);created=None;previous_undo=s.direct_undo_state
 try:
  restore_tracks(rig,state);a.retarget.restore_pose(rig,state['snapshot'])
  if s.keyframe:created=a.pose_tools.copy_action(rig,state['snapshot'])
  _channels(rig,preview['bones'],state['bones'])
  for name,key in state['properties']:rig.pose.bones[name][key]=preview['bones'][name]['properties'][key]
  a.retarget._update(rig)
  actual=rig.matrix_world@rig.pose.bones[state['bones'][-1]].head
  if (actual-Vector(state['actual'])).length>max(1e-4,sum(state['lengths'])*.001):raise ValueError('原动画覆盖了拖动结果，请调整动画轨道后再试')
  if s.keyframe:a.contact_solver.key_touched(rig,state['bones'],context.scene.frame_current,state['properties'])
  if not state['backup']:rig['screenshot_pose_backup']=json.dumps(state['snapshot'],ensure_ascii=False)
  else:a.pose_tools.restore_backup(rig,state['backup'])
  _sync_guides(s,state,rig)
  result=json.loads(state['last_result']) if state['last_result'] else {}
  result['direct_edit']={k:state[k] for k in ('endpoint','requested','actual','residual','unreachable','lock_height')};result['direct_edit']['keyframe']=s.keyframe
  undo_state={k:state[k] for k in ('target','frame','signature','snapshot','backup','last_result','guides','nla_mutes','before_action_signature')}
  undo_state.update(after=a.retarget.snapshot_pose(rig),after_guides={o.name:list(o.location) for o in a.preview.marker_objects(s)},after_action_signature=action_signature(rig.animation_data.action if rig.animation_data else None))
  s.direct_undo_state=json.dumps(undo_state,ensure_ascii=False)
  s.last_result=json.dumps(result,ensure_ascii=False);a.preview.refresh_lines(context,s);s.candidate_state=''
  s.status='局部拖动已确认'+('并打关键帧' if s.keyframe else '（未打关键帧）')+'；已同步对应二维关节点'
 except Exception:
  a.pose_tools.cancel(s,True)
  s.direct_undo_state=previous_undo
  if created:
   created.use_fake_user=False
   if created.users==0:bpy.data.actions.remove(created)
  raise

def stop_modals():
 for modal in list(_MODALS):modal.finish()

class SP_OT_reference_overlay(Operator):
 bl_idname='screenshot_pose.reference_overlay';bl_label='原图叠加预览'
 action:EnumProperty(items=[('SHOW','显示',''),('HIDE','隐藏','')],default='SHOW')
 def execute(self,context):
  try:
   s=context.scene.screenshot_pose
   if self.action=='HIDE':hide_overlay(s)
   else:show_overlay(context,s)
   s.status='原图叠加已隐藏' if self.action=='HIDE' else '原图已叠加；调整透明度后在匹配相机中比较'
  except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}
  return {'FINISHED'}

class SP_OT_direct_action(Operator):
 bl_idname='screenshot_pose.direct_action';bl_label='确认 / 取消局部拖动'
 action:EnumProperty(items=[('CONFIRM','确认',''),('CANCEL','取消',''),('COMPARE','对比拖动前后',''),('UNDO','撤销上次拖动','')])
 def execute(self,context):
  try:
   s=context.scene.screenshot_pose
   if self.action=='UNDO':undo(context,s);return {'FINISHED'}
   if not active(s):raise ValueError('没有进行中的局部拖动')
   if self.action=='CONFIRM':confirm(context,s)
   elif self.action=='COMPARE':toggle_compare(s)
   else:addon().pose_tools.cancel(s);s.status='已取消拖动，恢复原姿势'
  except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}
  return {'FINISHED'}

class SP_OT_direct_drag(Operator):
 bl_idname='screenshot_pose.direct_drag';bl_label='在角色上拖动目标'
 _handle=None;_timer=None;_dragging=False;_pending=None;_finished=False
 def finish(self):
  if self._finished:return
  self._finished=True
  if self._handle:
   bpy.types.SpaceView3D.draw_handler_remove(self._handle,'WINDOW');self._handle=None
  if self._timer:
   bpy.context.window_manager.event_timer_remove(self._timer);self._timer=None
  if self in _MODALS:_MODALS.remove(self)
  if getattr(self,'_area',None):self._area.tag_redraw()
 def invoke(self,context,event):
  try:
   if not context.area or context.area.type!='VIEW_3D':raise ValueError('请在三维视图侧栏启动拖动')
   if context.space_data.region_3d.view_perspective!='CAMERA':raise ValueError('请先点击“显示原图 / 进入匹配相机”')
   camera=context.space_data.camera if context.space_data.use_local_camera else context.scene.camera
   if not camera or camera.name!=context.scene.screenshot_pose.fit_camera or camera.data.type!='ORTHO':raise ValueError('请先进入插件生成的正交匹配相机')
   self._area=context.area;self._region=next(r for r in context.area.regions if r.type=='WINDOW');self._space=context.space_data;self._scene=context.scene
   self._camera=camera;self._camera_values=(tuple(camera.location),tuple(camera.rotation_euler),camera.data.ortho_scale,camera.data.shift_x,camera.data.shift_y)
   self._finished=False;self._dragging=False;self._pending=None
   begin(context,self._scene.screenshot_pose)
   self._handle=bpy.types.SpaceView3D.draw_handler_add(self.draw_overlay,(),'WINDOW','POST_PIXEL')
   self._timer=context.window_manager.event_timer_add(.08,window=context.window);_MODALS.append(self)
   context.window_manager.modal_handler_add(self);context.area.tag_redraw()
   self._scene.screenshot_pose.status='拖动亮色圆点；Enter 确认，Esc / 右键取消；其他肢体与肩 / 骨盆固定'
  except Exception as exc:
   if getattr(self,'_scene',None) and active(self._scene.screenshot_pose):addon().pose_tools.cancel(self._scene.screenshot_pose,True)
   self.finish();self.report({'ERROR'},str(exc));return {'CANCELLED'}
  return {'RUNNING_MODAL'}
 def point(self):
  s=self._scene.screenshot_pose;state=addon().pose_tools.state(s)
  return s.target.matrix_world@s.target.pose.bones[state['bones'][-1]].head
 def draw_overlay(self):
  if self._finished or bpy.context.area!=self._area or not active(self._scene.screenshot_pose):return
  from bpy_extras import view3d_utils
  import gpu,blf
  from gpu_extras.batch import batch_for_shader
  p=view3d_utils.location_3d_to_region_2d(self._region,self._space.region_3d,self.point())
  if p is None:return
  vertices=[(p.x+12*math.cos(i*math.tau/48),p.y+12*math.sin(i*math.tau/48)) for i in range(49)]
  shader=gpu.shader.from_builtin('UNIFORM_COLOR');shader.bind();shader.uniform_float('color',(0.25,1.,.78,1.));batch_for_shader(shader,'LINE_STRIP',{'pos':vertices}).draw(shader)
  text='修改前 | B 返回拖动预览' if addon().pose_tools.state(self._scene.screenshot_pose).get('comparing') else '拖动目标 | B 对比 | Enter 确认 | Esc 取消'
  tx=min(max(p.x+18,8),max(8,self._region.width-350));ty=min(p.y+12,self._region.height-28)
  shader.uniform_float('color',(.035,.09,.10,1.));batch_for_shader(shader,'TRI_FAN',{'pos':[(tx-5,ty-6),(tx+338,ty-6),(tx+338,ty+21),(tx-5,ty+21)]}).draw(shader)
  blf.position(0,tx,ty,0);blf.size(0,14);blf.color(0,.85,1.,.95,1.);blf.draw(0,text)
 def apply_mouse(self,context,mouse):
  from bpy_extras import view3d_utils
  coordinate=(mouse[0]-self._region.x,mouse[1]-self._region.y)
  point=view3d_utils.region_2d_to_location_3d(self._region,self._space.region_3d,coordinate,self._depth)
  move(context,self._scene.screenshot_pose,point+self._offset);self._area.tag_redraw()
 def modal(self,context,event):
  if self._finished:return {'CANCELLED'}
  s=self._scene.screenshot_pose
  if not active(s):self.finish();return {'FINISHED'}
  try:
   if context.scene!=self._scene or context.area!=self._area:raise ValueError('视图或工程已改变，已取消拖动')
   camera=self._space.camera if self._space.use_local_camera else context.scene.camera
   if camera!=self._camera or self._space.region_3d.view_perspective!='CAMERA' or (tuple(camera.location),tuple(camera.rotation_euler),camera.data.ortho_scale,camera.data.shift_x,camera.data.shift_y)!=self._camera_values:raise ValueError('匹配相机已改变，已取消拖动')
   validate(s)
   if not self._dragging and event.type in ('LEFTMOUSE','MOUSEMOVE'):
    sidebar=next((r for r in self._area.regions if r.type=='UI'),None)
    if sidebar and sidebar.x<=event.mouse_x<sidebar.x+sidebar.width and sidebar.y<=event.mouse_y<sidebar.y+sidebar.height:return {'PASS_THROUGH'}
   if event.type in ('ESC','RIGHTMOUSE') and event.value=='PRESS':addon().pose_tools.cancel(s);self.finish();return {'CANCELLED'}
   if event.type=='B' and event.value=='PRESS' and not self._dragging:
    if addon().pose_tools.state(s)['changed']:toggle_compare(s)
    else:s.status='尚未拖动目标；移动亮色圆点后可按 B 对比'
    self._area.tag_redraw();return {'RUNNING_MODAL'}
   if event.type in ('RET','NUMPAD_ENTER') and event.value=='PRESS':
    if self._pending:self.apply_mouse(context,self._pending);self._pending=None
    confirm(context,s);self.finish();return {'FINISHED'}
   if event.type=='LEFTMOUSE' and event.value=='PRESS':
    from bpy_extras import view3d_utils
    p=view3d_utils.location_3d_to_region_2d(self._region,self._space.region_3d,self.point())
    mouse=Vector((event.mouse_x-self._region.x,event.mouse_y-self._region.y))
    if p is not None and (p-mouse).length<22:
     self._depth=self.point().copy();self._offset=self._depth-view3d_utils.region_2d_to_location_3d(self._region,self._space.region_3d,mouse,self._depth);self._dragging=True
   elif event.type=='MOUSEMOVE' and self._dragging:self._pending=(event.mouse_x,event.mouse_y)
   elif event.type=='TIMER' and self._pending:self.apply_mouse(context,self._pending);self._pending=None
   elif event.type=='LEFTMOUSE' and event.value=='RELEASE' and self._dragging:
    self.apply_mouse(context,(event.mouse_x,event.mouse_y));self._pending=None;self._dragging=False
   return {'RUNNING_MODAL'}
  except Exception as exc:
   if active(s):addon().pose_tools.cancel(s,True)
   self.finish();self.report({'ERROR'},str(exc));return {'CANCELLED'}
 def cancel(self,context):
  if active(self._scene.screenshot_pose):addon().pose_tools.cancel(self._scene.screenshot_pose,True)
  self.finish()

OPERATORS=(SP_OT_reference_overlay,SP_OT_direct_action,SP_OT_direct_drag)
