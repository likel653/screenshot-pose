"""Runs inside isolated Blender, using the actual add-on's Bake transaction."""
import bpy,addon_utils,json,sys,shutil
from pathlib import Path
from mathutils import Vector
job=json.loads(Path(sys.argv[sys.argv.index('--')+1]).read_text(encoding='utf8'));out=Path(job['output']);root=Path(job['project_root']);sys.path.insert(0,str(root))
addon_utils.enable('screenshot_pose',default_set=False)
import screenshot_pose as a
assert Path(a.__file__).resolve().is_relative_to(root),'加载了错误的插件副本'
scene=bpy.context.scene;s=scene.screenshot_pose
rig=bpy.data.objects.get(job['rig']) if job['rig'] else None
if not rig:
 candidates=[o for o in scene.objects if o.type=='ARMATURE']
 if len(candidates)!=1:raise ValueError('请指定骨架 --rig，候选：'+', '.join(o.name for o in candidates))
 rig=candidates[0]
if rig.type!='ARMATURE' or rig.name not in scene.objects:raise ValueError('目标不是当前场景骨架')
s.target=rig;a.set_mapping(s);s.video_cache=job['cache'];s.video_start_frame=job['start_frame'];s.coordinate_mode='HYBRID';s.video_temporal_mode='ADAPTIVE';s.video_stabilize=True;s.video_smooth=.12;s.video_set_range=True
scene.render.fps=round(job['fps']);scene.render.fps_base=round(job['fps'])/job['fps']
session=a.video_tools.Bake(bpy.context,s,job['cache'])
try:
 while session.tick():
  (out/'progress.json').write_text(json.dumps(dict(stage='生成角色关键帧',status=s.status),ensure_ascii=False),encoding='utf8')
except Exception:session.rollback();raise
assert session.committed
rig.animation_data.action.name='自动生成 · '+Path(job['source']).stem
curves=list(a.animation_preview.curves(rig.animation_data.action));keys=sum(len(c.keyframe_points) for c in curves)
if not keys:raise ValueError('生成动作没有关键帧')
# Keep source shader graphs. Eevee supports toon Shader to RGB graphs.
if any(n.type=='SHADER_TO_RGB' for m in bpy.data.materials if m.use_nodes for n in m.node_tree.nodes):scene.render.engine='BLENDER_EEVEE'
for screen in bpy.data.screens:
 for area in screen.areas:
  if area.type=='VIEW_3D' and scene.render.engine!='BLENDER_WORKBENCH':
   area.spaces.active.shading.type='MATERIAL'
   area.spaces.active.shading.use_scene_world=False;area.spaces.active.shading.use_scene_lights=False
warnings=sorted(session.warnings)
for image in bpy.data.images:
 if image.source=='FILE' and not image.packed_file:
  try:image.pack()
  except Exception:warnings.append('未能打包贴图：'+image.name)
if not job.get('keep_camera') or not scene.camera:
 points=[];mapping=a.pose_tools.mapping_for(s)
 for frame in {session.frames[0],session.frames[len(session.frames)//2],session.frames[-1]}:
  scene.frame_set(frame)
  for name in set(mapping.values()):
   bone=rig.pose.bones.get(name)
   if bone:points.extend((rig.matrix_world@bone.head,rig.matrix_world@bone.tail))
 deps=bpy.context.evaluated_depsgraph_get()
 for ob in scene.objects:
  if ob.type=='MESH' and not ob.hide_render and any(m.type=='ARMATURE' and m.object==rig for m in ob.modifiers):
   ev=ob.evaluated_get(deps);points.extend(ev.matrix_world@Vector(c) for c in ev.bound_box)
 if not points:raise ValueError('找不到角色边界，无法创建预览相机')
 low=Vector([min(p[i] for p in points) for i in range(3)]);high=Vector([max(p[i] for p in points) for i in range(3)]);center=(low+high)*.5
 data=bpy.data.cameras.new('自动全身预览相机');cam=bpy.data.objects.new(data.name,data);scene.collection.objects.link(cam);cam.location=center+Vector((0,-max(5,(high-low).length*3),0));cam.rotation_euler=(center-cam.location).to_track_quat('-Z','Y').to_euler();data.type='ORTHO'
 data.sensor_fit='VERTICAL'
 for marker in scene.timeline_markers:marker.camera=None
 aspect=scene.render.resolution_x/max(1,scene.render.resolution_y);data.ortho_scale=max(1.,high.z-low.z,(high.x-low.x)/aspect)*1.25;scene.camera=cam
# Keep solve records and review report with the deliverable, not user runtime.
local_bake=out/'bake';shutil.copytree(session.folder,local_bake,dirs_exist_ok=True)
s.video_report=str(local_bake/'report.json');rig.animation_data.action['sp_video_report']=s.video_report
scene.frame_set(session.frames[len(session.frames)//2]);scene.tool_settings.use_keyframe_insert_auto=False
if not job.get('keep_camera'):scene.camera=cam
bpy.ops.wm.save_as_mainfile(filepath=str(out/'project.blend'))
if job['render']:
 scene.render.resolution_percentage=50;scene.render.filepath=str(out/'preview.png')
 if scene.render.engine=='CYCLES':scene.cycles.samples=16
 bpy.ops.render.render(write_still=True)
result=dict(success=False,stage='awaiting_reopen',kind=job['kind'],rig=rig.name,action=rig.animation_data.action.name,keys=keys,curves=len(curves),frame_start=session.frames[0],frame_end=session.frames[-1],sample_count=len(session.frames),addon_version=list(a.bl_info['version']),warnings=warnings,project=str(out/'project.blend'),report=s.video_report)
(out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print('PROJECT_BUILT',flush=True)
