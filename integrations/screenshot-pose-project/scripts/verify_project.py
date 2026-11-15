import bpy,json,sys,addon_utils
from pathlib import Path
job=json.loads(Path(sys.argv[sys.argv.index('--')+1]).read_text(encoding='utf8'));out=Path(job['output']);sys.path.insert(0,job['project_root']);addon_utils.enable('screenshot_pose',default_set=False)
import screenshot_pose as a
r=json.loads((out/'result.json').read_text(encoding='utf8'));rig=bpy.data.objects[r['rig']];action=rig.animation_data.action;assert action and action.name==r['action']
curves=list(a.animation_preview.curves(action));assert sum(len(c.keyframe_points) for c in curves)==r['keys'] and r['keys']>0
assert all(b.rotation_mode=='QUATERNION' for b in rig.pose.bones if any(c.data_path==b.path_from_id()+'.rotation_quaternion' for c in curves))
assert Path(action['sp_video_source']).is_file()
for f in (r['frame_start'],r['frame_end']):
 bpy.context.scene.frame_set(f);rig.evaluated_get(bpy.context.evaluated_depsgraph_get())
(out/'reopen_check.json').write_text(json.dumps(dict(passed=True,keys=r['keys'],action=action.name,rig=rig.name),ensure_ascii=False,indent=2),encoding='utf8');print('REOPEN_PASS',flush=True)
