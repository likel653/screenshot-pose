"""Non-keyed poses must override animated FK/IK switches as well as rotations.

Only a private Action copy is muted. Original animation is restored on frame
change, restore, load or shutdown. The user's Action curves are never edited.
"""
import json
import bpy
from . import retarget
KEY='sp_animation_preview'

def curves(action):
 if not action:return []
 if hasattr(action,'fcurves'):return list(action.fcurves)
 return [fc for layer in action.layers for strip in layer.strips for bag in strip.channelbags for fc in bag.fcurves]

def end(rig,restore=True):
 raw=rig.get(KEY)
 if not raw:return
 state=json.loads(raw);preview=bpy.data.actions.get(state['preview'])
 # Do not replace an Action the user has explicitly changed since preview.
 current=rig.animation_data.action if rig.animation_data else None
 if restore and current==preview:
  retarget.restore_pose(rig,state['snapshot'],update=False)
  for t in rig.animation_data.nla_tracks:
   if t.name in state['tracks']:t.mute=state['tracks'][t.name]
 name=state['snapshot'].get('action');original=bpy.data.actions.get(name) if name else None
 if original and restore:original.use_fake_user=state['snapshot'].get('action_fake_user',False)
 del rig[KEY]
 # Keep the temporary datablock available for Blender undo / candidate backups.

def begin(rig,snapshot,solved,names,tracks,frame):
 original=bpy.data.actions.get(snapshot['action']) if snapshot.get('action') else None
 if not original and not any(not mute for mute in tracks.values()):
  retarget.restore_pose(rig,solved);return False
 action=original.copy() if original else bpy.data.actions.new(rig.name+' Preview')
 action.name=rig.name+' · 临时姿势（未打帧）'
 if original:original.use_fake_user=True
 prefixes=tuple(rig.pose.bones[n].path_from_id()+'.' for n in names if n in rig.pose.bones)
 props=tuple(rig.pose.bones[n].path_from_id()+'[' for n in names if n in rig.pose.bones)
 muted=[]
 for fc in curves(action):
  if (fc.data_path.startswith(prefixes+props) or fc.data_path in {retarget.object_property_path(k) for k in solved.get('object_properties',{})}) and not fc.mute:fc.mute=True;muted.append(fc.data_path)
 action['sp_preview_muted_paths']=json.dumps(sorted(set(muted)))
 rig[KEY]=json.dumps(dict(snapshot=snapshot,tracks=tracks,preview=action.name,frame=frame))
 solved=dict(solved,action=action.name,action_fake_user=False)
 retarget.restore_pose(rig,solved)
 for t in rig.animation_data.nla_tracks:t.mute=True
 retarget._update(rig)
 return True

def before_frame(scene,*args):
 pending={}
 if hasattr(scene,'screenshot_pose') and scene.screenshot_pose.candidate_state:
  pending=json.loads(scene.screenshot_pose.candidate_state)
 for rig in scene.objects:
  if rig.type=='ARMATURE' and KEY in rig:
   if pending.get('target')==rig.name:continue
   if json.loads(rig[KEY])['frame']!=scene.frame_current:end(rig)

def cleanup():
 for rig in bpy.data.objects:
  if rig.type=='ARMATURE' and KEY in rig:end(rig)

def prepare_copy(rig):
 """A subsequent explicit keyed edit owns the new copy, not the old preview."""
 if KEY in rig:end(rig,restore=False)

def unmute_keyed(rig,paths):
 action=rig.animation_data.action if rig.animation_data else None
 if not action:return
 muted=set(json.loads(action.get('sp_preview_muted_paths','[]')))
 for fc in curves(action):
  if fc.data_path in paths and fc.data_path in muted:fc.mute=False
