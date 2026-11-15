"""Small display helpers without Blender dependencies."""
import unicodedata

def wrap_text(text,width=32):
 lines=[];line='';used=0
 for char in str(text):
  cost=2 if unicodedata.east_asian_width(char) in 'WF' else 1
  if char=='\n' or (line and used+cost>width):
   lines.append(line);line='';used=0
   if char=='\n':continue
  line+=char;used+=cost
 if line:lines.append(line)
 return lines

def body_source(result):
 selection=result.get('depth_selection',{})
 if selection.get('chosen'):
  return {'sam3d_mhr_depth':'SAM 三维身体','rtmw3d_relative_depth':'RTMW3D 身体',
          'mediapipe_estimate':'MediaPipe 深度候选','multihmr_anny_depth':'Anny 三维身体',
          'flat_fallback':'平面备用候选'}.get(selection['chosen'],selection['chosen'])
 if result.get('full_source_comparison',{}).get('chosen')=='full_source_rotation':return '完整源骨架旋转'
 if result.get('full_rig'):return '完整源骨架旋转'
 return '关节点拟合'
