"""Describe real available predictions; never fabricate alternate poses."""
import copy
LABELS={'sam3d_mhr_depth':'SAM 三维深度','rtmw3d_relative_depth':'RTMW3D 三维深度','mediapipe_estimate':'MediaPipe 深度','flat_fallback':'平面深度','multihmr_anny_depth':'Anny 三维深度','sam3d_multiview_depth':'多图辅助 · 三维深度'}
def available(data,edited=False):
 result=[]
 if data.get('full_skeleton') and not data.get('edited_indices') and not edited:
  source=data['full_skeleton'].get('source','anny')
  result.append(dict(id='FULL',name=('SAM' if source=='sam3d_mhr' else 'Anny')+' 完整骨架',kind='FULL',source=source))
 if not data.get('edited_indices') and not edited:
  for index,item in enumerate(data.get('full_skeleton_candidates',[])):
   result.append(dict(id='FULL_AUX_'+str(index),name=item['name'],kind='FULL',source=item['source'],skeleton_index=index))
 seen=set()
 for index,candidate in enumerate(data.get('body_candidates') or [dict(source=data.get('depth_source',data.get('source','default')),world_landmarks=data['world_landmarks'])]):
  source=candidate['source']
  if source in seen:continue
  seen.add(source);result.append(dict(id='DEPTH_'+str(index),name=LABELS.get(source,source),kind='DEPTH',source=source,index=index))
 return result
def document(data,spec):
 result=copy.deepcopy(data)
 if spec['kind']=='FULL' and 'skeleton_index' in spec:
  item=data['full_skeleton_candidates'][spec['skeleton_index']]
  result['full_skeleton']=copy.deepcopy(item['skeleton']);result['world_landmarks']=copy.deepcopy(item['world_landmarks'])
 if spec['kind']=='DEPTH':
  candidates=data.get('body_candidates') or [dict(world_landmarks=data['world_landmarks'])]
  result['world_landmarks']=copy.deepcopy(candidates[spec['index']]['world_landmarks'])
  result['depth_source']=spec['source']
 result['body_candidates']=[]
 return result
