"""Transport SAM/MHR's anatomical sole frame, not ankle-to-toe slope.

MHR rest coordinates are Y-up. The ankle is above the ball even in a neutral
foot; its segment inclination is not the inclination of the supporting sole.
"""
import numpy as np
from . import orientation_math as om
from .foot_data import LOCAL_SOURCE

def native_frame(data,side):
 source=data.get('full_skeleton',{})
 if source.get('source')!='sam3d_mhr' or source.get('rest_axes')!='MHR_native':return None
 bones=source['bones'];name=source['role_map'].get('foot.'+side);ball=('l_' if side=='L' else 'r_')+'ball'
 if name not in bones or ball not in bones:return None
 bone=bones[name];forward=np.array(bones[ball]['rest_head'])-np.array(bone['rest_head']);up=np.array([0.,1.,0.])
 forward-=up*np.dot(forward,up)
 rest=om.frame(np.cross(forward,up),forward)
 rotation=om.proper_rotation(bone['pose_rotation'])@om.proper_rotation(bone['rest_rotation']).T
 return om.proper_rotation(rotation@rest)

def frame_for(data,side,observation):
 source=observation.get('source')
 if source=='user_foot_direction_choice':
  return om.proper_rotation(observation['frame']),source
 frame=native_frame(data,side)
 if frame is not None:return frame,'sam3d_native_sole_rotation'
 return om.proper_rotation(observation['frame']),observation['source']


def visible_shoe_axis(data, side):
 """Heel-to-toe image evidence, accepted only with a matching native limb."""
 wholebody=data.get('wholebody',())
 if len(wholebody)!=133:return None
 ids=(15,17,18,19) if side=='L' else (16,20,21,22)
 rows=[wholebody[i] for i in ids]
 if any(row.get('confidence',0)<.5 or row.get('outside_frame') for row in rows):return None
 aspect=data['image']['height']/data['image']['width']
 ankle,big,small,heel=np.array([[row['x'],row['y']*aspect] for row in rows])
 if any(not (0<=row['x']<=1 and 0<=row['y']<=1) for row in rows):return None
 axis=(big+small)*.5-heel;span=np.linalg.norm(axis)
 if span<.035 or np.linalg.norm(ankle-heel)>.65*span:return None
 source=data.get('full_skeleton',{});bones=source.get('bones',{})
 foot=bones.get(source.get('role_map',{}).get('foot.'+side))
 ball=bones.get(('l_' if side=='L' else 'r_')+'ball')
 if not foot or not ball:return None
 native=(np.asarray(ball['pose_head'])-foot['pose_head'])[[0,2]]*[1.,-1.]
 if np.linalg.norm(native)<1e-6 or axis@native < .7*span*np.linalg.norm(native):return None
 return axis


def reviewed_shoe_axis(data, side, observation):
 """The local crop supplies a screen direction, not a new sole normal."""
 if observation.get('source')!=LOCAL_SOURCE:return None
 points=np.asarray(observation.get('local_2d'),float)
 if points.shape!=(4,2) or not np.isfinite(points).all():return None
 width,height=data['image']['width'],data['image']['height']
 if np.any(points<0) or np.any(points>[width,height]):return None
 ankle,big,small,heel=points/width
 axis=(big+small)*.5-heel;span=np.linalg.norm(axis)
 if span<.035 or np.linalg.norm(ankle-heel)>.75*span:return None
 reference=visible_shoe_axis(data,side)
 if reference is None or axis@reference<.8*span*np.linalg.norm(reference):return None
 return axis
