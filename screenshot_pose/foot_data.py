"""Conservative 2-D foot refinement; depth remains the original estimate."""
import copy,numpy as np
from . import orientation_math as om
INDICES={'L':(11,13,15,17,18,19),'R':(12,14,16,20,21,22)}
LOCAL_SOURCE='local_rtmw_2d_original_depth'

def image_points(data):
 raw=data['wholebody'];raw=raw.get('keypoints',raw.get('points')) if isinstance(raw,dict) else raw
 if raw and isinstance(raw[0],dict):return np.array([[p['x']*data['image']['width'],p['y']*data['image']['height']] for p in raw])
 return np.asarray(raw,float)

def crop_box(xy,side,size,margin):
 p=np.asarray(xy)[list(INDICES[side])];lo=p.min(0);hi=p.max(0)
 span=np.maximum(hi-lo,[size[0]*.18,size[1]*.22]);center=(lo+hi)*.5
 lo=np.maximum(0,center-span*(.5+margin));hi=np.minimum(size,center+span*(.5+margin))
 return [int(lo[0]),int(lo[1]),int(hi[0]),int(hi[1])]

def observation(data,side,xy,scores):
 """Reject ambiguous identity using hip/knee/ankle, not screen-left/right."""
 base=image_points(data)
 # Callers normalize cached wholebody structures before entering this helper.
 ids=INDICES[side];other=INDICES['R' if side=='L' else 'L'];xy=np.asarray(xy,float);scores=np.asarray(scores,float)
 if xy.shape!=(133,2) or scores.shape!=(133,) or not np.isfinite(xy).all() or not np.isfinite(scores).all():return None,'局部点或分数无效'
 length=max(np.linalg.norm(base[ids[0]]-base[ids[1]])+np.linalg.norm(base[ids[1]]-base[ids[2]]),1.)
 costs=[np.mean(np.linalg.norm(xy[list(candidate[:3])]-base[list(ids[:3])],axis=1))/length for candidate in (ids,other)]
 winner=int(np.argmin(costs));chosen=(ids,other)[winner]
 if costs[winner]>.3 or abs(costs[0]-costs[1])<.06:return None,'左右归属或腿链匹配不明确'
 points=xy[list(chosen[2:])];raw_score=float(np.min(scores[list(chosen[2:])]))
 conf=float(np.clip(raw_score,0,1))
 if conf<.5:return None,'脚踝、脚跟或脚尖置信度不足'
 if np.linalg.norm(points[1:3].mean(0)-points[3])<length*.015:return None,'脚尖与脚跟过近，无法确定朝向'
 if np.linalg.norm(points[0]-base[ids[2]])>length*.15:return None,'脚踝偏离原腿链过大'
 return dict(xy=points.tolist(),confidence=conf,raw_score=raw_score,assignment_cost=costs[winner],semantic_output='same' if winner==0 else 'swapped_by_leg_chain'),''

def refine(data,side,observations):
 result=copy.deepcopy(data);old=data.get('orientation',{}).get('feet',{}).get(side)
 if not old or not old.get('points'):return result,dict(adopted=False,reason='缺少原脚部三维参考，保留原结果')
 if len(observations)<2:return result,dict(adopted=False,reason='不足两次一致的局部观测')
 a,b=observations[:2];points=np.asarray(a['xy']);second=np.asarray(b['xy'])
 span=max(np.linalg.norm(points[1:3].mean(0)-points[3]),1.)
 if np.max(np.linalg.norm(points-second,axis=1))/span>.35:return result,dict(adopted=False,reason='不同裁剪尺度的脚点不一致')
 conf=min(a['confidence'],b['confidence'])
 if conf<old.get('confidence',0)+.03:return result,dict(adopted=False,reason='局部观测没有明确提高置信度')
 names=['ankle','big_toe','small_toe','heel'];world=np.array([old['points'][n] for n in names],float)
 base=image_points(data)[list(INDICES[side][2:])]
 world_scale=np.linalg.norm((world[1:3].mean(0)-world[3])[[0,2]])/max(np.linalg.norm(base[1:3].mean(0)-base[3]),1.)
 if world_scale<1e-7:return result,dict(adopted=False,reason='原脚部投影近乎重合，不能可靠恢复朝向')
 xy=(points+second)*.5
 world[:,0]=world[0,0]+(xy[:,0]-xy[0,0])*world_scale
 world[:,2]=world[0,2]-(xy[:,1]-xy[0,1])*world_scale
 try:frame=om.frame((world[1]-world[2])*(1 if side=='L' else -1),world[1:3].mean(0)-world[3])
 except ValueError:return result,dict(adopted=False,reason='局部鞋跟和鞋尖无法确定稳定朝向')
 result['orientation']['feet'][side]=dict(old,frame=frame.tolist(),points=dict(zip(names,world.tolist())),confidence=conf,confidence_method='clipped_model_score_not_calibrated_3d_probability',source=LOCAL_SOURCE,local_2d=xy.tolist(),depth_method='retained_original_not_new_3d')
 return result,dict(adopted=True,confidence=conf,depth='沿用原深度；不代表新的三维脚部识别',observations=observations)

def candidates(frame):
 frame=om.proper_rotation(frame)
 return [dict(name=label,frame=(frame@rotation).tolist()) for label,rotation in [('原脚向',np.eye(3)),('反向鞋尖',np.diag([-1,-1,1])),('翻转足底',np.diag([-1,1,-1]))]]
