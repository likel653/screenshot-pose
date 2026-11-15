"""Optional uncalibrated alignment of SAM hypotheses, not triangulation."""
import copy,math,json
from pathlib import Path
import numpy as np
try:
 from . import pose_data,sam_data,orientation_math as om
except ImportError:
 import pose_data,sam_data,orientation_math as om

ANCHORS=list(sam_data.BODY.values())+['c_neck','c_head','c_spine3','root']
def skeleton(data):
 pose_data.validate(data);s=data.get('full_skeleton',{})
 if s.get('source')!='sam3d_mhr' or s.get('pose_axes')!='X_right_Y_away_Z_up' or len(s.get('bones',{}))!=127:raise ValueError('多图辅助需要 SAM 的完整 127 关节结果')
 for n,v in s['bones'].items():
  if np.asarray(v['pose_head']).shape!=(3,) or not np.isfinite(v['pose_head']).all():raise ValueError('参考图骨架位置无效')
  om.proper_rotation(v['pose_rotation'])
 if any(n not in s['bones'] for n in ANCHORS):raise ValueError('参考图缺少同名身体关节')
 return s
def confidence(data,name):
 for i,n in sam_data.BODY.items():
  if n==name:
   p=data['landmarks'][i];return min(p.get('visibility',1),p.get('presence',1))
 return float(np.mean([min(data['landmarks'][i].get('visibility',1),data['landmarks'][i].get('presence',1)) for i in (11,12,23,24)]))
def similarity(source,target,weights):
 x=np.asarray(source,float);y=np.asarray(target,float);w=np.asarray(weights,float)
 if x.shape!=y.shape or x.ndim!=2 or x.shape[1]!=3 or w.shape!=(len(x),) or not all(np.isfinite(v).all() for v in (x,y,w)) or np.any(w<0) or w.sum()<1e-8:raise ValueError('多图对齐数据无效')
 w=w/w.sum();mx=(x*w[:,None]).sum(0);my=(y*w[:,None]).sum(0);a=x-mx;b=y-my
 u,d,vt=np.linalg.svd((a*w[:,None]).T@b);sign=np.linalg.det(u@vt)
 correction=np.array([1.,1.,1. if sign>0 else -1.]);r=(u@np.diag(correction)@vt).T
 variance=float(np.sum(w*np.sum(a*a,axis=1)))
 if variance<1e-9 or d[1]<1e-9:raise ValueError('可用关节退化，无法估计视角')
 scale=float(np.sum(d*correction)/variance);translation=my-scale*r@mx
 return r,scale,translation,d,sign
def align(primary,reference):
 p=skeleton(primary)['bones'];q=skeleton(reference)['bones']
 if set(p)!=set(q):raise ValueError('参考图骨架命名不一致')
 x=np.array([q[n]['pose_head'] for n in ANCHORS]);y=np.array([p[n]['pose_head'] for n in ANCHORS])
 weights=np.array([min(confidence(primary,n),confidence(reference,n)) for n in ANCHORS]);mask=weights>=.35
 if mask.sum()<8:raise ValueError('可靠的对应身体关节不足，无法对齐')
 x=x[mask];y=y[mask];initial=weights[mask];weights=initial.copy()
 torso=max(np.linalg.norm((np.array(p['l_uparm']['pose_head'])+p['r_uparm']['pose_head'])/2-(np.array(p['l_upleg']['pose_head'])+p['r_upleg']['pose_head'])/2),.1)
 for _ in range(6):
  r,scale,t,d,sign=similarity(x,y,weights)
  residual=np.linalg.norm(scale*x@r.T+t-y,axis=1)
  weights=initial*np.minimum(1.,max(float(np.median(residual))*1.5,.025*torso)/np.maximum(residual,1e-9))
 normalized=residual/torso;angle=math.degrees(math.acos(float(np.clip((np.trace(r)-1)/2,-1,1))))
 if sign<0 and d[-1]/max(d[0],1e-9)>.03:raise ValueError('参考图可能镜像，或姿势的左右关系不一致')
 if not .6<scale<1.7:raise ValueError('三维预测尺度差异过大，请尝试其他视角并确认角色一致')
 median=float(np.median(normalized));p90=float(np.quantile(normalized,.9))
 if median>.18 or p90>.45:raise ValueError('姿势不一致或三维预测差异过大，未采用此参考图')
 if angle<8:raise ValueError('估计视角与主图过于接近，未增加融合候选')
 return dict(rotation=r.tolist(),scale=scale,translation=t.tolist(),view_angle_deg=angle,median_torso_error=median,p90_torso_error=p90,
             quality=float(np.clip(1-median/.25,.15,1)),confidence_method='heuristic_not_model_probability')
def mean_rotation(matrices,weights):
 m=sum(float(w)*np.asarray(r) for r,w in zip(matrices,weights));u,d,vt=np.linalg.svd(m)
 if d[-1]<1e-8:raise ValueError('参考旋转相互抵消，无法可靠融合')
 return u@np.diag([1,1,1 if np.linalg.det(u@vt)>0 else -1])@vt
def fuse(primary,references,strength=.35):
 if not 0<strength<=.6:raise ValueError('辅助强度应介于 0 与 0.6 之间')
 source=skeleton(primary);names=list(source['bones']);bones=source['bones'];accepted=[];rejected=[]
 for entry in references:
  try:
   alignment=align(primary,entry['data']);accepted.append(dict(entry=entry,alignment=alignment))
  except (ValueError,KeyError,TypeError,np.linalg.LinAlgError) as exc:rejected.append(dict(image=entry.get('image',''),reason=str(exc)))
 result=copy.deepcopy(primary)
 report=dict(method='uncalibrated_SAM_hypothesis_alignment_not_triangulation',accepted=len(accepted),rejected=rejected,strength=strength,primary_preserved=True,primary_head_hand_refinement_preserved=True,
             references=[dict(image=v['entry'].get('image',''),**v['alignment']) for v in accepted])
 if not accepted:result['multiview']=report;return result
 points={};rotations={}
 for name in names:
  primary_weight=.6+.4*confidence(primary,name);weights=[primary_weight];ps=[np.array(bones[name]['pose_head'])];rs=[np.array(bones[name]['pose_rotation'])]
  for item in accepted:
   ref=item['entry']['data'];b=ref['full_skeleton']['bones'][name];fit=item['alignment'];r=np.array(fit['rotation'])
   weight=strength/len(accepted)*fit['quality']*(.4+.6*confidence(ref,name))
   weights.append(weight);ps.append(fit['scale']*r@np.array(b['pose_head'])+fit['translation']);rs.append(r@np.array(b['pose_rotation']))
  points[name]=np.average(ps,axis=0,weights=weights);rotations[name]=mean_rotation(rs,weights)
 # Do not shorten bones by averaging differently oriented segments. Rebuild
 # each link with the primary prediction's length, keeping its original root.
 # Auxiliary occluded fingers and faces must not dilute primary local poses.
 # Carry their original parent-relative rotations/offsets along with the arm/neck.
 frozen={'l_wrist_twist','r_wrist_twist','c_head'}
 while True:
  expanded=frozen|{n for n in names if bones[n].get('parent') in frozen}
  if expanded==frozen:break
  frozen=expanded
 report['primary_local_joints_preserved']=len(frozen)
 normalized={};pending=set(names)
 while pending:
  done=[]
  for name in pending:
   parent=bones[name].get('parent')
   if parent and parent not in normalized:continue
   if not parent:normalized[name]=np.array(bones[name]['pose_head'])
   elif name in frozen:
    delta=rotations[parent]@np.array(bones[parent]['pose_rotation']).T
    rotations[name]=delta@np.array(bones[name]['pose_rotation'])
    normalized[name]=normalized[parent]+delta@(np.array(bones[name]['pose_head'])-bones[parent]['pose_head'])
   else:
    original=np.array(bones[name]['pose_head'])-bones[parent]['pose_head'];length=np.linalg.norm(original);direction=points[name]-points[parent];norm=np.linalg.norm(direction)
    normalized[name]=normalized[parent]+(direction/norm*length if norm>1e-9 else original)
   done.append(name)
  if not done:raise ValueError('主图骨架层级包含循环或缺失父级')
  pending.difference_update(done)
 geometry_names=json.loads(Path(sam_data.__file__).with_name('sam_geometry.json').read_text())['names']
 if set(geometry_names)!=set(names):raise ValueError('SAM 骨架命名不兼容')
 raw=dict(joints=np.array([normalized[n]@om.C_DOWN for n in geometry_names])[None],
          joint_rotations=np.array([om.C_FACE.T@rotations[n] for n in geometry_names])[None])
 converted=sam_data.convert(raw,primary);new_skeleton=converted['full_skeleton'];new_skeleton['fusion']='uncalibrated_multiview_estimate'
 worlds=converted['world_landmarks']
 result['body_candidates']=[c for c in result.get('body_candidates',[]) if c['source']!='sam3d_multiview_depth']
 if not result['body_candidates']:result['body_candidates']=[dict(source=primary.get('depth_source','sam3d_mhr_depth'),world_landmarks=copy.deepcopy(primary['world_landmarks']))]
 result['body_candidates'].append(dict(source='sam3d_multiview_depth',world_landmarks=worlds,manual_only=True))
 result['full_skeleton_candidates']=[c for c in result.get('full_skeleton_candidates',[]) if c.get('source')!='sam3d_multiview']
 result['full_skeleton_candidates'].append(dict(source='sam3d_multiview',name='多图辅助 · 完整骨架',skeleton=new_skeleton,world_landmarks=worlds))
 report['source_joints']=len(new_skeleton['bones']);result['multiview']=report
 return pose_data.validate(result)
