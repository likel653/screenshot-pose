"""Manual 21-point corrections. Estimated depth is explicit, never new inference."""
import copy,math
from pathlib import Path
import numpy as np
try:from . import orientation_math as om,pose_data,worker
except ImportError:import orientation_math as om,pose_data,worker
FINGERS=('thumb','index','middle','ring','pinky')
NAMES=['手腕']+[f'{f} · {j}' for f in ('拇指','食指','中指','无名指','小指') for j in ('根部','第二节','第三节','指尖')]
EDGES=[(0,1+4*i) for i in range(5)]+[(1+4*i+j,2+4*i+j) for i in range(5) for j in range(3)]

def initialize(data,side,character_seed=None):
 if side not in ('L','R'):raise ValueError('请选择左手或右手')
 pose_data.validate(data);hd=copy.deepcopy(data.get('orientation',{}).get('hands',{}).get(side,{}));b=data.get('wholebody',[]);start=91 if side=='L' else 112
 reference=bool(hd.get('reference_landmarks') and hd.get('reference_image'))
 image=hd['reference_image'] if reference else data['image']['path'];im=worker.prepare_image(image);w,h=im.size
 xy=np.array(hd.get('reference_landmarks') if reference else hd.get('landmarks',[]),float)
 initialization='已有手部二维点'
 if xy.shape!=(21,2) or not np.isfinite(xy).all():
  reference=False;image=data['image']['path'];im=worker.prepare_image(image);w,h=im.size
  xy=np.array([[v['x'],v['y']] for v in b[start:start+21]],float) if len(b)==133 else np.empty((0,2));initialization='全身模型手点（需核对）'
 if xy.shape!=(21,2) or not np.isfinite(xy).all() or np.linalg.norm((xy[9]-xy[0])*[w,h])<3:
  wi=15 if side=='L' else 16;wrist=np.array([data['landmarks'][wi]['x'],data['landmarks'][wi]['y']]);xy=np.zeros((21,2));xy[0]=wrist
  for i in range(5):
   base=1+4*i;xy[base]=wrist+[(2-i)*.012,-.035]
   for j in range(1,4):xy[base+j]=xy[base]+[0,-j*.017]
  initialization='手工模板（不是识别结果）';image=data['image']['path'];im=worker.prepare_image(image);w,h=im.size;reference=False
 r=om.proper_rotation(hd.get('view_rotation',np.eye(3))) if reference else np.eye(3)
 seeded=character_seed is not None
 points=np.asarray(character_seed['world'] if seeded else hd.get('world',[]),float)
 valid=points.shape==(21,3) and np.isfinite(points).all()
 if valid:
  points=(points-points[0])@r
  palm=np.linalg.norm(points[9]);lengths=np.array([np.linalg.norm(points[2+4*i+j]-points[1+4*i+j]) for i in range(5) for j in range(3)])
  valid=palm>1e-5 and min(lengths)>.002 and max(lengths)<palm*1.5
 pixels=xy*np.array([w,h]);palm_pixels=max(float(np.linalg.norm(pixels[9]-pixels[0])),6.)
 if valid:scale=float(np.clip(np.linalg.norm(points[9])/palm_pixels,.00005,.01));depth_source='沿用原三维估计，可手动改深度'
 else:
  scale=.07/palm_pixels;points=np.column_stack(((pixels[:,0]-pixels[0,0])*scale,np.zeros(21),-(pixels[:,1]-pixels[0,1])*scale));depth_source='平面起始深度，需要手工校正'
  # Missing/unstable model fingers receive ordinary anatomical length priors.
  for i,segments in enumerate(((.028,.023,.02),(.032,.022,.018),(.036,.024,.02),(.033,.023,.018),(.025,.017,.015))):
   base=1+4*i
   for j,length in enumerate(segments):
    v=points[base+j+1]-points[base+j];n=np.linalg.norm(v)
    points[base+j+1]=points[base+j]+(v/n if n>1e-8 else np.array([0,0,1.]))*length
 if seeded and valid:
  wrist_xy=xy[0].copy();xy=wrist_xy+points[:,[0,2]]*np.array([1.,-1.])/scale/[w,h]
  initialization='当前角色手指姿势（编辑起点）';depth_source='沿用当前姿势的估计深度，源掌长已归一化'
 try:
  selected_frame=hd.get('frame',character_seed.get('frame') if seeded else None)
  frame=om.proper_rotation(selected_frame) if selected_frame is not None else om.palm_frame(points);frame_raw=copy.deepcopy(selected_frame) if selected_frame is not None else frame.tolist()
 except ValueError:frame_raw=[[1.,0,0],[0,0,-1.],[0,1.,0]]
 return dict(side=side,image=str(Path(image).resolve()),width=w,height=h,xy=xy.tolist(),original_xy=xy.tolist(),depth=points[:,1].tolist(),original_depth=points[:,1].tolist(),world_seed=points.tolist(),view_rotation=r.tolist(),primary_frame=frame_raw,meters_per_pixel=scale,changed=[],initialization=initialization,depth_source=depth_source)

def move(state,index,x,y,depth=None):
 if not 0<=index<21 or not np.isfinite([x,y]).all() or not 0<=x<=1 or not 0<=y<=1:raise ValueError('手部关节点位置无效')
 if index==0:raise ValueError('手腕锚点固定；请使用身体关节编辑器调整手腕位置')
 if depth is not None:
  if not math.isfinite(depth) or abs(depth)>.5:raise ValueError('手部深度请输入 -0.5 至 0.5 米')
 state['xy'][index]=[float(x),float(y)]
 if depth is not None:
  state['depth'][index]=float(depth)
 state['changed']=sorted(set(state['changed'])|{index})

def _rotation(a,b):
 a=a/max(np.linalg.norm(a),1e-12);b=b/max(np.linalg.norm(b),1e-12);v=np.cross(a,b);c=float(np.clip(np.dot(a,b),-1,1));s=np.linalg.norm(v)
 if s<1e-9:
  if c>0:return np.eye(3)
  axis=np.cross(a,[1.,0,0])
  if np.linalg.norm(axis)<1e-6:axis=np.cross(a,[0,1.,0])
  axis/=np.linalg.norm(axis);return 2*np.outer(axis,axis)-np.eye(3)
 k=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
 return np.eye(3)+k+k@k*((1-c)/(s*s))

def solve_chain(initial,goals,weights):
 """CCD with interior targets; preserves all three lengths, clamps joint bends."""
 q=np.array(initial,float).copy();goals=np.asarray(goals,float);lengths=np.linalg.norm(np.diff(q,axis=0),axis=1)
 if min(lengths)<1e-8:raise ValueError('手指骨长无效')
 q+=goals[0]-q[0]
 for _ in range(55):
  old=q.copy()
  for joint in (2,1,0):
   ids=np.arange(joint+1,4);a=q[ids]-q[joint];b=goals[ids]-q[joint];w=np.asarray(weights)[ids]
   u,_,vt=np.linalg.svd((a*w[:,None]).T@b);r=vt.T@np.diag([1,1,1 if np.linalg.det(vt.T@u.T)>0 else -1])@u.T
   q[ids]=(q[ids]-q[joint])@r.T+q[joint]
  for j,limit in ((1,math.radians(140)),(2,math.radians(110))):
   previous=q[j]-q[j-1];along=q[j+1]-q[j];angle=math.acos(float(np.clip(np.dot(previous,along)/np.linalg.norm(previous)/np.linalg.norm(along),-1,1)))
   if angle>limit:
    a=previous/np.linalg.norm(previous);b=along/np.linalg.norm(along);axis=b-a*np.dot(a,b)
    if np.linalg.norm(axis)<1e-8:axis=np.cross(a,[1.,0,0])
    if np.linalg.norm(axis)<1e-8:axis=np.cross(a,[0,1.,0])
    axis/=np.linalg.norm(axis);desired=(a*math.cos(limit)+axis*math.sin(limit))*lengths[j];r=_rotation(along,desired)
    q[j+1:]=(q[j+1:]-q[j])@r.T+q[j]
  if np.max(abs(q-old))<1e-7:break
 return q

def document(data,state,apply_on_return=False):
 if not state['changed']:raise ValueError('尚未修改任何手部关节点')
 result=copy.deepcopy(data);side=state['side'];seed=np.asarray(state['world_seed'],float);goals=seed.copy();xy=np.asarray(state['xy']);original=np.asarray(state['original_xy']);delta=(xy-original)*[state['width'],state['height']]*state['meters_per_pixel']
 goals[:,0]+=delta[:,0];goals[:,2]-=delta[:,1];goals[:,1]=state['depth'];solved=seed.copy();fingers=sorted({FINGERS[(i-1)//4] for i in state['changed'] if i})
 for finger in fingers:
  i=FINGERS.index(finger);base=1+4*i;weights=[20.]+[12. if base+j in state['changed'] else .15 for j in (1,2,3)]
  solved[base:base+4]=solve_chain(seed[base:base+4],goals[base:base+4],weights)
 r=np.asarray(state['view_rotation']);world=solved@r.T
 original_world=np.asarray(data.get('orientation',{}).get('hands',{}).get(side,{}).get('world',[]),float)
 if original_world.shape==(21,3) and np.isfinite(original_world).all():
  world+=original_world[0];selected={1+4*FINGERS.index(f)+j for f in fingers for j in range(4)}
  for i in range(21):
   if i not in selected:world[i]=original_world[i]
 details=result.setdefault('orientation',{});details.setdefault('schema','screenshot_pose/orientation_v1');details.setdefault('camera_axes','X_right_Y_away_Z_up');hands=details.setdefault('hands',{})
 history=result.setdefault('hand_edit_history',{}).setdefault(side,[]);history.append(copy.deepcopy(hands.get(side)));result['hand_edit_history'][side]=history[-8:]
 hd=hands.setdefault(side,{})
 previous=set(hd.get('manual_fingers',[]));hd.update(frame=state['primary_frame'],world=world.tolist(),source='manual_hand_landmarks',confidence=hd.get('confidence',0.),manual_point_confidence=1.,confidence_method='original_palm_quality_manual_fingers_separate',manual_fingers=sorted(previous|set(fingers)),reference_landmarks=state['xy'],reference_image=state['image'],view_rotation=state['view_rotation'],alternatives=[])
 hd.pop('bone_frames',None);hd.pop('landmarks',None)
 residual=float(np.sqrt(np.mean(np.sum((solved[state['changed']]-goals[state['changed']])**2,axis=1))))
 result['hand_edit']=dict(side=side,fingers=fingers,changed_indices=state['changed'],apply_on_return=bool(apply_on_return),palm_updated=False,body_preserved=True,depth_source=state['depth_source'],initialization=state['initialization'],fit_residual_m=residual,bone_lengths_preserved=True)
 return pose_data.validate(result)

def undo(data,side):
 result=copy.deepcopy(data);history=result.get('hand_edit_history',{}).get(side,[])
 if not history:raise ValueError('此手没有可撤回的校正数据')
 previous=history.pop();hands=result['orientation']['hands']
 if previous is None:hands.pop(side,None)
 else:hands[side]=previous
 result.pop('hand_edit',None);return pose_data.validate(result)

def save(data,state,output,apply_on_return=False):
 out=Path(output).resolve();protected={Path(data['image']['path']).resolve(),Path(state['image']).resolve()}
 if out in protected:raise ValueError('输出不能覆盖参考图片')
 if out.exists():raise ValueError('输出文件已存在，请使用新的保存位置')
 result=document(data,state,apply_on_return);worker.write_json(out,result);return result
