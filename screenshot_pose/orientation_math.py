"""Geometric orientation helpers, in a camera frame X right / Y away / Z up."""
import numpy as np
C_FACE=np.array([[1,0,0],[0,0,-1],[0,1,0]],float)
C_DOWN=np.array([[1,0,0],[0,0,1],[0,-1,0]],float)
def proper_rotation(matrix):
 a=np.asarray(matrix,dtype=float)
 if a.shape!=(3,3) or not np.isfinite(a).all():raise ValueError('旋转矩阵无效')
 if np.linalg.det(a)<0:raise ValueError('旋转矩阵含镜像，请使用正确左右手坐标系')
 u,s,v=np.linalg.svd(a)
 if min(s)<1e-6:raise ValueError('旋转矩阵退化')
 r=u@v
 if np.linalg.det(r)<0:u[:,-1]*=-1;r=u@v
 return r

def palm_frame(points):
 p=np.asarray(points,dtype=float)
 if p.shape!=(21,3) or not np.isfinite(p).all():raise ValueError('手部需要 21 个有效三维点')
 # Only stable palm joints, never curled fingertips, determine wrist orientation.
 along=(p[5]+p[9]+p[17])/3-p[0];across=p[5]-p[17]
 return frame(across,along)

def frame(across,along):
 y=np.array(along,dtype=float);ny=np.linalg.norm(y)
 if ny<1e-8:raise ValueError('朝向参考点重合')
 y/=ny;x=np.array(across,dtype=float);x-=y*np.dot(x,y);nx=np.linalg.norm(x)
 if nx<1e-8:raise ValueError('朝向参考点共线')
 x/=nx;z=np.cross(x,y)
 return np.column_stack((x,y,z))

def unrotate_hand(world,points,angle):
 """Undo a PIL CCW crop rotation for CV coordinates (x right/y down/z away)."""
 t=np.deg2rad(angle);c,s=np.cos(t),np.sin(t);R=np.array([[c,-s,0],[s,c,0],[0,0,1]])
 world=np.asarray(world)@R.T
 xy=np.asarray(points)-.5;xy=xy@R[:2,:2].T+.5
 return world@C_DOWN.T,xy

def curls(points):
 p=np.asarray(points);result={}
 for finger,ids in [('thumb',(1,2,3,4)),('index',(5,6,7,8)),('middle',(9,10,11,12)),('ring',(13,14,15,16)),('pinky',(17,18,19,20))]:
  vectors=np.diff(p[list(ids)],axis=0);angles=[]
  for a,b in zip(vectors,vectors[1:]):
   den=np.linalg.norm(a)*np.linalg.norm(b)
   if den<1e-10:raise ValueError('指骨点重合')
   angles.append(np.arccos(np.clip(np.dot(a,b)/den,-1,1)))
  result[finger]=float(np.clip(sum(angles)/2.5,0,1))
 return result
