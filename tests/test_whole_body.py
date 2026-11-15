import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'screenshot_pose'))
import whole_body_math as m,camera_data as camera,fit_math as f

def fixture():
 return np.array([[.2,0,1.4],[.45,-.1,1.1],[.6,-.25,.95],[.15,0,.85],[.25,-.4,.55],[.15,0,.08],[-.2,0,1.4],[-.45,-.1,1.1],[-.6,-.25,.95],[-.15,0,.85],[-.25,-.4,.55],[-.15,0,.08]])

class WholeBodyTests(unittest.TestCase):
 def setUp(self):self.P=fixture();self.camera=dict(rotation=np.eye(3).tolist(),scale=.55,shift=[.5,.85])
 def test_root_translation_recovers_projected_torso_shift(self):
  desired=self.P+[.04,0,-.055];obs=camera.project(desired,self.camera)
  r=m.solve(self.P,obs,np.ones(12),self.camera,steps=55)
  self.assertTrue(r['accepted'],r);self.assertGreater(np.linalg.norm(r['root_delta']),.02);self.assertLess(r['after'],r['before']*.15)
 def test_contact_requires_depth_root_motion_and_preserves_all_bone_lengths(self):
  p=self.P.copy();p[1]=p[0]+[0,.3,0];p[2]=p[0]+[0,.6,0]
  target=p[2]+[0,.08,0];pins={2:dict(target=target),5:dict(target=p[5]),11:dict(target=p[11])}
  obs=camera.project(p,self.camera);r=m.solve(p,obs,np.ones(12),self.camera,contacts=pins,steps=65)
  self.assertTrue(r['accepted'],r);self.assertGreater(r['root_delta'][1],.04)
  for end,job in pins.items():np.testing.assert_allclose(r['points'][end],job['target'],atol=1e-10)
  for root,mid,end in m.CHAINS:
   for a,b in ((root,mid),(mid,end)):self.assertAlmostEqual(np.linalg.norm(r['points'][b]-r['points'][a]),np.linalg.norm(p[b]-p[a]),delta=.002)
 def test_unreachable_target_cannot_be_accepted_by_stretching(self):
  pins={2:dict(target=self.P[2]+[2.,0,0])};r=m.solve(self.P,camera.project(self.P,self.camera),np.ones(12),self.camera,contacts=pins,steps=20)
  self.assertFalse(r['accepted']);self.assertFalse(r['valid']);self.assertGreater(r['max_length_error'],.1)
 def test_already_matching_pose_has_no_unnecessary_change(self):
  r=m.solve(self.P,camera.project(self.P,self.camera),np.ones(12),self.camera)
  self.assertFalse(r['accepted']);np.testing.assert_allclose(r['root_delta'],0,atol=1e-9)
 def test_locked_root_cannot_invent_translation(self):
  obs=camera.project(self.P+[.03,0,-.05],self.camera);r=m.solve(self.P,obs,np.ones(12),self.camera,root_lower=np.zeros(3),root_upper=np.zeros(3))
  np.testing.assert_array_equal(r['root_delta'],np.zeros(3))
 def test_joint_flip_fails_anatomy_check(self):
  p=self.P.copy();r,mid,e=m.CHAINS[1]
  hip=p[r];ankle=p[e];old=p[mid];v=ankle-hip;v/=np.linalg.norm(v);p[mid]=hip+(old-hip)@v*v-((old-hip)-((old-hip)@v)*v)
  self.assertFalse(m.anatomy(p,self.P)['valid'])
 def test_input_not_modified_and_invalid_confidence_rejected(self):
  before=self.P.copy();m.solve(self.P,camera.project(self.P,self.camera),np.ones(12),self.camera,steps=5);np.testing.assert_equal(self.P,before)
  with self.assertRaises(ValueError):m.solve(self.P,np.zeros((12,2)),np.zeros(12),self.camera)
 def test_fixed_perspective_camera_and_intrinsics_preserved(self):
  cfg=dict(type='PERSPECTIVE',rotation=np.eye(3).tolist(),translation=[0,2,-.7],normalized_intrinsics=[1.1,1.2,.5,.8])
  obs=camera.project(self.P+[.03,0,-.02],cfg);saved=str(cfg);r=m.solve(self.P,obs,np.ones(12),cfg)
  self.assertLess(r['after'],r['before']);self.assertEqual(str(cfg),saved)

if __name__=='__main__':unittest.main()
