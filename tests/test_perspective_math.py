import sys,unittest,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'screenshot_pose'))
import perspective_math as m,camera_data as c

CHAINS=[(0,1,2),(3,4,5),(6,7,8),(9,10,11)]
P=np.array([[-.25,0,1.4],[-.48,-.25,1.1],[-.35,-.65,1.25],[-.14,0,.85],[-.2,-.5,.5],[-.12,-.05,.1],
 [.25,0,1.4],[.5,.3,1.05],[.55,.55,1.4],[.14,0,.85],[.22,-.3,.5],[.2,.05,.1]])

class PerspectiveTests(unittest.TestCase):
 def camera(self):return dict(type='PERSPECTIVE',rotation=np.eye(3).tolist(),translation=[.06,2.,-.8],normalized_intrinsics=[1.05,1.08,.48,.75])
 def test_recovers_camera_on_strong_depth_variation(self):
  obs=c.project(P,self.camera());r=m.solve(P,obs,np.ones(12),[1050,1080,480,750],1000,CHAINS,steps=70)
  self.assertLess(r['after'],.004);self.assertLess(r['after'],r['before']*.1)
 def test_fixed_intrinsics_and_exact_bone_lengths(self):
  obs=c.project(P,self.camera());altered=P.copy();altered[1]+=[.03,.03,.02]
  r=m.solve(altered,obs,np.ones(12),[1050,1080,480,750],1000,CHAINS)
  for a,b,e in CHAINS:
   for i,j in ((a,b),(b,e)):self.assertAlmostEqual(np.linalg.norm(r['points'][i]-r['points'][j]),np.linalg.norm(altered[i]-altered[j]),places=9)
  self.assertEqual(r['camera']['normalized_intrinsics'],[1.05,1.08,.48,.75])
 def test_grounded_ankles_do_not_move(self):
  r=m.solve(P,c.project(P,self.camera()),np.ones(12),[1050,1080,480,750],1000,CHAINS,locked_ends=(5,11))
  np.testing.assert_allclose(r['points'][[5,11]],P[[5,11]],atol=1e-12)
 def test_zero_confidence_outlier_ignored(self):
  obs=c.project(P,self.camera());obs[2]=[10,-10];weights=np.ones(12);weights[2]=0
  r=m.solve(P,obs,weights,[1050,1080,480,750],1000,CHAINS)
  self.assertLess(r['after'],.004)
 def test_projection_rejects_behind_camera(self):
  with self.assertRaises(ValueError):c.project([[0,-3,0]],self.camera())
 def test_calibration_is_bound_to_image_size(self):
  calibration=dict(schema='screenshot_pose/camera_v1',valid=True,intrinsics=[1000,1000,500,750],image_size=[1000,1500])
  with self.assertRaises(ValueError):c.intrinsics(calibration,900,1600)
 def test_nan_input_rejected(self):
  p=P.copy();p[0,0]=np.nan
  with self.assertRaises(ValueError):m.solve(p,np.zeros((12,2)),np.ones(12),[1000,1000,500,750],1000,CHAINS)
 def test_pelvis_coordination_preserves_planted_feet_and_bone_lengths(self):
  target=P.copy();target[[0,1,2,3,6,7,8,9]]+=[.04,.02,-.02]
  obs=c.project(target,self.camera())
  r=m.solve(P,obs,np.ones(12),[1050,1080,480,750],1000,CHAINS,locked_ends=(5,11),coordinate_pelvis=True)
  np.testing.assert_allclose(r['points'][[5,11]],P[[5,11]],atol=1e-12)
  for a,b,e in CHAINS:
   for i,j in ((a,b),(b,e)):self.assertAlmostEqual(np.linalg.norm(r['points'][i]-r['points'][j]),np.linalg.norm(P[i]-P[j]),places=5)
  self.assertTrue(r['pelvis_coordination'])
 def test_real_eve_camera_does_not_sacrifice_feet_to_torso(self):
  data=json.loads(Path(__file__).with_name('perspective_eve_regression.json').read_text(encoding='utf8'))
  r=m.solve(data['points'],data['observed'],data['weights'],data['intrinsics'],data['width'],CHAINS,steps=60)
  self.assertLessEqual(r['camera_only'],r['before']+1e-8)
  self.assertLessEqual(r['after'],r['camera_only']+1e-8)
  self.assertLess(r['after']*data['width'],30.)

if __name__=='__main__':unittest.main()
