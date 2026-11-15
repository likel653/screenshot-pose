import sys,unittest,copy
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'screenshot_pose'))
import fit_math,worker_high,pose_data
class HighTests(unittest.TestCase):
 def test_wholebody_mapping_and_depth(self):
  p=np.arange(266,dtype=float).reshape(133,2);c=np.ones(133)*.8
  base=pose_data.template_pose();original=copy.deepcopy(base);base['world_landmarks'][15]['z']=.3
  result=worker_high.convert(p,c,(1000,1000),base)
  self.assertEqual(result['landmarks'][15]['x'],p[9,0]/1000)
  self.assertEqual(result['landmarks'][19]['x'],p[99,0]/1000)
  self.assertEqual(result['world_landmarks'][15]['z'],.3)
  self.assertEqual(base['landmarks'],original['landmarks'])
  pose_data.validate(result)
 def test_fallback_and_bad_input(self):
  p=np.ones((133,2))*50;r=worker_high.convert(p,np.ones(133),(100,100))
  self.assertEqual(r['depth_source'],'flat_fallback');self.assertTrue(all(v['z']==0 for v in r['world_landmarks']))
  with self.assertRaises(ValueError):worker_high.convert(p[:10],np.ones(10),(100,100))
 def test_camera_and_bone_lengths(self):
  p=np.array([[.2,0,1.4],[.5,0,1.1],[.8,0,.9],[.15,0,.8],[.3,-.2,.4],[.2,0,0],[-.2,0,1.4],[-.5,0,1.1],[-.8,0,.9],[-.15,0,.8],[-.3,-.2,.4],[-.2,0,0]])
  desired=p.copy();desired[1]+=[-.1,-.03,.1];desired[2]+=[-.1,-.1,.2]
  q=desired@fit_math.rotation([.12,-.08,.1]).T;obs=q[:,[0,2]]*[.5,-.5]+[.5,.8]
  chains=[(0,1,2),(3,4,5),(6,7,8),(9,10,11)]
  f=fit_math.solve(p,obs,np.ones(12),chains,locked_ends=(5,11))
  self.assertLess(f['after'],f['before']*.7)
  for root,mid,end in chains:
   for a,b in ((root,mid),(mid,end)):self.assertAlmostEqual(np.linalg.norm(p[a]-p[b]),np.linalg.norm(f['points'][a]-f['points'][b]),places=8)
  np.testing.assert_allclose(f['points'][[5,11]],p[[5,11]],atol=1e-10)
  self.assertTrue(np.isfinite(f['points']).all())
 def test_insufficient_confidence(self):
  with self.assertRaises(ValueError):fit_math.solve(np.zeros((12,3)),np.zeros((12,2)),np.zeros(12),[])
 def test_exact_camera(self):
  rng=np.random.default_rng(6);p=rng.normal(size=(12,3));q=p@fit_math.rotation([.1,.2,-.1]).T;obs=q[:,[0,2]]*[.8,-.8]+[.3,.6]
  pred,c,err=fit_math.project(p,obs,np.ones(12),[.1,.2,-.1]);self.assertLess(err,1e-10);self.assertAlmostEqual(c['scale'],.8)
if __name__=='__main__':unittest.main()
