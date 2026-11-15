import sys,unittest,numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'screenshot_pose'))
import source_fit_math as m,fit_math
from test_whole_body import fixture
class SourceFitTests(unittest.TestCase):
 def test_serialized_camera_angles_match_rotation(self):
  for v in ([.2,-.3,.1],[np.pi,0,0],[0,0,0]):
   R=fit_math.rotation(v);np.testing.assert_allclose(fit_math.rotation(m.rotation_vector(R)),R,atol=1e-6)
 def test_camera_frame_comes_from_source_not_target_2d(self):
  p=fixture();R=fit_math.rotation([.2,-.3,.1]);np.testing.assert_allclose(m.anchor_rotation(p,p@R.T*2+[3,4,5]),R,atol=1e-8)
 def test_matching_pose_does_not_move(self):
  p=fixture();obs=fit_math.project(p,np.zeros((12,2)),np.ones(12))[0]
  # Use a non-degenerate image scale rather than the zero-target initializer.
  obs=p[:,[0,2]]*[.6,-.6]+[.5,.9];r=m.solve(p,p,obs,np.ones(12))
  np.testing.assert_allclose(r['points'],p,atol=1e-8);self.assertFalse(r['pose_accepted'])
 def test_conflicting_image_cannot_create_large_depth_bend(self):
  p=fixture();obs=p[:,[0,2]]*[.6,-.6]+[.5,.9];obs[[4,10],1]+=.2
  r=m.solve(p,p,obs,np.ones(12));self.assertLessEqual(max(m.deviation(r['points'],p)),8.1)
  for a,b,c in m.CHAINS:
   for u,v in ((a,b),(b,c)):self.assertAlmostEqual(np.linalg.norm(r['points'][v]-r['points'][u]),np.linalg.norm(p[v]-p[u]),places=7)
 def test_planted_pose_is_camera_only(self):
  p=fixture();obs=p[:,[0,2]]*[.6,-.6]+[.5,.9];obs[2,0]+=.2
  np.testing.assert_allclose(m.solve(p,p,obs,np.ones(12),camera_only=True)['points'],p)
 def test_insufficient_independent_observations_rejected(self):
  with self.assertRaises(ValueError):m.solve(fixture(),fixture(),np.zeros((12,2)),np.zeros(12))
 def test_calibrated_perspective_keeps_lens_and_body(self):
  import camera_data
  p=fixture();K=[1.2,1.3,.5,.8];cfg=dict(type='PERSPECTIVE',rotation=np.eye(3),translation=[.03,3,-.6],normalized_intrinsics=K)
  obs=camera_data.project(p,cfg);r=m.perspective_camera_only(p,p,obs,np.ones(12),K)
  np.testing.assert_allclose(r['points'],p);self.assertEqual(r['camera']['normalized_intrinsics'],K);self.assertLess(r['after'],.003)
if __name__=='__main__':unittest.main()
