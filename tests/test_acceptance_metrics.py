import sys,unittest,numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import acceptance_metrics as m
from test_whole_body import fixture
class AcceptanceMetricsTests(unittest.TestCase):
 def test_identical_body_and_unit_scale_translation(self):
  p=fixture();self.assertLess(m.body(p*3+[10,2,1],p)['torso_normalized_mean'],1e-8)
 def test_depth_flip_can_preserve_2d_but_not_3d(self):
  p=fixture();q=p.copy();q[:,1]*=-1
  self.assertAlmostEqual(m.image_points(p[:,[0,2]],q[:,[0,2]],[1000,1000])['mean_px'],0)
  self.assertGreater(m.body(p,q)['torso_normalized_mean'],.1)
 def test_proper_similarity_does_not_hide_reflection(self):
  p=fixture();q=p.copy();q[:,0]*=-1
  self.assertGreater(m.fidelity(p,q)['torso_normalized_mean'],.01)
 def test_fidelity_removes_coordinate_frame(self):
  p=fixture();rot=np.array([[0,-1,0],[1,0,0],[0,0,1]])
  self.assertLess(m.fidelity(p,p@rot*2+[1,2,3])['torso_normalized_mean'],1e-7)
 def test_missing_hand_is_unavailable_not_zero_error(self):self.assertFalse(m.hand(None,np.zeros((21,3)))['available'])
 def test_unknown_real_photo_not_classified_as_accurate(self):self.assertIn('no_ground_truth',m.suspect_layers(False,None,None,None,None,{})[0])
 def test_bad_shapes_rejected(self):
  with self.assertRaises(ValueError):m.body(np.zeros((3,3)),fixture())
if __name__=='__main__':unittest.main()
