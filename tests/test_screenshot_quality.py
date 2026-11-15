import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'screenshot_pose'))
import screenshot_quality as q

class QualityTests(unittest.TestCase):
 def test_tie_preserves_full_pose(self):
  self.assertEqual(q.body_choice(dict(projection_px=10,rotation_deg=2),dict(projection_px=10,rotation_deg=70),1000)[0],'full_source_rotation')
 def test_tiny_pixel_gain_cannot_erase_twist(self):
  self.assertEqual(q.body_choice(dict(projection_px=10,rotation_deg=2),dict(projection_px=9.5,rotation_deg=80),1000)[0],'full_source_rotation')
 def test_large_image_gain_can_win(self):
  self.assertEqual(q.body_choice(dict(projection_px=100,rotation_deg=2),dict(projection_px=15,rotation_deg=30),1000)[0],'landmark_fit')
 def test_invalid_native_falls_back(self):
  self.assertEqual(q.body_choice(dict(projection_px=float('nan'),rotation_deg=2),dict(projection_px=15,rotation_deg=30),1000)[0],'landmark_fit')
 def test_no_valid_body_raises(self):
  with self.assertRaises(ValueError):q.body_choice(dict(projection_px=float('nan'),rotation_deg=2),dict(projection_px=float('inf'),rotation_deg=2),1000)
 def test_body_choice_resolution_invariant(self):
  def run(scale):return q.body_choice(dict(projection_px=20*scale,rotation_deg=4),dict(projection_px=16*scale,rotation_deg=50),1000*scale)[0]
  self.assertEqual(run(1),run(4))
 def test_uncertain_fingertips_do_not_select(self):
  p=np.zeros((21,2));loc=dict(refs=p,scores=[.95]*4+[.4]*17,span=100)
  self.assertIsNone(q.hand_reference_score(p,loc));self.assertEqual(q.hand_choice([None,None,None]),0)
 def test_three_tips_required(self):
  p=np.zeros((21,2));scores=np.ones(21);scores[[12,16,20]]=.3
  self.assertIsNone(q.hand_reference_score(p,dict(refs=p,scores=scores,span=100)))
 def test_independent_observations_choose_better_crop(self):
  p=np.column_stack((np.arange(21)*3.,np.zeros(21)));loc=dict(refs=p,scores=np.ones(21),span=100)
  scores=[q.hand_reference_score(p+20,loc),q.hand_reference_score(p+2,loc),q.hand_reference_score(p+30,loc)]
  self.assertEqual(q.hand_choice(scores),1)
 def test_overlapping_fingertips_do_not_vote_for_a_crop(self):
  p=np.column_stack((np.arange(21)*3.,np.zeros(21)));p[12]=p[8]
  prediction=p.copy();prediction[12]+=30
  score=q.hand_reference_score(prediction,dict(refs=p,scores=np.ones(21),span=100))
  self.assertEqual(score['observed_tips'],3)
  self.assertEqual(score['error'],0.)
 def test_insignificant_crop_gain_keeps_default(self):
  self.assertEqual(q.hand_choice([dict(error=.1),dict(error=.09)]),0)

if __name__=='__main__':unittest.main()
