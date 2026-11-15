import importlib.util,math,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('motion_diagnostics',Path(__file__).resolve().parents[1]/'screenshot_pose/motion_diagnostics.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class MotionDiagnosticsTests(unittest.TestCase):
 def test_quaternion_sign_and_normalization(self):
  self.assertAlmostEqual(m.quaternion_angle([1,0,0,0],[-2,0,0,0]),0)
  self.assertAlmostEqual(m.quaternion_angle([1,0,0,0],[math.sqrt(.5),0,0,math.sqrt(.5)]),90)
 def test_direction_agreement_ignores_bone_lengths(self):
  a={'shoulder.L':[0,0,0],'elbow.L':[1,0,0]};b={'shoulder.L':[4,2,1],'elbow.L':[7,2,1]}
  self.assertEqual(m.limb_agreement(a,b),{'upper_arm.L':0.})
  b['elbow.L']=[4,4,1];self.assertAlmostEqual(m.limb_agreement(a,b)['upper_arm.L'],90)
 def test_centering_does_not_mutate_world_data(self):
  p={'pelvis':[1,2,3],'hip.L':[2,2,3]};d=m.display_points(p,2)
  self.assertEqual(d['hip.L'],[.5,0,0]);self.assertEqual(p['hip.L'],[2,2,3])
 def test_missing_data_not_a_perfect_score(self):
  self.assertEqual(m.stats([None,float('nan')])['count'],0)
  self.assertIsNone(m.stats([])['median'])
 def test_camera_position_not_claimed_as_world_motion(self):
  with self.assertRaises(ValueError):m.source_packet({})
  self.assertEqual(m.span([[0,1,2],[0,3,8]]),[0,2,6])

if __name__=='__main__':unittest.main()
