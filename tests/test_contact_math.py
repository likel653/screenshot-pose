import importlib.util,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('contact_math',Path(__file__).resolve().parents[1]/'screenshot_pose/contact_math.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class ReachTest(unittest.TestCase):
 def test_different_supports_do_not_force_same_height(self):
  shift,ok=m.pelvis_shift([((0,0,1),(0,0,0),.55,.55),((.2,0,1),(.2,0,.5),.55,.55)])
  self.assertTrue(ok);self.assertEqual(shift,0)
 def test_nearest_feasible_root_height(self):
  shift,ok=m.pelvis_shift([((0,0,1.3),(0,0,0),.5,.5)])
  self.assertTrue(ok);self.assertAlmostEqual(shift,-.300001)
 def test_incompatible_supports_and_horizontal_reach(self):
  self.assertFalse(m.pelvis_shift([((0,0,1),(3,0,0),.5,.5)])[1])
  self.assertFalse(m.pelvis_shift([((0,0,1),(0,0,0),.5,.5),((0,0,1),(0,0,3),.5,.5)])[1])
if __name__=='__main__':unittest.main()
