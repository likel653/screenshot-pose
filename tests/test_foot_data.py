import sys,types,importlib,unittest
from pathlib import Path
import numpy as np
pkg=types.ModuleType('foot_test_pkg');pkg.__path__=[str(Path(__file__).resolve().parents[1]/'screenshot_pose')];sys.modules['foot_test_pkg']=pkg
m=importlib.import_module('foot_test_pkg.foot_data')
class FootDataTest(unittest.TestCase):
 def test_rotations_are_proper_and_distinct(self):
  frames=[np.array(x['frame']) for x in m.candidates(np.eye(3))]
  for f in frames:self.assertAlmostEqual(np.linalg.det(f),1);np.testing.assert_allclose(f.T@f,np.eye(3))
  self.assertEqual(len(frames),3)
 def test_missing_depth_reference_is_not_invented(self):
  data={'orientation':{'feet':{}}};out,report=m.refine(data,'L',[])
  self.assertFalse(report['adopted']);self.assertEqual(out,data)
 def test_normalized_keypoints(self):
  d={'image':{'width':100,'height':200},'wholebody':[{'x':.2,'y':.3}]*133}
  np.testing.assert_allclose(m.image_points(d)[0],[20,60])
 def test_crossed_leg_uses_chain_not_screen_side(self):
  xy=np.zeros((133,2));xy[[11,13,15]]=[[200,0],[100,100],[50,200]];xy[[12,14,16]]=[[0,0],[90,100],[150,200]]
  xy[[17,18,19]]=[[60,205],[55,210],[40,200]]
  d={'wholebody':xy.tolist()};obs,reason=m.observation(d,'L',xy,np.ones(133))
  self.assertIsNotNone(obs,reason);self.assertEqual(obs['semantic_output'],'same')
if __name__=='__main__':unittest.main()
