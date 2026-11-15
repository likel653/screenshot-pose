import importlib,sys,types,unittest
from pathlib import Path
import numpy as np
from PIL import Image
pkg=types.ModuleType('track_test_pkg');pkg.__path__=[str(Path(__file__).resolve().parents[1]/'screenshot_pose')];sys.modules[pkg.__name__]=pkg
T=importlib.import_module(pkg.__name__+'.video_tracking').Continuity
class TrackingTests(unittest.TestCase):
 def test_crop_contains_current_fast_motion(self):
  t=T();im=Image.new('RGB',(100,100));t.begin(im,0);t.select([[10,10,50,90]],im.size);t.begin(im,.1)
  b=t.select([[20,5,65,95]],im.size);self.assertLessEqual(b[0],20);self.assertGreaterEqual(b[2],65);self.assertLessEqual(b[1],5)
 def test_cut_gap_missing_reset(self):
  t=T();t.begin(Image.new('RGB',(100,100)),0);t.select([[1,1,90,90]],(100,100));t.hands={'L':{}}
  t.begin(Image.new('RGB',(100,100),'white'),.1);self.assertEqual(t.reason,'scene_cut');self.assertFalse(t.hands);self.assertIsNone(t.box)
  t.begin(Image.new('RGB',(100,100),'white'),1);self.assertEqual(t.reason,'time_gap');t.missing();self.assertIsNone(t.box)
 def test_multiple_people_not_silently_selected(self):
  with self.assertRaises(ValueError):T().select([[1,1,20,20],[30,30,90,90]],(100,100))
 def test_hand_candidates_only_comparable_fit(self):
  t=T();t.frame_events=[];t.hands={'L':{'frame':np.eye(3).tolist()}}
  same=dict(frame=np.eye(3).tolist(),reprojection_error=.105);flip=dict(frame=np.diag([-1,-1,1]).tolist(),reprojection_error=.1)
  self.assertIs(t.choose_hand('L',[flip,same]),same)
  same['reprojection_error']=.15;self.assertIs(t.choose_hand('L',[flip,same]),flip)
 def test_expired_hand_not_reused(self):
  t=T();t.begin(Image.new('RGB',(100,100)),0);t.hands={'L':{'frame':np.eye(3).tolist()}};t.commit({'landmarks':[]});self.assertEqual(t.hands,{})
