import importlib,sys,types,unittest,math
from pathlib import Path
pkg=types.ModuleType('stability_test_pkg');pkg.__path__=[str(Path(__file__).resolve().parents[1]/'screenshot_pose')];sys.modules['stability_test_pkg']=pkg
v=importlib.import_module('stability_test_pkg.video_stability')
def q(deg):return [math.cos(math.radians(deg)/2),math.sin(math.radians(deg)/2),0,0]
class StabilityTests(unittest.TestCase):
 def review(self,degrees,confidence=None,times=None):
  records=[{'leg':dict(rotation=q(d),confidence=c)} for d,c in zip(degrees,confidence or [1.]*len(degrees))]
  return v.inspect(times or [i/12 for i in range(len(records))],lambda i:records[i],{'leg'})
 def test_isolated_low_confidence_flip_repaired(self):
  fixes,issues=self.review([0,150,0],[.9,.2,.9])
  self.assertAlmostEqual(v.angle(fixes[1]['leg'],q(0)),0);self.assertTrue(issues[0]['repaired'])
 def test_real_high_confidence_peak_only_flagged(self):
  fixes,issues=self.review([0,150,0]);self.assertFalse(fixes);self.assertFalse(issues[0]['repaired'])
 def test_continuous_turn_and_sustained_new_pose_preserved(self):
  for degrees in ([0,50,100,150,180],[0,140,140,0],[0,0,140,140,140]):
   fixes,_=self.review(degrees,[.2]*len(degrees));self.assertFalse(fixes)
 def test_edges_low_rate_and_weak_neighbors_not_fabricated(self):
  self.assertFalse(self.review([150,0,0],[.1,.9,.9])[0])
  self.assertFalse(self.review([0,150,0],[.9,.1,.9],[0,.33,.66])[0])
  self.assertFalse(self.review([0,150,0],[.2,.1,.2])[0])
 def test_quaternion_sign_and_shortest_path(self):
  a=q(170);b=[-x for x in q(175)];r=v.slerp(a,b,.5)
  self.assertAlmostEqual(v.angle(r,q(172.5)),0.,places=5)
 def test_short_local_dropout_is_bracketed(self):
  fixes,issues=self.review([70,0,0,70],[.9,0.,0.,.9])
  self.assertEqual(set(fixes),{1,2});self.assertTrue(all(i['repaired'] for i in issues))
  self.assertAlmostEqual(v.angle(fixes[1]['leg'],q(70)),0.)
 def test_long_dropout_and_real_fast_change_not_filled(self):
  self.assertFalse(self.review([70,0,0,0,0,70],[.9,0,0,0,0,.9])[0])
  self.assertFalse(self.review([0,50,160],[.9,0,.9])[0])
 def test_confidence_mapping(self):
  points=[dict(visibility=1.,presence=1.) for _ in range(33)];points[25]['visibility']=.2
  r=v.confidences(dict(landmarks=points,orientation={'hands':{'R':{'confidence':.3}}}),{'thigh.L':'leg','thigh.R':'other'},dict(R=['finger']))
  self.assertEqual(r,dict(leg=.2,other=1.,finger=.3))
