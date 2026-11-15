import importlib,math,sys,types,unittest
from pathlib import Path
pkg=types.ModuleType('patch_test_pkg');pkg.__path__=[str(Path(__file__).resolve().parents[1]/'screenshot_pose')];sys.modules[pkg.__name__]=pkg
p=importlib.import_module(pkg.__name__+'.video_patch');v=importlib.import_module(pkg.__name__+'.video_stability')
def q(deg):return [math.cos(math.radians(deg)/2),math.sin(math.radians(deg)/2),0,0]
class PatchTests(unittest.TestCase):
 def test_single_anchor_exact_and_boundaries_unchanged(self):
  anchors={10:{'hand':dict(rotation=p.delta(q(20),q(65)),location=[1,2,3])}}
  for frame in (-1,0,20,30):
   rot,loc=p.apply(frame,'hand',q(20),[0,0,0],0,20,anchors);self.assertAlmostEqual(v.angle(rot,q(20)),0);self.assertEqual(loc,[0,0,0])
  rot,loc=p.apply(10,'hand',q(20),[0,0,0],0,20,anchors);self.assertAlmostEqual(v.angle(rot,q(65)),0);self.assertEqual(loc,[1,2,3])
 def test_underlying_motion_is_retained(self):
  anchors={10:{'hand':dict(rotation=q(30),location=[0,0,0])}}
  a,_=p.apply(10,'hand',q(20),[0,0,0],0,20,anchors);b,_=p.apply(10,'hand',q(60),[0,0,0],0,20,anchors)
  self.assertAlmostEqual(v.angle(a,b),40)
 def test_two_anchors_shortest_rotation_path(self):
  anchors={5:{'x':dict(rotation=q(20),location=[0,0,0])},15:{'x':dict(rotation=[-x for x in q(60)],location=[2,0,0])}}
  rot,loc=p.offset(10,'x',0,20,anchors);self.assertAlmostEqual(v.angle(rot,q(40)),0);self.assertEqual(loc,[1,0,0])
 def test_invalid_ranges_and_edge_anchors_rejected(self):
  for start,end,anchors in [(0,1,{1:{}}),(0,20,{}),(0,20,{0:{}}),(0,20,{20:{}}),(0,2001,{5:{}})]:
   with self.assertRaises(ValueError):p.validate(start,end,anchors)
if __name__=='__main__':unittest.main()
