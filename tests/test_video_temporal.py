import copy,importlib,math,sys,types,unittest
from pathlib import Path
pkg=types.ModuleType('temporal_test_pkg');pkg.__path__=[str(Path(__file__).resolve().parents[1]/'screenshot_pose')];sys.modules[pkg.__name__]=pkg
v=importlib.import_module(pkg.__name__+'.video_stability');t=importlib.import_module(pkg.__name__+'.video_temporal')
def q(deg):return [math.cos(math.radians(deg)/2),math.sin(math.radians(deg)/2),0,0]
def record(angle,confidence=1.,location=None):return {'bone':dict(rotation=q(angle),confidence=confidence,location=location or [0.,0.,0.])}
class TemporalTests(unittest.TestCase):
 def smooth(self,records,times=None,**kw):
  times=times or [i/30 for i in range(len(records))]
  return t.Smoother(times,lambda i:records[i],lambda i,n:records[i][n]['rotation'],**kw)
 def test_irregular_constant_motion_has_no_temporal_shift(self):
  times=[0,.02,.04,.07,.12,.16,.2];records=[record(90*s,location=[s,2*s,0]) for s in times]
  s=self.smooth(records,times,window=.15)
  for i in range(len(times)):
   rotation,location=s.sample(i,'bone');self.assertLess(v.angle(rotation,q(90*times[i])),1e-5)
   for a,b in zip(location,records[i]['bone']['location']):self.assertAlmostEqual(a,b,places=7)
 def test_small_static_jitter_reduced_and_sources_unchanged(self):
  records=[record(2*(-1)**i) for i in range(31)];before=copy.deepcopy(records);s=self.smooth(records)
  errors=[v.angle(s.sample(i,'bone')[0],q(0)) for i in range(4,27)]
  self.assertLess(sum(errors)/len(errors),1.);self.assertEqual(records,before)
 def test_fast_peak_keeps_time_and_amplitude(self):
  values=[0]*7+[15,45,15]+[0]*7;records=[record(a) for a in values]
  adaptive=self.smooth(records);legacy=self.smooth(records,mode='LEGACY')
  self.assertLess(v.angle(adaptive.sample(8,'bone')[0],q(45)),.5)
  self.assertGreater(v.angle(legacy.sample(8,'bone')[0],q(45)),10)
  amplitudes=[v.angle(adaptive.sample(i,'bone')[0],q(0)) for i in range(len(values))]
  self.assertEqual(amplitudes.index(max(amplitudes)),8)
  self.assertLess(max(amplitudes[:7]+amplitudes[10:]),1.,'fast motion must not leak noticeably into quiet frames')
 def test_sign_near_180_and_protected_frames(self):
  records=[record(x) for x in (165,170,175,180,185,190,195)]
  for i in (1,3,5):records[i]['bone']['rotation']=[-x for x in records[i]['bone']['rotation']]
  s=self.smooth(records)
  for i in range(len(records)):self.assertLess(v.angle(s.sample(i,'bone')[0],records[i]['bone']['rotation']),1e-5)
  records=[record(x,location=[x,0,0]) for x in (0,150,0)]
  s=self.smooth(records,protected={(1,'bone')})
  self.assertEqual(s.sample(1,'bone'),(q(150),[150,0,0]))
 def test_long_gap_and_off_preserve_raw_pose(self):
  records=[record(x) for x in (0,5,0)]
  self.assertEqual(self.smooth(records,[0,.4,.8],window=.5).sample(1,'bone')[0],q(5))
  self.assertEqual(self.smooth(records,mode='OFF').sample(1,'bone')[0],q(5))
 def test_short_weak_flip_burst_fixed_but_real_motion_kept(self):
  records=[record(a,c) for a,c in zip([10,150,155,12],[.9,.2,.2,.9])];times=[0,.06,.12,.18]
  fixes,issues=v.inspect(times,lambda i:records[i],{'bone'})
  self.assertEqual(set(fixes),{1,2});self.assertTrue(all(i['repaired'] for i in issues))
  for i in (1,2):self.assertLess(v.angle(fixes[i]['bone'],q(10+2*i/3)),1e-5)
  for r in records:r['bone']['confidence']=.9
  self.assertFalse(v.inspect(times,lambda i:records[i],{'bone'})[0])
  records=[record(a,c) for a,c in zip([0,150,155,120],[.9,.2,.2,.9])]
  self.assertFalse(v.inspect(times,lambda i:records[i],{'bone'})[0])
 def test_burst_not_extrapolated_or_filled_across_long_gap(self):
  records=[record(a,c) for a,c in zip([0,150,155,0],[.9,.2,.2,.9])]
  self.assertFalse(v.inspect([0,.2,.4,.6],lambda i:records[i],{'bone'})[0])
  records[0]['bone']['confidence']=.2
  self.assertFalse(v.inspect([0,.05,.1,.15],lambda i:records[i],{'bone'})[0])

if __name__=='__main__':unittest.main()
