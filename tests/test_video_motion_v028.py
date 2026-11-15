import copy,importlib,math,sys,types,unittest
from pathlib import Path
pkg=types.ModuleType('motion_v028_tests');pkg.__path__=[str(Path(__file__).resolve().parents[1]/'screenshot_pose')];sys.modules[pkg.__name__]=pkg
motion=importlib.import_module(pkg.__name__+'.video_motion');sample=importlib.import_module(pkg.__name__+'.video_sampling');resample=importlib.import_module(pkg.__name__+'.video_resample');rot=importlib.import_module(pkg.__name__+'.video_stability')
def q(angle):return [math.cos(math.radians(angle)/2),math.sin(math.radians(angle)/2),0,0]
def observation(y=100,left=(40,200),right=(70,200),confidence=1.):
 return dict(hips=[50,y],confidence=confidence,leg_pixels=100,tracking={'reset':'continuous'},feet={s:dict(point=list(p),confidence=confidence) for s,p in (('L',left),('R',right))})
class MotionTests(unittest.TestCase):
 def test_height_and_screen_translation_follow_observation(self):
  data=[observation(100),observation(110),observation(120)];before=copy.deepcopy(data)
  result=motion.plan([0,.1,.2],data)
  self.assertEqual(result['roots'],[[0,0,0],[0,0,-.1],[0,0,-.2]]);self.assertFalse(result['world_trajectory_recovered']);self.assertEqual(data,before)
  data[-1]['hips'][0]=70
  self.assertEqual(motion.plan([0,.1,.2],data,'SCREEN')['roots'][-1],[.2,0,-.2])
 def test_no_image_motion_no_bob_and_low_confidence_no_move(self):
  data=[observation(),observation(120,confidence=.1),observation()]
  self.assertEqual(motion.plan([0,.1,.2],data)['roots'],[[0,0,0]]*3)
 def test_cuts_rebase_without_root_jump_and_end_support(self):
  data=[observation(100),observation(110),observation(400),observation(410)]
  result=motion.plan([0,.1,.2,.3],data,cuts={2})
  self.assertAlmostEqual(result['roots'][2][2],-.1);self.assertAlmostEqual(result['roots'][3][2],-.2);self.assertEqual(result['segments'],[])
 def test_stationary_floor_supported_lifted_foot_free(self):
  data=[observation(left=(40,160)) for i in range(10)];result=motion.plan([i*.1 for i in range(10)],data)
  self.assertEqual({s['side'] for s in result['segments']},{'R'})
 def test_moving_and_untracked_feet_not_locked(self):
  data=[observation(left=(i*10,200),right=(i*10+30,200)) for i in range(10)]
  self.assertEqual(motion.plan([i*.1 for i in range(10)],data)['segments'],[])
  data=[observation() for i in range(10)]
  for d in data:d['tracking']={}
  self.assertEqual(motion.plan([i*.1 for i in range(10)],data)['segments'],[])
 def test_jumping_feet_are_not_contact_until_landing(self):
  # Both feet have clear screen velocity during ascent/descent.
  heights=[200,200,180,160,140,160,180,200,200,200,200]
  data=[observation(left=(40,y),right=(70,y)) for y in heights]
  segments=motion.plan([i*.1 for i in range(len(data))],data)['segments']
  self.assertTrue(segments);self.assertTrue(all(s['start']>=.8 for s in segments))
 def test_root_never_interpolates_cut(self):
  self.assertEqual(motion.root_at([0,.1,.2],[[0,0,0],[0,0,1],[0,0,2]],.05,{1}),[0,0,0])
 def test_support_review_uses_actual_output_frame_and_nearest_reference(self):
  issues=motion.review_issues([2.,2.1,2.2],{'foot.L':'left'},[dict(start=2.,end=2.2,side='L')],
      [dict(time=2.15,side='L',accepted=False,residual=.1)],10,2.,30)
  self.assertEqual([i['frame'] for i in issues],[13,14]);self.assertTrue(all(i['automatic_support'] and not i['repaired'] for i in issues))
class ResampleTests(unittest.TestCase):
 def make(self,times,angles,positions=None,**kw):
  rows=[dict(rotation=q(x),location=[p,0,0]) for x,p in zip(angles,positions or angles)]
  return resample.Resampler(times,lambda i:rows[i],lambda i,n:(rows[i]['rotation'],rows[i]['location']),**kw)
 def test_constant_velocity_uses_real_irregular_timestamps(self):
  times=[0,.07,.18,.3];s=self.make(times,[90*t for t in times],[2*t for t in times])
  for i in range(31):
   stamp=i*.01;quat,pos=s.at(stamp,'bone');self.assertLess(rot.angle(quat,q(90*stamp)),1e-5);self.assertAlmostEqual(pos[0],2*stamp)
 def test_peak_and_sign_are_preserved_and_no_overshoot(self):
  s=self.make([0,.1,.2,.3],[0,30,0,0],[0,1,0,0])
  for i in range(31):
   quat,p=s.at(i*.01,'bone');self.assertGreaterEqual(p[0],0);self.assertLessEqual(p[0],1);self.assertLessEqual(rot.angle(quat,q(0)),30.00001)
  self.assertLess(rot.angle(s.at(.1,'bone')[0],q(30)),1e-6)
 def test_cut_long_gap_and_large_rotation(self):
  for kwargs in ({'cuts':{1}},{'max_gap':.05}):self.assertEqual(self.make([0,.1,.2],[0,90,0],**kwargs).at(.05,'bone')[0],q(0))
  s=self.make([0,.1],[0,170]);self.assertLess(rot.angle(s.at(.05,'bone')[0],q(85)),1e-5)
 def test_half_turn_sign_and_properties_not_blended(self):
  s=self.make([0,.1,.2],[170,180,190]);s.sample=lambda i,n:([-x for x in q([170,180,190][i])],[i,0,0])
  self.assertLess(rot.angle(s.at(.15,'bone')[0],q(185)),1e-5);self.assertEqual(s.bracket(.15)[0],1)
class SamplingTests(unittest.TestCase):
 def test_only_fast_regions_add_observations_and_keep_base(self):
  base=list(range(0,60,3));scores={i:(.2 if 20<=i<=35 else 0.) for i in range(60)};plan=sample.refine(base,scores,30,24)
  self.assertTrue(set(base)<=set(plan));self.assertGreater(len(plan),len(base));self.assertTrue(all(20<=i<=35 for i in set(plan)-set(base)))
 def test_inference_budget_and_stationary_no_extra_samples(self):
  base=list(range(0,90,5));plan=sample.refine(base,{i:1 for i in range(90)},30,15)
  self.assertTrue(all(sum(int(i/30)==b for i in plan)<=15 for b in range(3)))
  self.assertEqual(sample.refine(base,{i:0 for i in range(90)},30,30),base)
if __name__=='__main__':unittest.main()
