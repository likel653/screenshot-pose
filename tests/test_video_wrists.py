import copy,importlib,math,sys,types,unittest
from pathlib import Path
pkg=types.ModuleType('wrist_tests');pkg.__path__=[str(Path(__file__).resolve().parents[1]/'screenshot_pose')];sys.modules[pkg.__name__]=pkg
w=importlib.import_module(pkg.__name__+'.video_wrists');h=importlib.import_module(pkg.__name__+'.video_hands');r=importlib.import_module(pkg.__name__+'.video_stability');vd=importlib.import_module(pkg.__name__+'.video_data')
def q(angle):return [math.cos(math.radians(angle)/2),math.sin(math.radians(angle)/2),0,0]
def row(angle,reliable=False,applied=True):
 return dict(relative_rotation=q(angle),source='rtmw3d_hand' if reliable else 'rtmw_palm_planar_pnp',confidence=.9,applied=applied,
    reliable_3d=reliable,axes=[[10,0],[0,10]],image_size=10,image_confidence=.9,protected=False,dependency_state={})
def records(rows):return [{'__wrists__':{'L':v}} for v in rows]
class WristTests(unittest.TestCase):
 def test_short_ambiguous_flip_uses_fixed_anchors_and_preserves_inputs(self):
  data=records([row(10,True),row(150),row(20,True)]);before=copy.deepcopy(data)
  fixes,events,_=w.plan([0,.1,.2],data)
  self.assertEqual(set(fixes),{1});self.assertLess(r.angle(fixes[1]['L'],q(15)),1e-5);self.assertEqual(data,before);self.assertTrue(events[0]['estimated'])
 def test_confident_rigid_3d_fast_turn_is_not_replaced(self):
  data=records([row(10,True),row(170,True),row(15,True)])
  self.assertFalse(w.plan([0,.1,.2],data)[0])
 def test_visible_palm_turn_is_preserved_even_with_planar_source(self):
  data=records([row(10,True),row(150),row(20,True)]);data[1]['__wrists__']['L']['axes']=[[0,10],[-10,0]]
  self.assertFalse(w.plan([0,.1,.2],data)[0])
 def test_missing_pixels_manual_changes_cuts_and_long_gap_block_repair(self):
  base=records([row(10,True),row(150),row(20,True)])
  for field,value in (('axes',None),('protected',True),('image_confidence',.2),('dependency_state',{'grasp':1})):
   data=copy.deepcopy(base);data[1]['__wrists__']['L'][field]=value;self.assertFalse(w.plan([0,.1,.2],data)[0])
  self.assertFalse(w.plan([0,.1,.2],base,cuts={1})[0]);self.assertFalse(w.plan([0,.2,.4],base)[0])
 def test_3d_shape_outlier_needs_consistent_anchor_shapes(self):
  data=records([row(10,True),row(150,True),row(20,True)])
  for i,factor in enumerate((1,2,1)):data[i]['__wrists__']['L']['shape']=[factor]*10
  fixes,events,_=w.plan([0,.1,.2],data);self.assertTrue(fixes);self.assertTrue(events[0]['shape_outlier'])
  data[2]['__wrists__']['L']['shape']=[1.5]*10;self.assertFalse(w.plan([0,.1,.2],data)[0])
 def test_overlapping_spikes_cannot_modify_each_others_anchors(self):
  data=records([row(a) for a in (0,120,0,120,0)])
  fixes,events,_=w.plan([i*.05 for i in range(5)],data)
  edited=set(fixes);anchors={e[k] for e in events for k in ('left','right')}
  self.assertFalse(edited&anchors);self.assertTrue(fixes)
 def test_memory_is_relative_bounded_and_fades_to_body(self):
  data=records([row(60,True)]+[row(0,applied=False) for i in range(4)])
  fixes,events=w.carry([0,.1,.2,.3,.4],data)
  self.assertEqual(set(fixes),{1,2,3});self.assertLess(r.angle(fixes[1]['L'],q(60)),1e-5)
  self.assertGreater(r.angle(fixes[2]['L'],q(0)),r.angle(fixes[3]['L'],q(0)));self.assertTrue(all(e['estimated'] for e in events))
 def test_memory_does_not_replace_observed_motion_or_cross_cut(self):
  data=records([row(60,True),row(150,True),row(0,applied=False)])
  fixes,_=w.carry([0,.1,.2],data);self.assertNotIn(1,fixes);self.assertLess(r.angle(fixes[2]['L'],q(150)),1e-5)
  self.assertFalse(w.carry([0,.1,.2],data,cuts={2})[0])
 def test_memory_dependency_and_manual_changes_clear_state(self):
  for key,value in (('protected',True),('dependency_state',{'fk':1})):
   data=records([row(60,True),row(0,applied=False),row(0,applied=False)]);data[1]['__wrists__']['L'][key]=value
   self.assertFalse(w.carry([0,.1,.2],data)[0])
 def test_uncertain_observation_consumes_older_memory(self):
  data=records([row(60,True),row(150),row(0,applied=False)])
  self.assertFalse(w.carry([0,.1,.2],data)[0])
 def test_recovery_anchor_has_priority_over_forward_hold(self):
  data=records([row(60,True),row(0,applied=False),row(100,True)])
  fixes,events=w.carry([0,.1,.2],data);self.assertLess(r.angle(fixes[1]['L'],q(80)),1e-5)
  self.assertEqual(events[0]['method'],'forearm_relative_missing_bracket')
 def test_memory_that_would_increase_recovery_jump_is_declined(self):
  data=records([row(0,True),row(40,applied=False),row(80)]);data[-1]['__wrists__']['L']['confidence']=.5
  fixes,events=w.carry([0,.1,.2],data);self.assertFalse(fixes);self.assertTrue(events[0]['quality_guard_rejected'])
 def test_existing_channel_bridge_outranks_forward_memory(self):
  data=records([row(60,True),row(0,applied=False),row(0,applied=False)])
  self.assertFalse(w.carry([0,.1,.2],data,blocked_decay={(1,'L')})[0])
 def test_manual_hand_locks_wrist_and_fingers_at_all_temporal_stages(self):
  data=[dict(__wrists__={'L':dict(hand='left_wrist',protected=True)},right_index=dict(protected=False))]
  locked=w.locked_controls(data,{'left_index':'L','right_index':'R'})
  self.assertEqual(locked,{(0,'left_wrist'),(0,'left_index')})
 def test_actual_wholebody_confidence_used_for_image_gate(self):
  points=[dict(x=.1,y=.1,confidence=.1) for _ in range(133)]
  for i,xy in ((91,(.1,.1)),(96,(.2,.2)),(100,(.1,.25)),(108,(0,.2))):points[i].update(x=xy[0],y=xy[1])
  evidence=w.evidence(dict(wholebody=points,image=dict(width=100,height=100)), 'L',False)
  self.assertEqual(evidence['image_confidence'],.1)
class FingerMemoryTests(unittest.TestCase):
 def data(self):return [{'finger':dict(rotation=q(a),observed=bool(i==0),confidence=.9 if i==0 else 0,protected=False)} for i,a in enumerate((60,0,0,0,0))]
 def test_missing_fingers_hold_then_decay_and_expire(self):
  data=self.data();before=copy.deepcopy(data);fix,events=h.carry_fingers([0,.1,.2,.3,.4],lambda i:data[i],{'finger'})
  self.assertEqual(set(fix),{1,2,3});self.assertLess(r.angle(fix[1]['finger'],q(60)),1e-5);self.assertEqual(data,before)
  self.assertGreater(r.angle(fix[2]['finger'],q(0)),r.angle(fix[3]['finger'],q(0)))
 def test_bracketed_repair_and_observations_have_priority(self):
  data=self.data();data[2]['finger'].update(observed=True,confidence=.3)
  fix,_=h.carry_fingers([0,.1,.2,.3,.4],lambda i:data[i],{'finger'},reserved={1:{'finger':q(45)}})
  self.assertFalse(fix)
class ReferencePlanTests(unittest.TestCase):
 def fixture(self):
  data=dict(start_seconds=4,frames=[dict(time=4+i/30,status='ok',pose='%06d/pose.json'%i) for i in range(30)],end_seconds=5)
  mapped=[dict(frame=10+round(i*24/30),time=f['time'],pose=f['pose']) for i,f in enumerate(data['frames'])];return data,mapped
 def test_30hz_observations_on_24fps_keep_all_references(self):
  data,mapped=self.fixture();self.assertLess(len({r['frame'] for r in mapped}),len(mapped))
  plan=vd.reference_plan(data,mapped,24,10);self.assertEqual(len(plan['rows']),30);self.assertEqual(plan['end_frame'],33);self.assertEqual(plan['origin_time'],4)
 def test_missing_person_keeps_actual_recording_thumbnail(self):
  data,mapped=self.fixture();data['frames'][2]=dict(time=4+2/30,status='missing');mapped.pop(2)
  plan=vd.reference_plan(data,mapped,24,10);self.assertEqual(plan['rows'][2]['status'],'missing');self.assertEqual(plan['rows'][2]['image_relative'],'000002/image.png')
 def test_legacy_rate_inferred_independently_of_current_scene(self):
  data=dict(frames=[dict(time=i/24,status='ok',pose='frame/pose.json') for i in range(24)])
  mapped=[dict(frame=1+i,time=r['time'],pose=r['pose']) for i,r in enumerate(data['frames'])]
  self.assertAlmostEqual(vd.reference_plan(data,mapped)['fps'],24)
 def test_bad_time_order_rejected_even_when_frame_numbers_repeat(self):
  data,mapped=self.fixture();mapped[1]['time']=mapped[0]['time']
  with self.assertRaises(ValueError):vd.reference_plan(data,mapped,24,10)
if __name__=='__main__':unittest.main()
