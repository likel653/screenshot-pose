import copy,importlib,math,sys,types,unittest
from pathlib import Path
import numpy as np
from unittest.mock import patch
pkg=types.ModuleType('hand_sequence_tests');pkg.__path__=[str(Path(__file__).resolve().parents[1]/'screenshot_pose')];sys.modules[pkg.__name__]=pkg
v=importlib.import_module(pkg.__name__+'.video_hands');r=importlib.import_module(pkg.__name__+'.video_stability')
def q(deg):return [math.cos(math.radians(deg)/2),math.sin(math.radians(deg)/2),0,0]
def record(deg,observed):return {'finger':dict(rotation=q(deg),confidence=.9 if observed else 0.,observed=observed)}
def matrix(deg):
 a=math.radians(deg);return [[math.cos(a),-math.sin(a),0],[math.sin(a),math.cos(a),0],[0,0,1]]
class HandSequenceTests(unittest.TestCase):
 def test_short_gap_uses_valid_rotations_and_preserves_inputs(self):
  data=[record(30,True),record(0,False),record(0,False),record(60,True)];before=copy.deepcopy(data)
  fixes,events=v.finger_gaps([0,.1,.2,.3],lambda i:data[i],{'finger'})
  self.assertEqual(set(fixes),{1,2});self.assertAlmostEqual(r.angle(fixes[1]['finger'],q(40)),0,places=5)
  self.assertEqual(data,before);self.assertTrue(all(e['estimated'] for e in events))
 def test_edges_long_gaps_and_cuts_are_not_fabricated(self):
  data=[record(20,True)]+[record(0,False)]*8+[record(30,True)]
  self.assertFalse(v.finger_gaps([i*.1 for i in range(10)],lambda i:data[i],{'finger'})[0])
  data=[record(20,True),record(0,False),record(40,True)]
  self.assertFalse(v.finger_gaps([0,.1,.2],lambda i:data[i],{'finger'},cuts={1})[0])
  data=[record(0,False),record(40,True),record(0,False)]
  self.assertFalse(v.finger_gaps([0,.1,.2],lambda i:data[i],{'finger'})[0])
 def test_low_confidence_observed_gesture_is_not_a_missing_finger(self):
  data=[record(20,True),record(90,True),record(20,True)];data[1]['finger']['confidence']=.3
  self.assertFalse(v.finger_gaps([0,.1,.2],lambda i:data[i],{'finger'})[0])
 def test_manual_hand_sample_breaks_automatic_reconstruction(self):
  data=[record(20,True),record(0,False),record(40,True)];data[1]['finger']['protected']=True
  self.assertFalse(v.finger_gaps([0,.1,.2],lambda i:data[i],{'finger'})[0])
 def test_grasp_or_fk_dependencies_must_agree(self):
  data=[record(20,True),record(0,False),record(40,True)]
  for row in data:row['finger']['dependency_state']={'grasp':0.}
  data[1]['finger']['dependency_state']['grasp']=.7
  self.assertFalse(v.finger_gaps([0,.1,.2],lambda i:data[i],{'finger'})[0])
 def test_estimated_controls_are_grouped_for_review(self):
  events=[dict(index=2,control='a'),dict(index=2,control='b')]
  issues=v.estimate_issues(events,{'a':'L','b':'L'},{'hand.L':'wrist'})
  self.assertEqual(len(issues),1);self.assertEqual(issues[0]['control'],'wrist');self.assertEqual(issues[0]['controls'],['a','b']);self.assertTrue(issues[0]['estimated'])
 def test_actual_update_not_palm_detection_defines_finger_confidence(self):
  data=dict(landmarks=[dict(visibility=1,presence=1)]*33,orientation={'hands':{'L':{'confidence':.9}}})
  self.assertEqual(r.confidences(data,{},dict(L=['finger']),{'touched':[]})['finger'],0)
  self.assertEqual(r.confidences(data,{},dict(L=['finger']),{'touched':['finger']})['finger'],.9)
 def test_body_turns_and_strong_fast_wrist_motion_are_preserved(self):
  docs=[dict(orientation={'hands':{'L':dict(frame=matrix(x),confidence=.9,source='rtmw3d_hand',world=[[0,0,0]]*21)}},forearms={'L':matrix(x)}) for x in (0,90,180)]
  before=copy.deepcopy(docs);plan,events=v.plan_palms(docs,[0,.1,.2]);self.assertFalse(plan);self.assertFalse(events);self.assertEqual(docs,before)
 def test_planar_fit_rejects_inconsistent_candidate(self):
  om=v.om;template=np.zeros((21,3));template[5]=[.035,.065,0];template[9]=[0,.075,0];template[13]=[-.018,.07,0];template[17]=[-.035,.055,0]
  ids=[0,5,9,13,17];points=template[ids]+[0,0,1];xy=points[:,:2]/points[:,2,None]*600+256
  whole=[dict(x=0.,y=0.) for _ in range(133)]
  for i,p in zip(ids,xy):whole[91+i]=dict(x=float(p[0]/512),y=float(p[1]/512))
  good=om.C_DOWN@om.palm_frame(template);bad=om.C_DOWN@np.asarray(matrix(80))@om.palm_frame(template)
  hand=dict(frame=good.tolist(),alternatives=[bad.tolist()],box=[0,0,512,512],confidence=.7,source='rtmw_palm_planar_pnp')
  doc=dict(wholebody=whole,image=dict(width=512,height=512),orientation={'hands':{'L':hand}},forearms={})
  self.assertLess(v.planar_error(doc,'L',hand,good),1e-8);self.assertGreater(v.planar_error(doc,'L',hand,bad),.01)
  plan,events=v.plan_palms([doc,doc],[0,.1]);self.assertFalse(events);np.testing.assert_allclose(plan[0]['L']['frame'],good)
 def test_equal_fit_ambiguity_uses_neighbors_without_suppressing_body_turn(self):
  docs=[]
  for i,body in enumerate((0,90,180)):
   hand=dict(frame=matrix(body),confidence=.9,source='rtmw3d_hand')
   if i==1:hand.update(frame=matrix(body+170),alternatives=[matrix(body)],source='rtmw_palm_planar_pnp')
   docs.append(dict(orientation={'hands':{'L':hand}},forearms={'L':matrix(body)}))
  before=copy.deepcopy(docs)
  with patch.object(v,'planar_error',return_value=.01):
   plan,events=v.plan_palms(docs,[0,.1,.2]);self.assertEqual(len(events),1);np.testing.assert_allclose(plan[1]['L']['frame'],matrix(90),atol=1e-12)
   _,events=v.plan_palms(docs,[0,.1,.2],cuts={1,2});self.assertFalse(events)
  self.assertEqual(docs,before)

if __name__=='__main__':unittest.main()
