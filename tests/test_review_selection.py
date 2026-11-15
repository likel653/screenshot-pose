import copy,sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'screenshot_pose'))
import review_data,depth_selection,hand_selection,pose_data,orientation_math as om

class ReviewTests(unittest.TestCase):
 def test_low_points_map_to_anatomical_side(self):
  d=pose_data.template_pose();d['landmarks'][25]['visibility']=.1
  rows=review_data.issues(d)
  self.assertTrue(any(r['id']=='low_LEG_L' for r in rows));self.assertFalse(any(r['id']=='low_LEG_R' for r in rows))
 def test_independent_detector_not_overconfident_projection(self):
  d=pose_data.template_pose();d['wholebody']=[dict(confidence=.99) for _ in range(133)];d['wholebody'][9]['confidence']=.1
  self.assertTrue(any(r['id']=='low_ARM_L' for r in review_data.issues(d)))
 def test_fallback_is_not_fresh_detection(self):
  d=pose_data.template_pose();d['hand_refinement']={'hands':{'L':dict(fallback=True,adopted=True)}}
  self.assertTrue(any(r['id']=='hand_fallback_L' for r in review_data.issues(d)))
 def test_manual_transfer_is_not_flagged_as_failure(self):
  d=pose_data.template_pose();r=dict(hand_transfer={'L':dict(mean_direction_error_deg=60,intentional_manual_or_partial_strength=True)})
  self.assertFalse(any(x['id']=='finger_transfer_L' for x in review_data.issues(d,r)))
 def test_ambiguity_visible(self):
  self.assertTrue(any(x['id']=='depth_tie' for x in review_data.issues(pose_data.template_pose(),{'depth_selection':{'ambiguous':True}})))

class DepthTests(unittest.TestCase):
 def test_source_structure_penalty_can_veto_pixel_only_switch(self):
  import screenshot_quality
  a=dict(projection_px=30,score_px=30,rotation_deg=0);b=dict(projection_px=5,score_px=45,rotation_deg=0)
  self.assertEqual(screenshot_quality.body_choice(a,b,1000)[0],'full_source_rotation')
 def test_measure_uses_fixed_camera_and_penalizes_unreachable_contact(self):
  from test_whole_body import fixture
  import camera_data
  p=fixture();camera=dict(rotation=np.eye(3).tolist(),scale=.55,shift=[.5,.85]);obs=camera_data.project(p,camera)
  before=copy.deepcopy(camera);a=depth_selection.measure(p,p,obs,np.ones(12),camera,1000)
  b=depth_selection.measure(p,p,obs,np.ones(12),camera,1000,[(2,p[2]+[3,0,0])])
  self.assertTrue(a['valid']);self.assertAlmostEqual(a['projection_px'],0);self.assertGreater(b['score_px'],a['score_px']+10);self.assertEqual(before,camera)
 def test_tie_keeps_default_and_reports_ambiguity(self):
  i,d=depth_selection.choose([dict(valid=True,score_px=10),dict(valid=True,score_px=9.5)],1000)
  self.assertEqual(i,0);self.assertTrue(d['ambiguous'])
 def test_clear_combined_gain_can_choose(self):
  i,d=depth_selection.choose([dict(valid=True,score_px=20),dict(valid=True,score_px=5)],1000)
  self.assertEqual(i,1);self.assertFalse(d['ambiguous'])
 def test_invalid_winner_not_selected(self):
  self.assertEqual(depth_selection.choose([dict(valid=True,score_px=20),dict(valid=False,score_px=0)],1000)[0],0)
 def test_no_valid_does_not_fake_success(self):
  with self.assertRaises(ValueError):depth_selection.choose([dict(valid=False,score_px=0)],1000)
 def test_resolution_invariant(self):
  for scale in (1,4):self.assertEqual(depth_selection.choose([dict(valid=True,score_px=10*scale),dict(valid=True,score_px=9.5*scale)],1000*scale)[0],0)

class HandTests(unittest.TestCase):
 def setUp(self):
  p=np.zeros((21,3));p[1:]=np.array([[i%4*.01,(i//4+1)*.02,.002*i] for i in range(20)])
  self.old=dict(world=p.tolist(),frame=np.eye(3).tolist(),source='old',confidence=.9)
  rot=om.proper_rotation([[0,-1,0],[1,0,0],[0,0,1]])
  self.new=dict(world=(p@rot.T*1.2).tolist(),frame=rot.tolist(),source='new',confidence=.85)
 def test_palm_only_preserves_palm_local_finger_shape(self):
  r=hand_selection.compose(self.old,self.new,'PALM')
  np.testing.assert_allclose(np.array(r['world'])@np.array(r['frame']),self.old['world'],atol=1e-9)
  np.testing.assert_allclose(r['frame'],self.new['frame'])
 def test_fingers_only_preserves_palm_frame(self):
  r=hand_selection.compose(self.old,self.new,'FINGERS')
  np.testing.assert_allclose(r['frame'],self.old['frame'])
  np.testing.assert_allclose(np.array(r['world'])@np.array(r['frame']),np.array(self.new['world'])@np.array(self.new['frame']),atol=1e-9)
 def test_does_not_mutate_inputs(self):
  before=copy.deepcopy((self.old,self.new));hand_selection.compose(self.old,self.new,'PALM');self.assertEqual(before,(self.old,self.new))
 def test_other_hand_and_body_preserved(self):
  d=pose_data.template_pose();d['orientation']={'hands':{'L':self.old,'R':self.old}}
  c=copy.deepcopy(d);c['orientation']['hands']['L']=self.new;c['hand_refinement']={'hands':{'L':dict(adopted=True)}}
  r,_=hand_selection.select(d,c,('L',),'FINGERS','HAMER')
  self.assertEqual(r['world_landmarks'],d['world_landmarks']);self.assertEqual(r['orientation']['hands']['R'],d['orientation']['hands']['R'])
 def test_manual_edits_and_rejected_candidates_protected(self):
  d=pose_data.template_pose();d['orientation']={'hands':{'L':self.old}}
  c=copy.deepcopy(d);c['hand_refinement']={'hands':{'L':dict(adopted=False)}}
  with self.assertRaises(ValueError):hand_selection.select(d,c,('L',),'ALL','HAMER')
  c['hand_refinement']['hands']['L']['adopted']=True;d['manual_hand_edits']={'L':True}
  with self.assertRaises(ValueError):hand_selection.select(d,c,('L',),'ALL','HAMER')
 def test_missing_finger_data_not_fabricated(self):
  with self.assertRaises(ValueError):hand_selection.compose(dict(frame=np.eye(3).tolist()),self.new,'PALM')
 def test_mano_bind_frames_stay_fixed_and_pose_rotates_once(self):
  hd=copy.deepcopy(self.old);hd.update(native_source='MANO',source_rest_palm=np.eye(3).tolist(),bone_frames={'index:1':dict(rest_head=[0,0,0],rest_tail=[0,.1,0],rest_rotation=np.eye(3).tolist(),pose_rotation=np.eye(3).tolist())})
  r=hand_selection.compose(hd,self.new,'PALM')
  self.assertEqual(r['bone_frames']['index:1']['rest_tail'],hd['bone_frames']['index:1']['rest_tail'])
  np.testing.assert_allclose(r['source_rest_palm'],hd['source_rest_palm']);np.testing.assert_allclose(r['bone_frames']['index:1']['pose_rotation'],self.new['frame'])

if __name__=='__main__':unittest.main()
