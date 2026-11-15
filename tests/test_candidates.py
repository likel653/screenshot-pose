import copy,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'screenshot_pose'))
import candidate_data as c,pose_data
class CandidatesTests(unittest.TestCase):
 def setUp(self):
  self.data=pose_data.template_pose();self.data.update(full_skeleton={'source':'sam3d_mhr'},body_candidates=[dict(source='sam3d_mhr_depth',world_landmarks=copy.deepcopy(self.data['world_landmarks'])),dict(source='mediapipe_estimate',world_landmarks=copy.deepcopy(self.data['world_landmarks']))])
 def test_available_candidates_are_actual_sources(self):
  specs=c.available(self.data);self.assertEqual([x['id'] for x in specs],['FULL','DEPTH_0','DEPTH_1'])
  self.assertEqual(c.available(self.data,True)[0]['id'],'DEPTH_0')
 def test_candidate_keeps_independent_2d_and_original(self):
  before=copy.deepcopy(self.data);self.data['body_candidates'][1]['world_landmarks'][15]['z']=.3
  d=c.document(self.data,c.available(self.data)[2]);self.assertEqual(d['world_landmarks'][15]['z'],.3)
  self.assertEqual(d['landmarks'],before['landmarks']);self.assertEqual(self.data['world_landmarks'],before['world_landmarks'])
  d['world_landmarks'][0]['x']=99;self.assertNotEqual(self.data['world_landmarks'][0]['x'],99)
 def test_sources_deduplicate_and_missing_depth_has_one_real_candidate(self):
  d=pose_data.template_pose();self.assertEqual(len(c.available(d)),1)
  self.data['body_candidates'].append(copy.deepcopy(self.data['body_candidates'][0]));self.assertEqual(len(c.available(self.data)),3)
if __name__=='__main__':unittest.main()
