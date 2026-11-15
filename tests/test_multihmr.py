import copy,json,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'screenshot_pose'))
import multihmr_data as m

class AnnyConversionTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  raw=ROOT/'test_output/user_crouch/multihmr.raw.json'
  if not raw.is_file():raise unittest.SkipTest('Local official model fixture not available')
  cls.raw=json.loads(raw.read_text(encoding='utf-8'))
  cls.ref=json.loads((ROOT/'test_output/user_crouch/gpu.json').read_text(encoding='utf-8'))
 def test_named_mapping_and_camera_depth(self):
  d=m.convert(self.raw,self.ref);candidate=d['body_candidates'][-1]
  p=np.asarray(self.raw['posed']['bone_heads'])[0];names=self.raw['bone_labels']
  root=p[[names.index('upperleg01.L'),names.index('upperleg01.R')]].mean(0)
  for i,name in m.BODY.items():
   q=candidate['world_landmarks'][i]
   np.testing.assert_allclose([q[k] for k in 'xyz'],p[names.index(name)]-root,atol=1e-9)
 def test_independent_observations_and_original_are_preserved(self):
  before=copy.deepcopy(self.ref);d=m.convert(self.raw,self.ref)
  self.assertEqual(self.ref,before);self.assertEqual(d['landmarks'],self.ref['landmarks'])
  self.assertEqual(d['world_landmarks'],self.ref['world_landmarks'])
  json.dumps(d,allow_nan=False)
 def test_frames_are_proper_and_bad_wrist_keeps_local_hand(self):
  d=m.convert(self.raw,self.ref)
  for item in [d['orientation']['head'],*d['orientation']['hands'].values(),*d['orientation']['feet'].values()]:
   R=np.asarray(item['frame']);np.testing.assert_allclose(R.T@R,np.eye(3),atol=1e-5);self.assertAlmostEqual(np.linalg.det(R),1,places=5)
  self.assertFalse(d['multihmr']['hand_quality']['R']['accepted'])
  self.assertEqual(d['orientation']['hands']['R'],self.ref['orientation']['hands']['R'])
 def test_full_source_rotations_are_retained_instead_of_flattened(self):
  d=m.convert(self.raw,self.ref);bones=d['full_skeleton']['bones']
  self.assertEqual(len(bones),163)
  original=np.asarray(self.raw['posed']['bone_poses'])[0]
  for i,name in enumerate(self.raw['bone_labels']):
   np.testing.assert_allclose(bones[name]['pose_rotation'],m.om.C_DOWN@original[i,:3,:3],atol=1e-9)
  self.assertEqual(len(d['orientation']['hands']['L']['bone_frames']),15)
  self.assertNotIn('bone_frames',d['orientation']['hands']['R'])
 def test_bad_data_cannot_create_pose(self):
  raw=copy.deepcopy(self.raw);raw['posed']['bone_heads'][0][0][0]=float('nan')
  with self.assertRaises(ValueError):m.convert(raw,self.ref)
  raw=copy.deepcopy(self.raw);raw['image_size']=[10,10]
  with self.assertRaises(ValueError):m.convert(raw,self.ref)

if __name__=='__main__':unittest.main()
