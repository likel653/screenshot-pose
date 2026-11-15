import copy,json,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'screenshot_pose'))
import sam_data as m
from worker_sam import read_result

class SamConversionTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.geometry=json.loads((ROOT/'screenshot_pose/sam_geometry.json').read_text())
  cls.ref=m.pose_data.template_pose();cls.ref['orientation']=dict(schema='screenshot_pose/orientation_v1',camera_axes='X_right_Y_away_Z_up',hands={'R':dict(source='independent_hand',world=[[0,0,0]]*21)},feet={},warnings=[])
  bind=np.asarray(cls.geometry['inverse_bind']);restrot=np.array([m.quaternion_matrix(v[3:7]).T for v in bind]);resthead=np.array([-r@v[:3]/v[7]/100 for r,v in zip(restrot,bind)])
  cls.raw=dict(joints=(resthead*np.array([1,-1,-1]))[None],joint_rotations=restrot[None])
 def test_neutral_axes_and_bind_roll(self):
  d=m.convert(self.raw,self.ref);np.testing.assert_allclose(d['orientation']['head']['frame'],m.om.C_FACE,atol=1e-6)
  joints,_=m.pose_data.to_joints(d)
  self.assertGreater(joints['head'][2],joints['chest'][2]);self.assertGreater(joints['chest'][2],joints['ankle.L'][2])
 def test_independent_hand_and_image_observations_are_not_replaced(self):
  before=copy.deepcopy(self.ref);d=m.convert(self.raw,self.ref)
  self.assertEqual(before,self.ref);self.assertEqual(d['landmarks'],before['landmarks']);self.assertEqual(d['orientation']['hands'],before['orientation']['hands'])
 def test_all_native_frames_parents_and_depth_retained(self):
  d=m.convert(self.raw,self.ref);bones=d['full_skeleton']['bones'];self.assertEqual(len(bones),127)
  for i,n in enumerate(self.geometry['names']):
   np.testing.assert_allclose(bones[n]['pose_rotation'],m.om.C_FACE@self.raw['joint_rotations'][0,i],atol=1e-6)
   p=self.geometry['parents'][i];self.assertEqual(bones[n]['parent'],self.geometry['names'][p] if p>=0 else None)
  self.assertEqual(d['body_candidates'][0]['source'],'sam3d_mhr_depth');json.dumps(d,allow_nan=False)
 def test_corrupt_native_data_rejected(self):
  raw=copy.deepcopy(self.raw);raw['joints'][0,0,0]=float('nan')
  with self.assertRaises(ValueError):m.convert(raw,self.ref)
 def test_real_native_binary_and_conversion(self):
  path=ROOT/'test_output/user_crouch/sam_integrated.sam.bin'
  if not path.is_file():self.skipTest('Local native inference fixture absent')
  raw=read_result(path);d=m.convert(raw,self.ref);self.assertEqual(len(d['full_skeleton']['bones']),127)
  for item in d['full_skeleton']['bones'].values():
   r=np.asarray(item['pose_rotation']);np.testing.assert_allclose(r.T@r,np.eye(3),atol=1e-6)

if __name__=='__main__':unittest.main()
