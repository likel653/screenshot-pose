import copy,json,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'screenshot_pose'))
import multi_view as m,candidate_data as c
FIXTURE=ROOT/'test_output/user_crouch/sam_integrated.json'
@unittest.skipUnless(FIXTURE.is_file(),'Local multiview fixture not available')
class MultiViewTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.primary=json.loads(FIXTURE.read_text(encoding='utf-8'))
 def auxiliary(self,angle=65):
  d=copy.deepcopy(self.primary);t=np.array([.1,-.2,.05]);a=np.deg2rad(angle);r=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1.]])
  for item in d['full_skeleton']['bones'].values():
   for key in ('pose_head','pose_tail'):item[key]=(r.T@(np.array(item[key])-t)/1.1).tolist()
   item['pose_rotation']=(r.T@np.asarray(item['pose_rotation'])).tolist()
  return d,r,t
 def test_known_camera_alignment(self):
  ref,r,t=self.auxiliary();fit=m.align(self.primary,ref)
  np.testing.assert_allclose(fit['rotation'],r,atol=1e-6);self.assertAlmostEqual(fit['scale'],1.1,places=6);np.testing.assert_allclose(fit['translation'],t,atol=1e-6)
 def test_primary_head_and_finger_local_frames_preserved(self):
  ref,_,_=self.auxiliary();ref['full_skeleton']['bones']['c_head']['pose_head'][0]+=.02
  data=m.fuse(self.primary,[dict(image='synthetic',data=ref)])
  before=self.primary['full_skeleton']['bones'];after=data['full_skeleton_candidates'][0]['skeleton']['bones']
  frozen={'l_wrist_twist','r_wrist_twist','c_head'}
  while True:
   more=frozen|{n for n in before if before[n]['parent'] in frozen}
   if more==frozen:break
   frozen=more
  self.assertGreater(len(frozen),30)
  for n in frozen:
   parent=before[n]['parent'];r0=np.array(before[parent]['pose_rotation']);r1=np.array(after[parent]['pose_rotation'])
   np.testing.assert_allclose(r1.T@after[n]['pose_rotation'],r0.T@before[n]['pose_rotation'],atol=1e-6)
   np.testing.assert_allclose(r1.T@(np.array(after[n]['pose_head'])-after[parent]['pose_head']),r0.T@(np.array(before[n]['pose_head'])-before[parent]['pose_head']),atol=1e-6)
 def test_preserves_primary_and_all_127_frames(self):
  ref,r,t=self.auxiliary();before=copy.deepcopy(self.primary);d=m.fuse(self.primary,[dict(image='synthetic_rotated_view',data=ref)])
  self.assertEqual(d['multiview']['accepted'],1);self.assertEqual(before,self.primary)
  for key in ('image','landmarks','world_landmarks','orientation','full_skeleton'):self.assertEqual(d[key],before[key])
  fused=d['full_skeleton_candidates'][0]['skeleton']['bones'];self.assertEqual(len(fused),127)
  for n,item in fused.items():
   np.testing.assert_allclose(item['pose_head'],before['full_skeleton']['bones'][n]['pose_head'],atol=1e-6)
   r=np.array(item['pose_rotation']);np.testing.assert_allclose(r.T@r,np.eye(3),atol=1e-6);self.assertAlmostEqual(np.linalg.det(r),1.,places=6)
  specs=c.available(d);self.assertTrue(any(x['id']=='FULL_AUX_0' for x in specs));self.assertTrue(any(x['source']=='sam3d_multiview_depth' for x in specs))
  spec=next(x for x in specs if x['id']=='FULL_AUX_0');doc=c.document(d,spec);self.assertEqual(doc['full_skeleton'],d['full_skeleton_candidates'][0]['skeleton'])
 def test_noise_fusion_keeps_link_lengths(self):
  ref,_,_=self.auxiliary();ref['full_skeleton']['bones']['r_wrist']['pose_head'][0]+=.025
  d=m.fuse(self.primary,[dict(image='noisy_view',data=ref)]);self.assertEqual(d['multiview']['accepted'],1)
  original=self.primary['full_skeleton']['bones'];fused=d['full_skeleton_candidates'][0]['skeleton']['bones']
  for n,v in fused.items():
   parent=v['parent']
   if parent:self.assertAlmostEqual(np.linalg.norm(np.array(v['pose_head'])-fused[parent]['pose_head']),np.linalg.norm(np.array(original[n]['pose_head'])-original[parent]['pose_head']),places=6)
 def test_same_view_and_different_pose_are_rejected(self):
  d=m.fuse(self.primary,[dict(image='duplicate_view',data=self.primary)]);self.assertEqual(d['multiview']['accepted'],0)
  ref,_,_=self.auxiliary()
  for n in ('l_uparm','l_lowarm','l_wrist','r_upleg','r_lowleg','r_foot'):ref['full_skeleton']['bones'][n]['pose_head'][2]+=.6
  d=m.fuse(self.primary,[dict(image='wrong_pose',data=ref)]);self.assertEqual(d['multiview']['accepted'],0)
  self.assertEqual(d['world_landmarks'],self.primary['world_landmarks'])
 def test_low_confidence_and_nonfinite_rejected(self):
  ref,_,_=self.auxiliary()
  for p in ref['landmarks']:p['visibility']=.1
  self.assertEqual(m.fuse(self.primary,[dict(data=ref)])['multiview']['accepted'],0)
  ref,_,_=self.auxiliary();ref['full_skeleton']['bones']['root']['pose_head'][0]=float('nan')
  self.assertEqual(m.fuse(self.primary,[dict(data=ref)])['multiview']['accepted'],0)
if __name__=='__main__':unittest.main()
