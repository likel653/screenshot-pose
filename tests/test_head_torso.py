import json,sys,tempfile,unittest,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'screenshot_pose'))
import head_math as h,torso_math as t,fit_math as f,camera_data as c

class HeadTests(unittest.TestCase):
 def setUp(self):
  self.K=np.array([900.,960.,500.,700.]);self.frame=h.om.C_FACE@f.rotation([.1,-.2,.15]);R=h.om.C_DOWN.T@self.frame
  p=h.TEMPLATE@R.T+[10.,20.,650.];self.xy=p[:,:2]/p[:,2,None]*self.K[:2]+self.K[2:]
  self.ref=dict(xy=self.xy,weights=np.ones(6),span=np.linalg.norm(np.ptp(self.xy,axis=0)))
  p=h.VALIDATION@R.T+[10.,20.,650.]
  self.ref.update(validation_points=h.VALIDATION,validation_xy=p[:,:2]/p[:,2,None]*self.K[:2]+self.K[2:],validation_weights=np.ones(len(p)))
 def test_fixed_rotation_translation_score(self):self.assertLess(h.score(self.frame,self.ref,self.K)['error_px'],.01)
 def test_independent_evidence_rejects_confident_wrong_head(self):
  wrong=h.om.C_FACE@f.rotation([.65,.4,-.3]);chosen,r=h.select(self.frame,[dict(frame=wrong.tolist(),confidence=.99,source='wrong')],self.ref,self.K)
  self.assertIsNone(chosen);self.assertEqual(r['chosen'],'body_head')
 def test_independent_evidence_accepts_correct_head(self):
  body=h.om.C_FACE@f.rotation([.5,-.45,.1]);chosen,r=h.select(body,[dict(frame=self.frame.tolist(),confidence=.8,source='correct')],self.ref,self.K)
  self.assertEqual(chosen['source'],'correct')
 def test_large_flip_and_low_confidence_rejected(self):
  candidates=[dict(frame=(self.frame@f.rotation([0,np.pi,0])).tolist(),confidence=.99,source='flip'),dict(frame=self.frame.tolist(),confidence=.2,source='weak')]
  chosen,r=h.select(self.frame,candidates,self.ref,self.K);self.assertIsNone(chosen);self.assertTrue(all(not a['valid'] for a in r['candidates']))
 def test_calibration_is_image_bound_and_has_fallback(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'image.png';path.write_bytes(b'fixture')
   d=dict(image=dict(path=str(path),width=1000,height=1400),camera_calibration=dict(schema='screenshot_pose/camera_v1',valid=True,image_size=[1000,1400],intrinsics=self.K.tolist(),processed_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
   K,info=h.camera(d,self.ref);self.assertTrue(info['calibrated']);np.testing.assert_equal(K,self.K)
   path.write_bytes(b'changed');K,info=h.camera(d,self.ref);self.assertFalse(info['calibrated']);self.assertTrue(info['reason'])
 def test_occluded_and_small_face_have_no_reference(self):
  b=[dict(x=.5,y=.5,confidence=1.) for _ in range(133)];d=dict(image=dict(width=1000,height=1400),wholebody=b)
  self.assertIsNone(h.reference(d))
 def test_six_point_pnp_cannot_validate_itself(self):
  ref={k:v for k,v in self.ref.items() if not k.startswith('validation_')}
  chosen,r=h.select(h.om.C_FACE,[dict(frame=self.frame.tolist(),source='rtmw_6point_pnp',confidence=.99)],ref,self.K)
  self.assertIsNone(chosen)
 def test_prior_better_than_body_is_preserved(self):
  prior=dict(frame=self.frame.tolist(),source='old_good_head',confidence=.9)
  chosen,r=h.select(h.om.C_FACE@f.rotation([.3,.3,0]),[],self.ref,self.K,prior=prior)
  self.assertEqual(chosen['source'],'old_good_head');self.assertTrue(r['preserved_prior'])

class TorsoTests(unittest.TestCase):
 def setUp(self):
  self.P=np.array([[-.3,0,1.4],[-.6,0,1.2],[-.7,-.2,1.1],[-.15,0,.8],[-.15,0,.4],[-.15,0,0],[.3,0,1.4],[.6,0,1.2],[.7,-.2,1.1],[.15,0,.8],[.15,0,.4],[.15,0,0]])
  self.controls=[dict(pivot=[0,0,1.0],points=[0,1,2,6,7,8],children=[],limit=.21)]
  self.camera=dict(rotation=np.eye(3).tolist(),scale=.5,shift=[.5,.8])
 def test_recovers_small_torso_tilt_without_changing_camera(self):
  q=t.geometry(self.P,self.controls,[0,.14,.08]);obs=c.project(q,self.camera)
  r=t.solve(self.P,obs,np.ones(12),self.camera,self.controls);self.assertTrue(r['accepted']);self.assertLess(r['after'],r['before']*.2)
 def test_no_unnecessary_change_to_already_matching_pose(self):
  r=t.solve(self.P,c.project(self.P,self.camera),np.ones(12),self.camera,self.controls);self.assertFalse(r['accepted'])
 def test_hierarchical_geometry_preserves_lengths(self):
  q=t.geometry(self.P,self.controls,[.1,.1,.1])
  for a,b in ((0,1),(1,2),(6,7),(7,8)):self.assertAlmostEqual(np.linalg.norm(q[a]-q[b]),np.linalg.norm(self.P[a]-self.P[b]))
 def test_grounded_ankles_are_exactly_locked(self):
  controls=[dict(pivot=[0,0,.8],points=list(range(12)),children=[],limit=.21)]
  q=t.geometry(self.P,controls,[.05,.02,0],(5,11));np.testing.assert_equal(q[[5,11]],self.P[[5,11]])
 def test_invalid_points_rejected(self):
  with self.assertRaises(ValueError):t.solve(self.P*np.nan,np.zeros((12,2)),np.ones(12),self.camera,self.controls)

if __name__=='__main__':unittest.main()
