import sys,unittest,json,copy
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'screenshot_pose'))
import orientation_math as om,local_pose_worker as local,pose_data
class OrientationTests(unittest.TestCase):
 def test_frames_are_right_handed(self):
  for side in (1,-1):
   p=np.zeros((21,3));p[5]=[.03*side,0,.07];p[9]=[0,0,.08];p[17]=[-.03*side,0,.06]
   R=om.palm_frame(p);np.testing.assert_allclose(R.T@R,np.eye(3),atol=1e-10);self.assertAlmostEqual(np.linalg.det(R),1)
   np.testing.assert_allclose(R[:,1],np.mean(p[[5,9,17]],axis=0)/np.linalg.norm(np.mean(p[[5,9,17]],axis=0)))
 def test_crop_rotation_is_reversed(self):
  p=np.random.default_rng(3).normal(size=(21,3));xy=p[:,:2]*.05+.5
  for angle in (0,90,180,270):
   theta=np.deg2rad(angle);R=np.array([[np.cos(theta),np.sin(theta),0],[-np.sin(theta),np.cos(theta),0],[0,0,1]])
   rotated=p@R.T;xyrot=(xy-.5)@R[:2,:2].T+.5
   restored,screen=om.unrotate_hand(rotated,xyrot,angle)
   np.testing.assert_allclose(restored,p@om.C_DOWN.T,atol=1e-10);np.testing.assert_allclose(screen,xy,atol=1e-10)
 def test_known_face_rotation_and_scale(self):
  R=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
  np.testing.assert_allclose(om.proper_rotation(R*1.4),R,atol=1e-10)
  np.testing.assert_allclose(om.C_FACE[:,2],[0,-1,0])
  self.assertAlmostEqual(np.linalg.det(om.C_FACE@R),1)
 def test_bad_geometry_is_rejected(self):
  for x in (np.zeros((21,3)),np.full((21,3),np.nan)):
   with self.assertRaises(ValueError):om.palm_frame(x)
  with self.assertRaises(ValueError):om.proper_rotation(np.zeros((3,3)))
 def test_straight_finger_curl(self):
  p=np.zeros((21,3))
  for ids in ((1,2,3,4),(5,6,7,8),(9,10,11,12),(13,14,15,16),(17,18,19,20)):
   for j,i in enumerate(ids):p[i]=[0,0,j+1]
  self.assertTrue(all(abs(x)<1e-10 for x in om.curls(p).values()))
 def test_real_document_is_valid_and_marked(self):
  root=Path(__file__).resolve().parents[1];p=root/'test_output/user_crouch/details.json'
  if not p.is_file():self.skipTest('local inference output not present')
  data=pose_data.read_pose(p);o=data['orientation'];self.assertTrue(o['head']);self.assertIn('R',o['hands'])
  for hand in o['hands'].values():
   np.testing.assert_allclose(np.array(hand['frame']).T@hand['frame'],np.eye(3),atol=1e-6)
  self.assertEqual(o['hands']['L']['source'],'rtmw_palm_planar_pnp')
if __name__=='__main__':unittest.main()
