import copy,json,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'screenshot_pose'))
import hand_geometry as h,orientation_math as om,hand_crop_gpu

def hand(gesture='V'):
 p=np.zeros((21,3))
 for i in range(5):
  base=1+i*4;p[base]=[(2-i)*.015,0,.035 if i==0 else .06]
  curled=gesture=='FIST' or (gesture=='V' and i>=3)
  offsets=[(0,0,.025),(0,.02,0),(0,0,-.015)] if curled else [(0,0,.025),(0,0,.02),(0,0,.015)]
  for j,v in enumerate(offsets):p[base+j+1]=p[base+j]+v
 return p

class HandGeometryTests(unittest.TestCase):
 def test_suggestions_are_conservative(self):
  self.assertEqual(h.suggest(hand('V')),'V');self.assertEqual(h.suggest(hand('OPEN')),'OPEN');self.assertEqual(h.suggest(hand('FIST')),'FIST')
  self.assertEqual(h.suggest(np.zeros((20,3))),'UNKNOWN')
 def test_2d_v_requires_visible_straight_and_curled_fingers(self):
  xy=hand('V')[:,[0,2]]
  for i in (13,17):xy[i+2]=xy[i+1]+[.012,0];xy[i+3]=xy[i+2]+[-.008,-.014]
  self.assertEqual(hand_crop_gpu.suggest_v(xy),'V')
  self.assertEqual(hand_crop_gpu.suggest_v(hand('OPEN')[:,[0,2]]),'UNKNOWN')
  self.assertEqual(hand_crop_gpu.suggest_v(np.zeros((21,2))),'UNKNOWN')
 def test_auxiliary_frame_transport_preserves_local_shape(self):
  p=hand();r=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
  original=dict(world=p.tolist(),frame=om.palm_frame(p).tolist(),landmarks=np.zeros((21,2)).tolist(),alternatives=[om.palm_frame(p).tolist()]);before=copy.deepcopy(original)
  out=h.transport(original,r);q=np.array(out['world'])
  np.testing.assert_allclose(q,p@r.T);np.testing.assert_allclose(out['frame'],r@om.palm_frame(p))
  self.assertEqual(original,before);self.assertNotIn('landmarks',out);self.assertIn('reference_landmarks',out)
  np.testing.assert_allclose(om.palm_frame(q),out['frame'])
 def test_reprojection_preserves_all_bone_lengths_and_palm(self):
  p=hand('OPEN');xy=p[:,[0,2]]*[4,-4]+.5;xy[8,0]+=.008;xy[12,0]-=.008
  out,report=h.refine(p,xy)
  self.assertTrue(report['accepted'],report);self.assertLess(report['after'],report['before'])
  for i in range(5):
   b=1+i*4;np.testing.assert_allclose(np.linalg.norm(np.diff(out[b:b+4],axis=0),axis=1),np.linalg.norm(np.diff(p[b:b+4],axis=0),axis=1),atol=1e-9)
  np.testing.assert_allclose(out[[0,1,5,9,13,17]],p[[0,1,5,9,13,17]])
 def test_perfect_projection_no_unnecessary_change(self):
  p=hand('OPEN');out,report=h.refine(p,p[:,[0,2]]*[4,-4]+.5)
  self.assertFalse(report['accepted']);np.testing.assert_array_equal(out,p)
 def test_invalid_points_and_reflections_rejected(self):
  with self.assertRaises(ValueError):h.transport(dict(world=hand().tolist(),frame=np.eye(3).tolist()),np.diag([-1,1,1]))
  with self.assertRaises(ValueError):h.refine(np.zeros((21,3)),np.zeros((21,2)))
  with self.assertRaises(ValueError):h.refine(hand(),np.zeros((20,2)))

if __name__=='__main__':unittest.main()
