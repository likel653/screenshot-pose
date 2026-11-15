import sys,types,importlib,unittest
from pathlib import Path
import numpy as np
pkg=types.ModuleType('foot_source_test');pkg.__path__=[str(Path(__file__).resolve().parents[1]/'screenshot_pose')];sys.modules['foot_source_test']=pkg
m=importlib.import_module('foot_source_test.foot_source')
class NativeSole(unittest.TestCase):
 def fixture(self):
  # An ankle 5cm above the ball must not add that slope to the sole.
  rot=np.array([[1,0,0],[0,0,-1],[0,1,0]])
  return {'full_skeleton':{'source':'sam3d_mhr','rest_axes':'MHR_native','role_map':{'foot.L':'l_foot'},'bones':{'l_foot':{'rest_head':[0,.05,0],'rest_rotation':np.eye(3).tolist(),'pose_rotation':rot.tolist()},'l_ball':{'rest_head':[0,0,.15]}}}}
 def test_neutral_sole_does_not_use_ankle_ball_slope(self):
  frame=m.native_frame(self.fixture(),'L');np.testing.assert_allclose(frame[:,2],[0,0,1],atol=1e-8);self.assertAlmostEqual(frame[2,1],0)
 def test_transports_tilt_without_flattening(self):
  d=self.fixture();angle=.5;rx=np.array([[1,0,0],[0,np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]])
  bone=d['full_skeleton']['bones']['l_foot'];bone['pose_rotation']=(rx@np.array(bone['pose_rotation'])).tolist()
  np.testing.assert_allclose(m.native_frame(d,'L')[:,2],rx@[0,0,1],atol=1e-8)
 def test_explicit_choice_wins(self):
  frame,source=m.frame_for(self.fixture(),'L',{'source':'user_foot_direction_choice','frame':np.eye(3).tolist()})
  np.testing.assert_allclose(frame,np.eye(3));self.assertEqual(source,'user_foot_direction_choice')
 def test_unsupported_source_unchanged(self):
  frame,source=m.frame_for({},'L',{'source':'rtmw3d_foot','frame':np.eye(3).tolist()})
  np.testing.assert_allclose(frame,np.eye(3));self.assertEqual(source,'rtmw3d_foot')
 def test_visible_axis_requires_native_agreement_and_valid_association(self):
  data=self.fixture()
  data['image']={'width':600,'height':900}
  data['wholebody']=[{} for _ in range(133)]
  for i,(x,y) in {15:(.45,.64),17:(.29,.81),18:(.32,.80),19:(.44,.67)}.items():
   data['wholebody'][i]=dict(x=x,y=y,confidence=.9)
  bones=data['full_skeleton']['bones']
  bones['l_foot']['pose_head']=[0.,0.,0.]
  bones['l_ball']['pose_head']=[-.1,-.07,-.15]
  np.testing.assert_allclose(m.visible_shoe_axis(data,'L'),[-.135,.2025])
  bones['l_ball']['pose_head']=[.1,0.,.15]
  self.assertIsNone(m.visible_shoe_axis(data,'L'))
  bones['l_ball']['pose_head']=[-.1,-.07,-.15]
  data['wholebody'][19]['outside_frame']=True
  self.assertIsNone(m.visible_shoe_axis(data,'L'))
 def test_reviewed_axis_drives_image_direction_but_not_native_sole_frame(self):
  data=self.fixture()
  data['image']={'width':600,'height':900}
  data['wholebody']=[{} for _ in range(133)]
  for i,(x,y) in {15:(.45,.64),17:(.29,.81),18:(.32,.80),19:(.44,.67)}.items():
   data['wholebody'][i]=dict(x=x,y=y,confidence=.9)
  bones=data['full_skeleton']['bones']
  bones['l_foot']['pose_head']=[0.,0.,0.]
  bones['l_ball']['pose_head']=[-.1,-.07,-.15]
  obs=dict(source=m.LOCAL_SOURCE,frame=np.eye(3),local_2d=[[270,576],[176,721],[191,732],[261,603]])
  np.testing.assert_allclose(m.reviewed_shoe_axis(data,'L',obs),[-.1291666666667,.2058333333333])
  frame,source=m.frame_for(data,'L',obs)
  np.testing.assert_allclose(frame,m.native_frame(data,'L'))
  self.assertEqual(source,'sam3d_native_sole_rotation')
  obs['local_2d'][1:3]=[[380,450],[401,441]]
  self.assertIsNone(m.reviewed_shoe_axis(data,'L',obs))
if __name__=='__main__':unittest.main()
