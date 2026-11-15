"""Original-image identity and MANO coordinate/rotation conservation."""
import copy,sys,tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'screenshot_pose'))
import hand_precision as p,orientation_math as om,hand_geometry as hg
from test_hand_geometry import hand

class PrecisionTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
 def test_verified_high_resolution_preserves_original_pixels(self):
  image=Image.new('RGB',(2400,1600),'#557799');image.putpixel((1200,800),(255,0,0))
  original=self.root/'original.png';cached=self.root/'cached.png';image.save(original);image.resize((600,400),Image.Resampling.LANCZOS).save(cached)
  bgr,scale,info=p.image(cached,original)
  self.assertEqual(bgr.shape,(1600,2400,3));np.testing.assert_array_equal(scale,[4,4]);self.assertTrue(info['original_used'])
  np.testing.assert_array_equal(bgr[800,1200],[0,0,255])
 def test_wrong_image_and_aspect_do_not_replace_cache(self):
  cached=self.root/'cached.png';Image.new('RGB',(200,100),'red').save(cached)
  for dimensions,color,reason in [((800,400),'blue','original_pixels_mismatch'),((800,800),'red','original_aspect_mismatch')]:
   original=self.root/'original.png';Image.new('RGB',dimensions,color).save(original)
   bgr,scale,info=p.image(cached,original);self.assertEqual(info['reason'],reason);self.assertEqual(bgr.shape,(100,200,3));self.assertFalse(info['original_used'])
 def test_exif_orientation_is_normalized_before_matching(self):
  original=self.root/'rotated.jpg';cached=self.root/'cached.png'
  im=Image.new('RGB',(800,400),'#227744');exif=im.getexif();exif[274]=6;im.save(original,exif=exif)
  p._rgb(original).resize((100,200),Image.Resampling.LANCZOS).save(cached)
  bgr,scale,info=p.image(cached,original);self.assertEqual(bgr.shape,(800,400,3));self.assertTrue(info['original_used'])
 def test_missing_original_keeps_processed_image(self):
  path=self.root/'cache.png';Image.new('RGB',(30,40),'white').save(path)
  image,scale,info=p.image(path,self.root/'missing.jpg');self.assertEqual(image.shape,(40,30,3));self.assertFalse(info['original_used'])
 def test_same_background_with_changed_small_hand_is_not_original(self):
  from PIL import ImageDraw
  original=self.root/'original.png';cached=self.root/'cached.png'
  im=Image.new('RGB',(800,400),'white');im.save(original)
  ImageDraw.Draw(im).rectangle((390,190,420,220),fill='blue');im.resize((200,100),Image.Resampling.LANCZOS).save(cached)
  _,_,info=p.image(cached,original);self.assertFalse(info['original_used']);self.assertEqual(info['reason'],'original_local_pixels_mismatch')
 def test_mano_left_mirror_preserves_proper_joint_rotations(self):
  rest=hand('OPEN');t=.4;r=np.array([[np.cos(t),-np.sin(t),0],[np.sin(t),np.cos(t),0],[0,0,1]])
  for side,sign in [('R',1),('L',-1)]:
   result=p.mano_frames(rest,np.tile(r,(16,1,1)),side);transform=om.C_DOWN@np.diag([sign,1,1])
   self.assertEqual(len(result['bone_frames']),15)
   for item in result['bone_frames'].values():
    np.testing.assert_allclose(item['pose_rotation'],transform@r@transform.T)
    self.assertAlmostEqual(np.linalg.det(item['pose_rotation']),1.)
   np.testing.assert_allclose(result['source_rest_palm'],om.palm_frame(rest@transform.T))
 def test_auxiliary_alignment_rotates_pose_not_bind_twice(self):
  rest=hand('OPEN');native=p.mano_frames(rest,np.tile(np.eye(3),(16,1,1)),'R')
  original=dict(world=rest.tolist(),frame=om.palm_frame(rest).tolist(),**native)
  r=np.array([[0,-1,0],[1,0,0],[0,0,1]])
  out=hg.transport(original,r)
  self.assertEqual(out['source_rest_palm'],original['source_rest_palm'])
  for key,item in out['bone_frames'].items():
   self.assertEqual(item['rest_head'],original['bone_frames'][key]['rest_head'])
   np.testing.assert_allclose(item['pose_rotation'],r@original['bone_frames'][key]['pose_rotation'])
 def test_degenerate_mano_rotations_rejected(self):
  with self.assertRaises(ValueError):p.mano_frames(hand('OPEN'),np.zeros((16,3,3)),'R')

if __name__=='__main__':unittest.main()
