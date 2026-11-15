import hashlib,sys,tempfile,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'screenshot_pose'))
import head_math as h,fit_math as f,local_pose_worker as worker

class PnPTests(unittest.TestCase):
 def test_gpu_worker_refuses_existing_output_before_loading_models(self):
  import worker_head3d
  with tempfile.TemporaryDirectory() as tmp:
   target=Path(tmp)/'model.pth';target.write_bytes(b'protected model')
   with self.assertRaises(ValueError):worker_head3d.run('missing_input',str(target),'missing_config')
   self.assertEqual(target.read_bytes(),b'protected model')
 def test_calibrated_six_points_recover_known_pose(self):
  import cv2
  with tempfile.TemporaryDirectory() as tmp:
   image=Path(tmp)/'image.png';image.write_bytes(b'image binding fixture')
   frame=h.om.C_FACE@f.rotation([.15,-.4,.12]);R=h.om.C_DOWN.T@frame
   K=np.array([[1300.,0,430.],[0,1100.,780.],[0,0,1.]])
   rv,_=cv2.Rodrigues(R);xy,_=cv2.projectPoints(h.TEMPLATE,rv,np.array([20.,-10.,650.]),K,None)
   whole=[dict(x=.5,y=.5,confidence=.95) for _ in range(133)]
   for i,p in zip(h.IDS,xy[:,0]):whole[i].update(x=p[0]/1000,y=p[1]/1600)
   d=dict(image=dict(path=str(image),width=1000,height=1600),wholebody=whole,camera_calibration=dict(schema='screenshot_pose/camera_v1',valid=True,
    image_size=[1000,1600],intrinsics=[1300.,1100.,430.,780.],processed_file_sha256=hashlib.sha256(image.read_bytes()).hexdigest()))
   result=worker.face_fallback(d,1000,1600)
   self.assertTrue(result['camera']['calibrated']);self.assertLess(h.angle(frame,result['frame']),.01)
   legacy=worker.face_fallback(d,1000,1600,calibrated=False)
   self.assertEqual(legacy['camera']['source'],'legacy_crop_focal');self.assertFalse(legacy['camera']['calibrated'])
   image.write_bytes(b'different');result=worker.face_fallback(d,1000,1600)
   self.assertFalse(result['camera']['calibrated'])
if __name__=='__main__':unittest.main()
