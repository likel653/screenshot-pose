import copy,json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'screenshot_pose'))
import worker_hands as w,orientation_math as om
import test_hand_geometry,test_multiview

class Detector:
 def __enter__(self):return self
 def __exit__(self,*args):pass

class HandWorkerTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
  self.base=json.loads((ROOT/'tests/fixtures/hand_contract_pose.json').read_text(encoding='utf-8'));self.base['image']['path']=str(self.root/'main.png')
  self.input=self.root/'main.json';self.request=self.root/'request.json';self.output=self.root/'out.json';self.model=self.root/'model.task'
  w.worker.write_json(self.input,self.base);self.model.write_bytes(b'model');(self.root/'hand_landmarker.task').write_bytes(b'fake')
  self.request.write_text(json.dumps(dict(R=dict(image='',gesture='V'))),encoding='utf-8')
  vision=types.SimpleNamespace(HandLandmarkerOptions=lambda **kw:kw,HandLandmarker=types.SimpleNamespace(create_from_options=lambda _:Detector()))
  mp=types.SimpleNamespace(tasks=types.SimpleNamespace(vision=vision,BaseOptions=lambda **kw:kw))
  p=patch.dict(sys.modules,mediapipe=mp);p.start();self.addCleanup(p.stop)
  self.hand=dict(world=test_hand_geometry.hand().tolist(),frame=om.palm_frame(test_hand_geometry.hand()).tolist(),confidence=.8,source='cropped_hand_refined',gesture_suggestion='V')
  p=patch.object(w,'infer_hand',return_value=(self.hand,dict(attempts=16,candidates=1)));p.start();self.addCleanup(p.stop)
  p=patch.object(w.mv,'pixels',side_effect=lambda path:str(Path(path).resolve()));p.start();self.addCleanup(p.stop)
 def run_job(self):return w.run(self.input,self.request,self.model,self.output,native_root_path=self.root)
 def test_preserves_body_and_unselected_hand(self):
  result=self.run_job()
  for key in self.base:
   if key!='orientation':self.assertEqual(result[key],self.base[key])
  self.assertEqual(result['orientation']['hands']['L'],self.base['orientation']['hands']['L'])
  self.assertTrue(result['hand_refinement']['hands']['R']['has_21_points'])
  self.assertEqual(result['orientation']['hands']['R']['gesture_constraint'],'V')
 def test_auxiliary_alignment_rotates_only_selected_hand(self):
  helper=test_multiview.MultiViewTests();helper.primary=self.base;aux,r,t=helper.auxiliary()
  self.request.write_text(json.dumps(dict(R=dict(image=str(self.root/'side.png'),gesture='V'))),encoding='utf-8')
  with patch.object(w,'reference',return_value=(aux,True)):result=self.run_job()
  np.testing.assert_allclose(result['orientation']['hands']['R']['frame'],r@self.hand['frame'],atol=1e-6)
  self.assertEqual(result['full_skeleton'],self.base['full_skeleton']);self.assertTrue(result['hand_refinement']['hands']['R']['cache_hit'])
 def test_reference_failure_keeps_existing_hand(self):
  self.request.write_text(json.dumps(dict(R=dict(image=str(self.root/'side.png'),gesture='V'))),encoding='utf-8')
  with patch.object(w,'reference',side_effect=ValueError('different pose')):result=self.run_job()
  self.assertEqual(result['orientation'],self.base['orientation']);self.assertFalse(result['hand_refinement']['hands']['R']['adopted'])
 def test_never_overwrites_model_or_input(self):
  self.output=self.root/'hand_landmarker.task'
  with self.assertRaises(ValueError):self.run_job()
  self.assertEqual(self.output.read_bytes(),b'fake')
  self.output=self.input
  with self.assertRaises(ValueError):self.run_job()
  self.assertEqual(json.loads(self.input.read_text(encoding='utf-8')),self.base)

if __name__=='__main__':unittest.main()
