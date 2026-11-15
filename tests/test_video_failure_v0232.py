import importlib,sys,types,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
pkg=types.ModuleType('failure_v0232');pkg.__path__=[str(Path(__file__).resolve().parents[1]/'screenshot_pose')];sys.modules[pkg.__name__]=pkg
errors=importlib.import_module(pkg.__name__+'.job_errors')
detectors=importlib.import_module(pkg.__name__+'.person_detector')

class VideoFailureTests(unittest.TestCase):
 def make(self,confidence=.5):
  class Base:
   def __init__(self,**kwargs):self.score_thr=kwargs['score_thr']
   def postprocess(self,outputs,ratio):return 'original decoder'
  with patch.dict(sys.modules,{'rtmlib':types.SimpleNamespace(YOLOX=Base)}):return detectors.make_detector(score_thr=confidence)
 def test_background_false_people_removed_at_requested_threshold(self):
  raw=np.array([[[28,142,342,632,.871659],[202,138,256,166,.359022],[286,190,335,225,.320855],[0,0,0,0,0]]])
  before=raw.copy();boxes=self.make().postprocess(raw,.5)
  np.testing.assert_array_equal(boxes,[[56,284,684,1264]])
  np.testing.assert_array_equal(raw,before)
  self.assertEqual(len(self.make(.34).postprocess(raw)),2)
 def test_real_second_person_remains_and_empty_is_valid(self):
  raw=np.array([[[0,0,100,200,.9],[200,0,300,200,.8]]])
  self.assertEqual(len(self.make().postprocess(raw)),2)
  self.assertEqual(self.make(.95).postprocess(raw).shape,(0,4))
  self.assertEqual(self.make().postprocess(np.zeros((1,0,5))).shape,(0,4))
 def test_non_nms_decoder_unchanged(self):
  self.assertEqual(self.make().postprocess(np.zeros((1,2,4))),'original decoder')
 def test_actual_failure_wins_over_buffered_load_lines(self):
  message='视频识别失败：视频 9.70 秒（源第 292 帧）检测到多个人物'
  text='WARNING: harmless\n'+message+'\nload detector.onnx with onnxruntime backend\nload pose.onnx with onnxruntime backend\n'
  self.assertEqual(errors.failure_message(text,1),message)
 def test_python_error_and_empty_output(self):
  self.assertEqual(errors.failure_message('Traceback (most recent call last):\nValueError: invalid\nload pose with onnxruntime backend',1),'ValueError: invalid')
  self.assertIn('进程退出 9',errors.failure_message('',9))
  self.assertNotIn('load ',errors.failure_message('load pose with onnxruntime backend',1))

if __name__=='__main__':unittest.main()
