import sys,unittest,json,copy
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'screenshot_pose'))
import worker_gpu,pose_data
class GpuTests(unittest.TestCase):
 def test_lifting_is_root_relative_and_unit_consistent(self):
  xy=np.random.default_rng(2).uniform(50,900,(133,2));z=np.linspace(-.3,.3,133)
  P,c=worker_gpu.lift_points(xy,z,(1000,1000));np.testing.assert_allclose(P[[11,12]].mean(0),np.zeros(3),atol=1e-10)
  self.assertAlmostEqual(P[20,2]-P[3,2],z[20]-z[3])
  self.assertTrue(np.isfinite(P).all());self.assertGreater(c['root_distance'],0)
 def test_invalid_outputs_are_rejected(self):
  with self.assertRaises(ValueError):worker_gpu.lift_points(np.zeros((20,2)),np.zeros(20),(100,100))
  with self.assertRaises(ValueError):worker_gpu.lift_points(np.full((133,2),np.nan),np.zeros(133),(100,100))
 def test_document_preserves_heel_and_both_toes(self):
  root=Path(__file__).resolve().parents[1];p=root/'test_output/rtmw3d_raw.json'
  if not p.is_file():self.skipTest('optional local fixture not present')
  raw=json.loads(p.read_text());d=worker_gpu.make_document(np.array(raw['xy']),np.array(raw['keypoints']),np.array(raw['scores']),(1095,1600),'image.png','original.png')
  pose_data.validate(d);self.assertEqual(d['depth_source'],'rtmw3d_relative_depth')
  self.assertEqual(set(d['orientation']['feet']['L']['points']),{'ankle','heel','big_toe','small_toe'})
  self.assertNotIn('L',d['orientation']['hands']) # Unstable palm proportions trigger a fallback.
 def test_cuda_execution_was_verified(self):
  p=Path(__file__).resolve().parents[1]/'test_output/models/gpu_ready.json'
  if not p.is_file():self.skipTest('optional CUDA probe not present')
  r=json.loads(p.read_text());self.assertTrue(r['ready']);self.assertGreater(r['providers'].get('CUDAExecutionProvider',0),0)
if __name__=='__main__':unittest.main()
