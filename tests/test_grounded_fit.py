import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'screenshot_pose'))
import fit_math as f

def fixture():
 return np.array([[.2,0,1.4],[.45,-.1,1.1],[.6,-.25,.95],[.15,0,.85],[.25,-.4,.55],[.15,0,.08],[-.2,0,1.4],[-.45,-.1,1.1],[-.6,-.25,.95],[-.15,0,.85],[-.25,-.4,.55],[-.15,0,.08]])
class GroundedTests(unittest.TestCase):
 def test_exact_chain_and_unreachable_reporting(self):
  hip=np.array([0.,0.,1.]);ankle=np.array([0.,0.,0.]);pole=np.array([0.,-.4,.5])
  knee,error=f.planted_knee(hip,ankle,pole,.6,.6,.2)
  self.assertLess(error,1e-7);self.assertAlmostEqual(np.linalg.norm(knee-hip),.6,places=7);self.assertAlmostEqual(np.linalg.norm(ankle-knee),.6,places=7)
  _,error=f.planted_knee(hip,ankle*0-[0,0,1.],pole,.6,.6);self.assertGreater(error,.7)
 def test_crouch_translation_preserves_contact_and_lengths(self):
  p=fixture();desired=p+np.array([0.,0.,-.15])
  for r,m,e in [(3,4,5),(9,10,11)]:
   desired[e]=p[e];desired[m],_=f.planted_knee(desired[r],desired[e],desired[m],np.linalg.norm(p[m]-p[r]),np.linalg.norm(p[e]-p[m]))
  obs=desired[:,[0,2]]*[.55,-.55]+[.5,.85]
  result=f.solve_grounded(p,obs,np.ones(12),steps=55)
  self.assertTrue(result['coordination']['accepted'],result['coordination']);self.assertLess(result['after'],result['coordination']['legacy_projection']*.9)
  np.testing.assert_allclose(result['points'][[5,11]],p[[5,11]],atol=1e-10)
  for r,m,e in [(0,1,2),(3,4,5),(6,7,8),(9,10,11)]:
   for a,b in ((r,m),(m,e)):self.assertAlmostEqual(np.linalg.norm(p[a]-p[b]),np.linalg.norm(result['points'][a]-result['points'][b]),places=7)
  delta=result['points']-p;np.testing.assert_allclose(delta[[0,3,6,9]],np.tile(delta[0],(4,1)),atol=1e-9)
 def test_no_gain_falls_back_and_no_input_mutation(self):
  p=fixture();old=p.copy();obs=p[:,[0,2]]*[.55,-.55]+[.5,.85];result=f.solve_grounded(p,obs,np.ones(12),steps=20)
  self.assertFalse(result['coordination']['accepted']);np.testing.assert_array_equal(p,old);self.assertLess(result['after'],1e-8)
 def test_invalid_observations(self):
  with self.assertRaises(ValueError):f.solve_grounded(fixture(),np.zeros((12,2)),np.zeros(12))
 def test_flex_convention(self):
  self.assertAlmostEqual(f.knee_flex(np.array([0,0,1.]),np.array([0,0,.5]),np.zeros(3)),0)
if __name__=='__main__':unittest.main()
