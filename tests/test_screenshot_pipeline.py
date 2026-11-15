import sys,json,copy,tempfile,unittest
import numpy as np
from pathlib import Path
from unittest.mock import patch
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'screenshot_pose'))
import worker_screenshot as w,pose_data,worker,sam_data,screenshot_recovery
class PipelineTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.image=self.root/'in.png';Image.new('RGB',(100,160),'gray').save(self.image)
  self.model=self.root/'model.task';self.model.write_bytes(b'model');self.output=self.root/'pose.json';self.overlay=self.root/'overlay.png';self.closed=[]
 def body(self,image,model,output,overlay,**kwargs):
  d=pose_data.template_pose(image,100,160);d['orientation']={'hands':{'R':{'source':'old_R'},'L':{'source':'old_L'}}};worker.write_json(output,d);Image.new('RGB',(100,160)).save(overlay);return d
 def session(self,*args,**kwargs):
  outer=self
  class Session:
   def close(self):outer.closed.append(True)
  return Session()
 def run_job(self,mode='AUTO',**kwargs):return w.detect(self.image,self.model,self.output,self.overlay,hand_mode=mode,body_detect=self.body,session_factory=self.session,**kwargs)
 def test_off_does_not_load_hands(self):
  with patch.object(w.hand3d,'readiness',side_effect=AssertionError('must not load')):d=self.run_job('OFF')
  self.assertEqual(d['automatic_hands']['status'],'off');self.assertTrue(self.output.is_file())
 def test_missing_model_keeps_body_and_old_observations(self):
  with patch.object(w.hand3d,'readiness',return_value=dict(ready=False,missing=['MANO'])):d=self.run_job()
  self.assertEqual(d['orientation']['hands']['R']['source'],'old_R');self.assertEqual(d['automatic_hands']['status'],'unavailable')
 def test_commit_occurs_only_after_refinement_and_rejected_hand_preserved(self):
  def refine(base,backend,session):
   self.assertFalse(self.output.exists());d=copy.deepcopy(base);d['orientation']['hands']['L']={'source':'new_cuda'}
   return d,{'L':dict(adopted=True),'R':dict(adopted=False,reason='occluded')}
  with patch.object(w.hand3d,'readiness',return_value=dict(ready=True)),patch.object(w.hand3d,'refine',side_effect=refine):d=self.run_job()
  self.assertEqual(d['orientation']['hands']['R']['source'],'old_R');self.assertEqual(d['orientation']['hands']['L']['source'],'new_cuda');self.assertEqual(self.closed,[True])
 def test_runtime_failure_is_reported_and_session_closed(self):
  with patch.object(w.hand3d,'readiness',return_value=dict(ready=True)),patch.object(w.hand3d,'refine',side_effect=RuntimeError('OOM')):d=self.run_job()
  self.assertEqual(d['automatic_hands']['status'],'failed');self.assertEqual(self.closed,[True]);self.assertEqual(d['orientation']['hands']['L']['source'],'old_L')
 def test_hand_stage_cannot_change_body(self):
  def bad(base,*args):d=copy.deepcopy(base);d['landmarks'][0]['x']+=.1;return d,{}
  with patch.object(w.hand3d,'readiness',return_value=dict(ready=True)),patch.object(w.hand3d,'refine',side_effect=bad):
   with self.assertRaises(AssertionError):self.run_job()
  self.assertFalse(self.output.exists())
 def test_progress_cannot_overwrite_input(self):
  original=self.image.read_bytes()
  with self.assertRaises(ValueError):self.run_job(progress=self.image)
  self.assertEqual(self.image.read_bytes(),original)
 def test_existing_result_cannot_be_overwritten(self):
  self.output.write_text('original')
  with self.assertRaises(ValueError):self.run_job()
  self.assertEqual(self.output.read_text(),'original')
 def test_sam_confirms_legs_piled_up_outside_the_picture(self):
  d=self.body(self.image,self.model,self.output,self.overlay)
  for i in range(25,33):d['landmarks'][i]['y']=.97
  mhr=np.full((1,70,2),80.);mhr[0,11:21,1]=185.
  sam_data.check_screenshot_observations(d,dict(keypoints_pixels=mhr))
  self.assertTrue(all(d['landmarks'][i].get('outside_frame') for i in range(25,33)))
  self.assertFalse(d['landmarks'][13].get('outside_frame',False))
  self.assertEqual(d['landmarks'][25]['visibility'],0.)
 def test_sam_does_not_hide_visible_legs(self):
  d=self.body(self.image,self.model,self.output,self.overlay)
  mhr=np.full((1,70,2),80.);mhr[0,11:21,1]=185.
  sam_data.check_screenshot_observations(d,dict(keypoints_pixels=mhr))
  self.assertFalse(any(p.get('outside_frame') for p in d['landmarks']))
 def test_sam_flags_only_the_foot_with_a_model_disagreement(self):
  d=self.body(self.image,self.model,self.output,self.overlay)
  mhr=np.full((1,70,2),80.)
  for indices,target in (((27,29,31),(13,17,15)),((28,30,32),(14,20,18))):
   for i,j in zip(indices,target):
    p=d['landmarks'][i];mhr[0,j]=[p['x']*100,p['y']*160]
  mhr[0,16]=mhr[0,15]
  mhr[0,19]=mhr[0,18]
  mhr[0,15:17,0]-=15
  sam_data.check_screenshot_observations(d,dict(keypoints_pixels=mhr))
  self.assertEqual(d['foot_review_sides'],['L'])
 def test_distinct_ankles_can_be_reassigned_only_with_matching_hips(self):
  d=pose_data.template_pose(width=1000,height=800)
  d['wholebody']=[dict(x=i/140,y=.5,confidence=.9) for i in range(133)]
  d['body_candidates']=[dict(source='sam3d_mhr_depth'),dict(source='rtmw3d_relative_depth')]
  for i,x in ((23,400),(24,620),(25,340),(26,670),(27,810),(28,190)):
   d['landmarks'][i].update(x=x/1000,y=.5)
  mhr=np.full((70,2),400.)
  mhr[[9,10,11,12,13,14],0]=[400,620,340,670,190,810]
  self.assertTrue(screenshot_recovery.leg_evidence(d,mhr)['reassigned'])
  mhr[[9,10],0]=[620,400]
  self.assertFalse(screenshot_recovery.leg_evidence(d,mhr))
  hips=copy.deepcopy(d['landmarks'][23:25])
  world=copy.deepcopy(d['world_landmarks'])
  screenshot_recovery.reassign_legs(d)
  self.assertEqual([p['x'] for p in d['landmarks'][27:29]],[.19,.81])
  self.assertEqual(d['landmarks'][23:25],hips)
  self.assertEqual(d['world_landmarks'],world)
  self.assertEqual(len(d['body_candidates']),1)
if __name__=='__main__':unittest.main()
