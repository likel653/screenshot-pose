import importlib,json,sys,types,tempfile,unittest,math
from pathlib import Path
pkg=types.ModuleType('video_test_pkg');pkg.__path__=[str(Path(__file__).resolve().parents[1]/'screenshot_pose')];sys.modules['video_test_pkg']=pkg
v=importlib.import_module('video_test_pkg.video_data');w=importlib.import_module('video_test_pkg.worker_video')

class VideoTests(unittest.TestCase):
 def test_timing_does_not_change_duration(self):
  plan=v.sample_plan(30.,300,1,4,12)
  self.assertEqual(plan[0],30);self.assertLess(plan[-1],120)
  self.assertEqual(len(v.sample_plan(30,300,1,4,0)),90)
  data=dict(start_seconds=1,frames=[dict(time=i/30,status='ok') for i in plan])
  timed=v.timed_frames(data,24,20)
  self.assertEqual(timed[0][0],20);self.assertLess(abs(timed[-1][0]-(20+3*24)),4)
  self.assertEqual(v.sample_plan(29.97,300,0,0,120),list(range(300)))
 def test_short_gaps_and_trimming(self):
  data=dict(frames=[dict(time=0,status='missing'),dict(time=.1,status='ok'),dict(time=.2,status='missing'),dict(time=.3,status='ok'),dict(time=.4,status='missing')])
  self.assertEqual(len(v.timed_frames(data,30,1,.3)),2)
  with self.assertRaises(ValueError):v.timed_frames(data,30,1,.1)
 def test_limits(self):
  for args in [(0,100),(float('nan'),100),(30,300,5,4),(30,300,0,0,-1),(30,999999,0,0,0)]:
   with self.assertRaises(ValueError):v.sample_plan(*args)
 def test_rotation_sign(self):
  self.assertEqual(v.smooth_quaternion([1,0,0,0],[([1,0,0,0],1),([-1,0,0,0],1)]),[1,0,0,0])
  q=v.smooth_quaternion([1,0,0,0],[([1,0,0,0],2),([math.cos(.2),math.sin(.2),0,0],1)])
  self.assertAlmostEqual(sum(x*x for x in q),1.)
 def test_decoder_cache_and_missing(self):
  import cv2,numpy as np
  with tempfile.TemporaryDirectory() as temp:
   base=Path(temp);source=base/'source.avi';writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'MJPG'),30,(64,64))
   for _ in range(30):writer.write(np.zeros((64,64,3),dtype=np.uint8))
   writer.release();model=base/'model.task';model.write_bytes(b'fixture');seen=[]
   def detect(image,model,output,overlay,**kw):
    seen.append(kw['time'])
    if len(seen)==2:raise ValueError('没有识别到人物')
    w.worker.write_json(output,dict(pose_count=1));return dict(pose_count=1)
   result=w.run(source,base/'cache',model,backend='MEDIAPIPE',sample_fps=5,detect_fn=detect)
   self.assertEqual(len(seen),5);self.assertEqual(result['frames'][1]['status'],'missing')
   self.assertEqual(len(v.read(base/'cache/sequence.json')['frames']),5)
   with self.assertRaises(ValueError):w.run(source,base/'cache',model,detect_fn=detect)
   (base/'outside.json').write_text('{}');result['frames'][0]['pose']='../outside.json';w.worker.write_json(base/'cache/sequence.json',result)
   with self.assertRaises(ValueError):v.read(base/'cache/sequence.json')
 def test_multiple_people_never_silently_switch(self):
  import cv2,numpy as np
  with tempfile.TemporaryDirectory() as temp:
   base=Path(temp);source=base/'source.avi';writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'MJPG'),30,(64,64))
   for _ in range(3):writer.write(np.zeros((64,64,3),dtype=np.uint8))
   writer.release();model=base/'model.task';model.write_bytes(b'fixture')
   def detect(*args,**kwargs):return dict(pose_count=2)
   with self.assertRaisesRegex(ValueError,'多个人物'):w.run(source,base/'cache',model,detect_fn=detect)
   self.assertFalse((base/'cache/sequence.json').exists())

if __name__=='__main__':unittest.main()
