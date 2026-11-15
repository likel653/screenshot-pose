import importlib,sys,types,tempfile,unittest,threading,json
from pathlib import Path
import numpy as np,cv2
pkg=types.ModuleType('record_test_pkg');pkg.__path__=[str(Path(__file__).resolve().parents[1]/'screenshot_pose')];sys.modules['record_test_pkg']=pkg
r=importlib.import_module('record_test_pkg.screen_recording');w=importlib.import_module('record_test_pkg.record_worker');sc=r.screen_capture

class Clock:
 def __init__(self):self.time=0.
 def now(self):return self.time
 def wait(self,seconds):self.time+=max(0,seconds)+1e-7
class Grabber:
 def __init__(self,clock,width=64,height=64,cost=.001):self.clock=clock;self.width=width;self.height=height;self.cost=cost;self.count=0
 def grab(self):self.clock.time+=self.cost;self.count+=1;return np.full((self.height,self.width,3),self.count*10%255,dtype=np.uint8)

class RecordTests(unittest.TestCase):
 def test_negative_monitor_origin_and_unchanged_selection_pixels(self):
  import tkinter as tk
  from PIL import Image,ImageTk
  from types import SimpleNamespace
  root=tk.Tk();root.withdraw();image=Image.new('RGB',(100,120),(50,80,120))
  ui=sc.RegionCapture(root,10,grabber=lambda:image,show_ui=False,mode='VIDEO',origin=(-1920,-200));ui.begin()
  self.assertEqual(ImageTk.getimage(ui.photo).getpixel((50,50))[:3],(50,80,120))
  ui.press(SimpleNamespace(x=20,y=30));ui.release(SimpleNamespace(x=80,y=100))
  self.assertEqual(ui.selected_box,(-1900,-170,-1840,-100));self.assertIsNone(ui.result)
 def test_control_window_is_outside_selected_region(self):
  box=(-1600,100,-500,900);bounds=(-1920,0,1920,1080);x,y=r.panel_position(box,bounds)
  self.assertTrue(x+400<=box[0] or x>=box[2] or y+155<=box[1] or y>=box[3])
  self.assertIsNone(r.panel_position(bounds,bounds))
 def test_cfr_duration_and_even_padding(self):
  with tempfile.TemporaryDirectory() as temp:
   clock=Clock();video=Path(temp)/'录制.avi'
   stats=r.record((-100,0,-35,67),video,1.,10,grabber=Grabber(clock,65,67),clock=clock.now,wait=clock.wait,escape=False)
   self.assertEqual(stats['frames'],10);self.assertEqual(stats['video_seconds'],1.)
   cap=cv2.VideoCapture(str(video));self.assertEqual((cap.get(3),cap.get(4)),(66,68));cap.release()
   self.assertFalse(video.with_name('录制.partial.avi').exists())
 def test_slow_capture_does_not_speed_up_video(self):
  with tempfile.TemporaryDirectory() as temp:
   clock=Clock();stats=r.record((0,0,64,64),Path(temp)/'video.avi',1.,10,grabber=Grabber(clock,cost=.16),clock=clock.now,wait=clock.wait,escape=False)
   self.assertEqual(stats['frames'],10);self.assertGreater(stats['duplicates'],0);self.assertGreaterEqual(stats['elapsed_seconds'],1.)
 def test_cancel_removes_unfinished_video(self):
  with tempfile.TemporaryDirectory() as temp:
   clock=Clock();stop=threading.Event();video=Path(temp)/'video.avi'
   with self.assertRaises(r.RecordingCancelled):r.record((0,0,64,64),video,1.,10,stop=stop,progress=lambda s:stop.set(),grabber=Grabber(clock),clock=clock.now,wait=clock.wait,escape=False)
   self.assertFalse(video.exists());self.assertFalse(list(Path(temp).glob('*.partial.avi')))
 def test_invalid_options_and_no_overwrite(self):
  for duration,fps in [(float('nan'),15),(0.,15),(121.,15),(1.,0),(1.,True)]:
   with self.assertRaises(ValueError):r.check_options((0,0,64,64),duration,fps)
  with tempfile.TemporaryDirectory() as temp:
   p=Path(temp)/'video.avi';p.write_bytes(b'original')
   with self.assertRaises(ValueError):r.record((0,0,64,64),p)
   self.assertEqual(p.read_bytes(),b'original')

class WorkflowTests(unittest.TestCase):
 def recorder(self,box,path,duration,fps,progress):
  path.write_bytes(b'recording fixture');return dict(video=str(path),video_seconds=duration,width=64,height=64,fps=fps,frames=round(duration*fps),duplicates=0)
 def test_cancel_selection_never_calls_recorder(self):
  with tempfile.TemporaryDirectory() as temp:
   out=Path(temp)/'out';w.run(out,select_fn=lambda:None,record_fn=lambda *a:1/0)
   self.assertTrue(json.loads((out/'result.json').read_text())['cancelled']);self.assertFalse((out/'screen.avi').exists())
 def test_record_only_and_inference_failure_preserve_video(self):
  with tempfile.TemporaryDirectory() as temp:
   for identify in (False,True):
    out=Path(temp)/str(identify)
    def fail(*a,**k):raise ValueError('model absent')
    final=w.run(out,identify=identify,select_fn=lambda:(0,0,64,64),record_fn=self.recorder,detect_fn=fail)
    self.assertTrue((out/'screen.avi').is_file());self.assertEqual(final['sequence'],None)
    self.assertEqual('inference_error' in final,identify)
 def test_success_relays_progress_and_commits_sequence(self):
  with tempfile.TemporaryDirectory() as temp:
   out=Path(temp)/'out'
   def detect(video,output,model,**kwargs):
    self.assertEqual(kwargs['progress_path'],out/'progress.json');output.mkdir();(output/'sequence.json').write_text('{}')
   final=w.run(out,identify=True,select_fn=lambda:(0,0,64,64),record_fn=self.recorder,detect_fn=detect)
   self.assertEqual(final['sequence'],'poses/sequence.json');self.assertFalse(final['cancelled'])
