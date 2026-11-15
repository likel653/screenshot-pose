"""Regression: a Windows progress-file lock must not strand the Tk recorder."""
import importlib,sys,types,tempfile,unittest,time,threading,os,json
from pathlib import Path
from unittest.mock import patch
pkg=types.ModuleType('progress_failure_pkg');pkg.__path__=[str(Path(__file__).resolve().parents[1]/'screenshot_pose')];sys.modules['progress_failure_pkg']=pkg
worker=importlib.import_module('progress_failure_pkg.worker');recording=importlib.import_module('progress_failure_pkg.screen_recording');flow=importlib.import_module('progress_failure_pkg.record_worker')

class ProgressFailureTests(unittest.TestCase):
 @unittest.skipUnless(os.name=='nt','Windows destination-handle contention')
 def test_real_shared_read_handle_releases_and_retry_completes(self):
  with tempfile.TemporaryDirectory() as temp:
   path=Path(temp)/'progress.json';worker.write_json(path,dict(old=True));holder=path.open('rb');seen=[]
   real_replace=worker.os.replace
   def replace(source,target):
    try:return real_replace(source,target)
    except PermissionError as exc:seen.append(exc.winerror);raise
   release=threading.Timer(.08,holder.close);release.start()
   try:
    with patch.object(worker.os,'replace',side_effect=replace):worker.write_json(path,dict(done=True))
   finally:release.join();holder.close()
   self.assertTrue(seen);self.assertEqual(json.loads(path.read_text()),dict(done=True))
   self.assertEqual(list(Path(temp).iterdir()),[path])
 def test_progress_failure_is_nonfatal_but_result_write_is_strict(self):
  with tempfile.TemporaryDirectory() as temp:
   path=Path(temp)/'progress.json';worker.write_json(path,dict(old=True));before=path.read_bytes()
   with patch.object(worker.os,'replace',side_effect=PermissionError('persistent denial')):
    self.assertFalse(worker.write_progress(path,dict(new=True)))
    with self.assertRaises(PermissionError):worker.write_json(path,dict(new=True))
   self.assertEqual(path.read_bytes(),before);self.assertEqual(list(Path(temp).iterdir()),[path])
 @unittest.skipUnless(os.name=='nt','Windows sharing violation retry policy')
 def test_persistent_windows_lock_has_bounded_retries(self):
  with tempfile.TemporaryDirectory() as temp:
   path=Path(temp)/'progress.json';worker.write_json(path,dict(old=True));busy=PermissionError('locked');busy.winerror=5
   with patch.object(worker.os,'replace',side_effect=busy) as replace,patch.object(worker.time,'sleep') as sleep:
    self.assertFalse(worker.write_progress(path,dict(done=True)))
    self.assertEqual(replace.call_count,6);self.assertEqual(sleep.call_count,5)
   self.assertEqual(json.loads(path.read_text()),dict(old=True))
 def test_tk_recorder_completes_after_progress_callback_throws(self):
  import tkinter as tk
  root=tk.Tk();root.withdraw();calls=[];timed_out=[]
  def record(*args,progress,**kwargs):
   progress(dict(elapsed=.1));time.sleep(.35);return dict(video='recorded.avi')
  def fail(value):calls.append(value);raise PermissionError('WinError 5 progress replacement denied')
  ui=recording.RecordingWindow(root,(100,100,300,300),Path('unused.avi'),1.,15,progress=fail,show_ui=False,record_fn=record)
  def timeout():timed_out.append(True);root.destroy()
  root.after(2000,timeout);ui.start();root.mainloop()
  self.assertFalse(timed_out);self.assertTrue(calls);self.assertEqual(ui.result,dict(video='recorded.avi'));self.assertIsNone(ui.error)
  self.assertIn('WinError 5',ui.progress_warning)
 def test_queued_completion_wins_over_stale_progress(self):
  import tkinter as tk
  root=tk.Tk();root.withdraw();calls=[]
  ui=recording.RecordingWindow(root,(100,100,300,300),Path('unused.avi'),1.,15,progress=lambda v:calls.append(v),show_ui=False)
  for _ in range(20):ui.events.put(('progress',dict(elapsed=.2)))
  ui.events.put(('done',dict(video='complete.avi')));ui.poll()
  self.assertEqual(ui.result,dict(video='complete.avi'));self.assertFalse(calls)
 def test_progress_failure_does_not_block_record_commit_or_inference(self):
  with tempfile.TemporaryDirectory() as temp:
   out=Path(temp)/'output';calls=[]
   def record(box,path,duration,fps,progress):
    path.write_bytes(b'completed recording');self.assertFalse(progress(dict(elapsed=.1)))
    return dict(video=str(path),video_seconds=duration,width=64,height=64,fps=fps)
   def infer(video,output,model,**kwargs):
    output.mkdir();(output/'sequence.json').write_text('{}');calls.append(True)
   with patch.object(flow.worker,'write_progress',return_value=False):
    result=flow.run(out,identify=True,select_fn=lambda:(0,0,64,64),record_fn=record,detect_fn=infer)
   self.assertTrue(calls);self.assertEqual(result['sequence'],'poses/sequence.json')
   self.assertEqual(json.loads((out/'result.json').read_text())['sequence'],'poses/sequence.json')
