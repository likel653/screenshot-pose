import sys,unittest,tempfile,json
from pathlib import Path
from types import SimpleNamespace
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'screenshot_pose'))
import screen_capture as sc,capture_worker as cw
class CaptureTests(unittest.TestCase):
 def test_reverse_drag_and_bounds(self):
  self.assertEqual(sc.selection_box((90,80),(-20,-10),100,100),(0,0,90,80))
  self.assertIsNone(sc.selection_box((10,10),(12,12),100,100))
 def test_cancel_does_not_consume_existing_image(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);model=root/'m.task';model.write_bytes(b'model')
   called=[];cw.run(root/'screen.png',root/'pose.json',root/'overlay.png',root/'status.json',model,capture_fn=lambda d:None,detect_fn=lambda *a,**k:called.append(1))
   self.assertFalse(called);self.assertFalse((root/'screen.png').exists());self.assertFalse((root/'pose.json').exists());self.assertTrue(json.loads((root/'status.json').read_text())['cancelled'])
 def test_captured_region_goes_to_detection(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);model=root/'m.task';model.write_bytes(b'model')
   def detector(image,model,output,overlay,**kw):
    with Image.open(image) as captured:self.assertEqual(captured.size,(50,60))
    self.assertEqual(kw['person_index'],1);self.assertEqual(kw['confidence'],.4);return {'ok':True}
   result=cw.run(root/'screen.png',root/'pose.json',root/'overlay.png',root/'status.json',model,person_index=1,confidence=.4,capture_fn=lambda d:Image.new('RGB',(50,60),'red'),detect_fn=detector)
   self.assertEqual(result,{'ok':True});self.assertFalse(json.loads((root/'status.json').read_text())['cancelled'])
 def test_withdrawn_overlay_returns_exact_crop(self):
  import tkinter as tk
  root=tk.Tk();root.withdraw();image=Image.new('RGB',(100,120),'blue');image.putpixel((30,40),(255,0,0))
  ui=sc.RegionCapture(root,delay=10,grabber=lambda:image,show_ui=False);ui.begin()
  from PIL import ImageTk
  self.assertEqual(ImageTk.getimage(ui.photo).getpixel((0,0))[:3],(0,0,255))
  self.assertEqual(ImageTk.getimage(ui.photo).getpixel((30,40))[:3],(255,0,0))
  ui.press(SimpleNamespace(x=20,y=30));ui.motion(SimpleNamespace(x=70,y=90));ui.release(SimpleNamespace(x=70,y=90))
  self.assertTrue(ui.finished);self.assertEqual(ui.result.size,(50,60));self.assertEqual(ui.result.getpixel((10,10)),(255,0,0))
 def test_grab_failure_closes_selector(self):
  import tkinter as tk
  def fail():raise OSError('capture unavailable')
  root=tk.Tk();root.withdraw();ui=sc.RegionCapture(root,delay=10,grabber=fail,show_ui=False);ui.begin()
  self.assertTrue(ui.finished);self.assertIsInstance(ui.error,OSError)
if __name__=='__main__':unittest.main()
