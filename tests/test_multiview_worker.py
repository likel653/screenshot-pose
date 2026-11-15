import copy,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'screenshot_pose'))
import worker_multiview as w
import test_multiview

@unittest.skipUnless(test_multiview.FIXTURE.is_file(),'Local multiview fixture not available')
class MultiViewWorkerTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
  helper=test_multiview.MultiViewTests();helper.primary=json.loads(test_multiview.FIXTURE.read_text(encoding='utf-8'))
  self.base=copy.deepcopy(helper.primary);self.ref=helper.auxiliary()[0]
  self.main=self.root/'main.png';self.main.write_bytes(b'fixture');self.base['image']['path']=str(self.main)
  self.source=self.root/'main.json';w.worker.write_json(self.source,self.base)
  self.model=self.root/'model.task';self.model.write_bytes(b'fixture')
  self.request=self.root/'request.json';self.out=self.root/'output.json';self.overlay=self.root/'overlay.png'
  self.image=self.root/'side.png';self.image.write_bytes(b'fixture')
  for name,value in [('fingerprint','test-model'),('pixels',lambda p:str(Path(p).resolve()))]:
   p=patch.object(w,name,side_effect=value if callable(value) else None,return_value=value);p.start();self.addCleanup(p.stop)
  for name in ('prepare_image','draw_overlay','save_png'):
   p=patch.object(w.worker,name);p.start();self.addCleanup(p.stop)
 def run_job(self,entries,**kwargs):
  self.request.write_text(json.dumps(entries),encoding='utf-8')
  return w.run(self.source,self.request,self.model,kwargs.get('output',self.out),self.overlay,native_root_path=self.root)
 def test_cache_and_individual_failures(self):
  def detect(image,model,output,overlay,**kwargs):
   w.worker.write_json(output,self.ref);return copy.deepcopy(self.ref)
  with patch.object(w.worker_sam,'detect',side_effect=detect) as model:
   entries=[dict(path=str(self.image)),dict(path=str(self.main))]
   first=self.run_job(entries);second=self.run_job(entries)
   self.assertEqual(model.call_count,1)
   self.assertEqual(first['multiview']['accepted'],1);self.assertEqual(second['multiview']['cache_hits'],1)
   self.assertEqual(len(second['multiview']['rejected']),1)
   self.assertEqual(second['full_skeleton'],self.base['full_skeleton'])
 def test_all_failed_preserves_primary(self):
  with patch.object(w.worker_sam,'detect',side_effect=RuntimeError('test inference failure')):
   data=self.run_job([dict(path=str(self.image)),dict(path=str(self.main))])
  self.assertEqual(data['multiview']['accepted'],0);self.assertEqual(len(data['multiview']['rejected']),2)
  for key in self.base:self.assertEqual(data[key],self.base[key])
 def test_no_overwrite_inputs(self):
  with self.assertRaises(ValueError):self.run_job([dict(path=str(self.image))],output=self.source)
  self.assertEqual(json.loads(self.source.read_text(encoding='utf-8')),self.base)

if __name__=='__main__':unittest.main()
