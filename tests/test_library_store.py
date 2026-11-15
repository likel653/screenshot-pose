import importlib,sys,types,json,tempfile,unittest,shutil
from pathlib import Path
pkg=types.ModuleType('library_test_pkg');pkg.__path__=[str(Path(__file__).resolve().parents[1]/'screenshot_pose')];sys.modules['library_test_pkg']=pkg
m=importlib.import_module('library_test_pkg.library_store')
class StoreTest(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name);self.root=self.base/'library'
  self.image=self.base/'shot.png';self.image.write_bytes(b'image fixture')
  self.doc={'schema':'screenshot_pose/v1','image':{'width':100,'height':200,'path':str(self.image),'original_path':str(self.image)},'landmarks':[{'x':0.,'y':0.,'z':0.} for _ in range(33)],'world_landmarks':[{'x':0.,'y':0.,'z':0.} for _ in range(33)]}
 def tearDown(self):self.temp.cleanup()
 def test_copy_relocate_materialize_and_source_independence(self):
  meta=m.save(self.root,self.doc,self.image,'抱膝','坐姿')
  self.image.unlink();other=self.base/'relocated';shutil.copytree(self.root,other)
  meta,doc=m.materialize(other,meta['id'],self.base/'working')
  self.assertTrue(Path(doc['image']['path']).is_file());self.assertEqual(Path(doc['image']['path']).read_bytes(),b'image fixture')
  m.trash(other,meta['id']);self.assertTrue(Path(doc['image']['path']).is_file())
 def test_names_search_rename_and_recoverable_trash(self):
  meta=m.save(self.root,self.doc,self.image,'抱膝','坐姿');m.save(self.root,self.doc,self.image,'抱膝','另一个版本')
  self.assertEqual(len(m.list_entries(self.root,'坐姿')[0]),1)
  m.rename(self.root,meta['id'],'侧坐','室内','备注')
  self.assertEqual(len(m.list_entries(self.root,'室内')[0]),1)
  path=m.trash(self.root,meta['id']);self.assertTrue((path/'pose.json').exists());self.assertEqual(len(m.list_entries(self.root)[0]),1)
 def test_traversal_and_corrupt_entries(self):
  for key in ('../oops','C:/oops','','a'*31):
   with self.assertRaises(ValueError):m.entry(self.root,key)
  meta=m.save(self.root,self.doc,self.image,'好姿势');folder=self.root/meta['id'];r=json.loads((folder/'entry.json').read_text(encoding='utf8'));r['image']='../outside.png';m.write(folder/'entry.json',r)
  rows,bad=m.list_entries(self.root);self.assertEqual(rows,[]);self.assertEqual(bad,1)
 def test_failed_save_leaves_no_partial_entry(self):
  with self.assertRaises(ValueError):m.save(self.root,self.doc,self.image,'')
  self.assertEqual(m.list_entries(self.root)[0],[])
 def test_original_resolution_and_hand_reference_travel_with_entry(self):
  original=self.base/'original.jpg';original.write_bytes(b'original full resolution')
  hand=self.base/'hand.jpg';hand.write_bytes(b'hand reference')
  self.doc['image']['original_path']=str(original);self.doc['orientation']={'hands':{'L':{'reference_image':str(hand)}}}
  meta=m.save(self.root,self.doc,self.image,'带局部参考')
  original.unlink();hand.unlink()
  _,doc=m.materialize(self.root,meta['id'],self.base/'work')
  self.assertEqual(Path(doc['image']['original_path']).read_bytes(),b'original full resolution')
  self.assertEqual(Path(doc['orientation']['hands']['L']['reference_image']).read_bytes(),b'hand reference')
 def test_calibration_survives_library_materialization(self):
  import hashlib
  sha=hashlib.sha256(self.image.read_bytes()).hexdigest()
  self.doc['camera_calibration']=dict(schema='screenshot_pose/camera_v1',valid=True,image_size=[100,200],
   intrinsics=[120,140,50,100],processed_file_sha256=sha)
  meta=m.save(self.root,self.doc,self.image,'标定姿势',options={'fit_model':'PERSPECTIVE'})
  loaded,doc=m.materialize(self.root,meta['id'],self.base/'calibrated')
  self.assertEqual(loaded['options']['fit_model'],'PERSPECTIVE')
  self.assertEqual(doc['camera_calibration']['processed_file_sha256'],hashlib.sha256(Path(doc['image']['path']).read_bytes()).hexdigest())
if __name__=='__main__':unittest.main()
