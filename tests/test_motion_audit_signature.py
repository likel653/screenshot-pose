import sys,json,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from motion_audit_signature import input_signature

class AuditSignatureTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
  (self.root/'screenshot_pose').mkdir();(self.root/'screenshot_pose/a.py').write_text('version=1')
  (self.root/'scene.blend').write_bytes(b'scene');(self.root/'pose.json').write_text('{}')
  self.manifest=self.root/'sequence.json';self.manifest.write_text(json.dumps(dict(frames=[dict(status='ok',pose='pose.json')])))
  self.config=dict(cache=str(self.manifest),template=str(self.root/'scene.blend'))
 def tearDown(self):self.temp.cleanup()
 def test_pose_changes_invalidate_checkpoint(self):
  before=input_signature(self.config,self.root);(self.root/'pose.json').write_text('{"changed":1}')
  self.assertNotEqual(before,input_signature(self.config,self.root))
 def test_plugin_changes_invalidate_checkpoint(self):
  before=input_signature(self.config,self.root);(self.root/'screenshot_pose/a.py').write_text('version=2')
  self.assertNotEqual(before,input_signature(self.config,self.root))
 def test_cache_cannot_read_parent_file(self):
  self.manifest.write_text(json.dumps(dict(frames=[dict(status='ok',pose='../outside.json')])))
  with self.assertRaises(ValueError):input_signature(self.config,self.root)

if __name__=='__main__':unittest.main()
