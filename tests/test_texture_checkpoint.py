import sys,tempfile,unittest,zipfile,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'screenshot_pose'))
from setup_hand_texture import verify_archive,cache_checkpoint
from hand3d_runner import inference_weights

class CheckpointIntegrityTests(unittest.TestCase):
 def test_cache_is_independent_and_does_not_replace_mismatched_existing_file(self):
  with tempfile.TemporaryDirectory() as tmp:
   source=Path(tmp)/'download.ckpt';source.write_bytes(b'checkpoint bytes');sha=hashlib.sha256(source.read_bytes()).hexdigest()
   cached=cache_checkpoint(Path(tmp)/'runtime',source,sha);self.assertNotEqual(source,cached);self.assertEqual(cached.read_bytes(),source.read_bytes())
   self.assertEqual(cache_checkpoint(Path(tmp)/'runtime',source,sha),cached)
   cached.write_bytes(b'wrong contents')
   with self.assertRaises(ValueError):cache_checkpoint(Path(tmp)/'runtime',source,sha)
   self.assertEqual(source.read_bytes(),b'checkpoint bytes')
 def test_texture_training_modules_excluded_but_unknown_and_core_preserved(self):
  state={'backbone.w':1,'mano_head.w':2,'mano.w':3,'recoLoss.net.w':4,'texture_model.w':5,'unknown.w':6}
  kept,ignored=inference_weights(state,'HAMER_TEXTURE')
  self.assertEqual(kept,{'backbone.w':1,'mano_head.w':2,'mano.w':3,'unknown.w':6});self.assertEqual(len(ignored),2)
  original,ignored=inference_weights(state,'HAMER');self.assertEqual(original,state);self.assertFalse(ignored)
 def test_truncated_tensor_archive_is_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'weights.ckpt'
   with zipfile.ZipFile(path,'w') as archive:
    archive.writestr('archive/data.pkl',b'fixture metadata');archive.writestr('archive/data/0',b'123456789')
   original=path.read_bytes();path.write_bytes(original[:-45]);before=path.read_bytes()
   with self.assertRaisesRegex(ValueError,'不完整|损坏'):verify_archive(path)
   self.assertEqual(path.read_bytes(),before)
 def test_complete_archive_is_crc_verified(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'weights.ckpt'
   with zipfile.ZipFile(path,'w') as archive:archive.writestr('archive/data.pkl',b'metadata');archive.writestr('archive/data/0',b'tensor')
   self.assertTrue(verify_archive(path)['crc_checked'])
 def test_non_checkpoint_zip_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'source.zip'
   with zipfile.ZipFile(path,'w') as archive:archive.writestr('README.md','source code only')
   with self.assertRaisesRegex(ValueError,'检查点'):verify_archive(path)

if __name__=='__main__':unittest.main()
