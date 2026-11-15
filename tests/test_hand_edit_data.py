import copy,json,sys,unittest,tempfile,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'screenshot_pose'))
import hand_edit_data as h
class HandEditTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.data=json.loads((ROOT/'tests/fixtures/hand_contract_pose.json').read_text(encoding='utf-8'))
  cls.data['image']['path']=str(ROOT/'tests/fixtures/hand_contract_image.png')
 def test_missing_world_is_explicitly_manual_and_finger_local(self):
  state=h.initialize(self.data,'R');self.assertIn('平面',state['depth_source']);before=copy.deepcopy(self.data)
  x,y=state['xy'][8];h.move(state,8,x+.02,y-.01);result=h.document(self.data,state)
  self.assertEqual(self.data,before)
  for key in self.data:
   if key!='orientation':self.assertEqual(result[key],self.data[key])
  self.assertEqual(result['orientation']['hands']['L'],before['orientation']['hands']['L'])
  self.assertEqual(result['hand_edit']['fingers'],['index']);self.assertFalse(result['hand_edit']['palm_updated'])
  self.assertEqual(result['orientation']['hands']['R']['frame'],self.data['orientation']['hands']['R']['frame'])
 def test_selected_bone_lengths_preserved(self):
  state=h.initialize(self.data,'R');x,y=state['xy'][8];h.move(state,8,x-.025,y-.02,.02);result=h.document(self.data,state)
  old=np.array(state['world_seed'])[5:9];new=np.array(result['orientation']['hands']['R']['world'])[5:9]
  np.testing.assert_allclose(np.linalg.norm(np.diff(old,axis=0),axis=1),np.linalg.norm(np.diff(new,axis=0),axis=1),atol=1e-8)
  for i,limit in ((1,140),(2,110)):
   a=new[i]-new[i-1];b=new[i+1]-new[i];angle=math.degrees(math.acos(float(np.clip(np.dot(a,b)/np.linalg.norm(a)/np.linalg.norm(b),-1,1))))
   self.assertLessEqual(angle,limit+1e-5)
 def test_other_existing_world_points_unchanged(self):
  state=h.initialize(self.data,'L');x,y=state['xy'][8];h.move(state,8,x+.01,y)
  result=h.document(self.data,state);a=result['orientation']['hands']['L']['world'];b=self.data['orientation']['hands']['L']['world']
  for i in set(range(21))-set(range(5,9)):self.assertEqual(a[i],b[i])
 def test_invalid_move_is_atomic_and_wrist_locked(self):
  state=h.initialize(self.data,'R');before=copy.deepcopy(state)
  for args in ((0,.4,.4),(8,float('nan'),.4),(8,.4,.4,2.)):
   with self.assertRaises(ValueError):h.move(state,*args)
   self.assertEqual(state,before)
 def test_save_cancel_and_undo_history(self):
  state=h.initialize(self.data,'R')
  with self.assertRaises(ValueError):h.document(self.data,state)
  x,y=state['xy'][8];h.move(state,8,x+.01,y)
  result=h.document(self.data,state);reverted=h.undo(result,'R')
  self.assertEqual(reverted['orientation'],self.data['orientation'])
  with tempfile.TemporaryDirectory() as temp:
   out=Path(temp)/'out.json';h.save(self.data,state,out,True)
   self.assertTrue(json.loads(out.read_text(encoding='utf8'))['hand_edit']['apply_on_return'])
   with self.assertRaises(ValueError):h.save(self.data,state,out)
 def test_chain_handles_unreachable_target_without_stretch(self):
  q=np.array([[0,0,0],[0,0,.03],[0,0,.05],[0,0,.065]])
  goals=q.copy();goals[3]=[1.,0,1.];out=h.solve_chain(q,goals,[20,.1,.1,12])
  np.testing.assert_allclose(out[0],q[0]);np.testing.assert_allclose(np.linalg.norm(np.diff(out,axis=0),axis=1),[.03,.02,.015],atol=1e-9)
 def test_character_pose_can_seed_editor_without_model_world(self):
  import test_hand_geometry
  p=test_hand_geometry.hand('OPEN');seed=dict(world=p.tolist(),frame=self.data['orientation']['hands']['R']['frame'])
  state=h.initialize(self.data,'R',seed)
  self.assertIn('当前角色',state['initialization']);np.testing.assert_allclose(state['world_seed'],p)
if __name__=='__main__':unittest.main()
