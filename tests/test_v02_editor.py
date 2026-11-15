import copy,json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'screenshot_pose'))
import pose_data,image_editor

FIXTURE=ROOT/'test_output/user_crouch/pose.json'
@unittest.skipUnless(FIXTURE.is_file(),'Local screenshot editor fixture not available')
class CorrectionTests(unittest.TestCase):
    def setUp(self):self.data=pose_data.read_pose(FIXTURE)
    def test_wrist_collision_flag_is_diagnostic_only(self):
        before=copy.deepcopy(self.data)
        self.assertTrue(any('双手腕' in n for n in pose_data.diagnostics(self.data)))
        self.assertEqual(self.data,before)
    def test_move_wrist_carries_only_its_hand(self):
        before=copy.deepcopy(self.data)
        image_editor.move_point(self.data,15,.318,.562)
        delta=.318-before['landmarks'][15]['x']
        for i in (15,17,19,21):self.assertAlmostEqual(self.data['landmarks'][i]['x']-before['landmarks'][i]['x'],delta)
        self.assertEqual(self.data['landmarks'][16],before['landmarks'][16])
        self.assertFalse(any('双手腕' in n for n in pose_data.diagnostics(self.data)))
    def test_hybrid_corrected_wrist_projection(self):
        image_editor.move_point(self.data,15,.318,.562)
        joints,_=pose_data.to_joints(self.data,'HYBRID')
        actual=joints['wrist.L'][2]-joints['elbow.L'][2]
        self.assertLess(abs(actual),.03)
        world,_=pose_data.to_joints(self.data,'WORLD')
        self.assertAlmostEqual(joints['wrist.L'][1],world['wrist.L'][1])
    def test_tk_editor_controls_save_and_reload(self):
        import tkinter as tk
        root=tk.Tk();root.withdraw()
        try:
            with tempfile.TemporaryDirectory(dir=ROOT/'test_output') as tmp:
                path=Path(tmp)/'校正.json'
                e=image_editor.Editor(root,self.data,path)
                e.choice.set('左手腕');e.select_name()
                self.assertEqual(e.selected,15)
                e.depth.set('0.15');e.set_depth()
                self.assertAlmostEqual(e.data['world_landmarks'][15]['z'],.15)
                e.undo();self.assertEqual(e.data,self.data)
                image_editor.move_point(e.data,15,.318,.562)
                e.save()
                self.assertEqual(pose_data.read_pose(path)['suggested_mode'],'HYBRID')
                self.assertTrue(path.with_suffix('.png').is_file())
        finally:
            try:root.destroy()
            except tk.TclError:pass

if __name__=='__main__':unittest.main()
