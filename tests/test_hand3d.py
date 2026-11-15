"""Coordinate/identity gates and non-destructive model comparison contracts."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'screenshot_pose'))
import hand3d as h
import worker_hand3d as jobs
import worker_hands
import orientation_math as om
from test_hand_geometry import hand


class Hand3DTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.image = self.root / 'input.png'
        Image.new('RGB', (600, 600), '#384040').save(self.image)
        self.base = json.loads((ROOT / 'tests/fixtures/hand_contract_pose.json').read_text(encoding='utf8'))
        self.base['image'] = dict(path=str(self.image), width=600, height=600)
        self.xy, self.raw = {}, {}
        self.points = hand('OPEN')
        for side, start, wi, cx in [('L', 91, 15, 420), ('R', 112, 16, 180)]:
            sign = 1 if side == 'R' else -1
            p = self.points.copy()
            p[:, 0] *= sign
            xy = p[:, [0, 2]] * [1100, -1100] + [cx, 300]
            self.xy[side] = xy
            self.raw[side] = dict(joints_cv=(self.points @ om.C_DOWN).tolist(), landmarks_px=xy.tolist(), device='cuda')
            self.base['landmarks'][wi].update(x=xy[0, 0]/600, y=xy[0, 1]/600)
            for i, (x, y) in enumerate(xy):
                self.base['wholebody'][start+i] = dict(x=x/600, y=y/600, confidence=.9)
        self.input = self.root / 'source.json'
        jobs.worker.write_json(self.input, self.base)
        self.request = self.root / 'request.json'
        self.output = self.root / 'job' / 'pose.json'
        self.model = self.root / 'model.task'
        self.model.write_bytes(b'base model')

    def test_edited_wrist_uses_current_anchors_for_that_hand(self):
        from image_editor import move_point

        original = h.hand_box(self.base, 'L')
        opposite = h.hand_box(self.base, 'R')
        for i, (x, y) in zip((15, 17, 19, 21), self.xy['L'][[0, 17, 5, 1]]):
            move_point(self.base, i, (x - 150) / 600, y / 600, carry_hand=False)
        current = h.hand_box(self.base, 'L')
        self.assertEqual(current['anchor_source'], 'edited_hand_anchors')
        self.assertIsNone(current['refs'])
        self.assertLess(current['box'][2], original['box'][0])
        unchanged_hand = h.hand_box(self.base, 'R')
        self.assertEqual(unchanged_hand.pop('other_wrist'), current['wrist'])
        opposite.pop('other_wrist')
        self.assertEqual(unchanged_hand, opposite)

    def test_face_region_cannot_claim_a_foot_or_the_opposite_arm(self):
        location = h.hand_box(self.base, 'L')
        location['regional_search'] = dict(head=[[300,150]], elbows=[[340,200],[120,220]],
                                           feet=[[280,350]])
        hand = dict(box=[280,125,330,177],right=1,score=.68)
        sole = dict(box=[245,315,320,390],right=0,score=.91)
        selected = h.regional_detection(location, [sole, hand])
        self.assertEqual(selected['box'], hand['box'])
        self.assertFalse(selected['detector_side_agrees'])
        location['regional_search']['elbows'].reverse()
        self.assertIsNone(h.regional_detection(location, [sole, hand]))

    def session(self):
        raw = self.raw
        class Fake:
            def __init__(self): self.calls=[]; self.closed=False
            def infer(self, backend, image, locations):
                self.calls.append((backend, len(locations)))
                return [copy.deepcopy(raw['R' if v['right'] else 'L']) for v in locations]
            def close(self):self.closed=True
        return Fake()

    def test_left_and_right_have_one_reflection_and_proper_frames(self):
        right = h.observation(self.raw['R'], 'R', h.hand_box(self.base, 'R'), 'WILOR')
        left = h.observation(self.raw['L'], 'L', h.hand_box(self.base, 'L'), 'HAMER')
        np.testing.assert_allclose(right['world'], self.points)
        mirrored = self.points * [-1, 1, 1]
        np.testing.assert_allclose(left['world'], mirrored)
        for value in (right, left):
            self.assertAlmostEqual(np.linalg.det(value['frame']), 1)
            np.testing.assert_allclose(np.asarray(value['frame']).T @ value['frame'], np.eye(3), atol=1e-8)
            self.assertEqual(value['confidence_kind'], 'geometric_gate_not_model_probability')

    def test_canonical_joint_order_and_curls_preserved(self):
        result = h.observation(self.raw['R'], 'R', h.hand_box(self.base, 'R'), 'HAMER')
        for start in (1, 5, 9, 13, 17):
            np.testing.assert_allclose(np.asarray(result['world'])[start:start+4], self.points[start:start+4])
        self.assertEqual(result['curls'], om.curls(self.points))

    def test_wrong_hand_invalid_geometry_and_cpu_rejected(self):
        loc = h.hand_box(self.base, 'R')
        for raw in [dict(self.raw['R'], device='cpu'),
                    dict(self.raw['R'], joints_cv=np.zeros((21, 3)).tolist()),
                    dict(self.raw['R'], landmarks_px=self.xy['L'].tolist()),
                    dict(self.raw['R'], joints_cv=np.full((21, 3), np.nan).tolist())]:
            with self.assertRaises(ValueError):h.observation(raw, 'R', loc, 'WILOR')

    def test_detector_association_uses_anatomical_side_and_wrist_region(self):
        loc=h.hand_box(self.base,'R');box=loc['box']
        good=dict(box=box,score=.9,right=1)
        far=dict(box=[10,10,40,40],score=.99,right=1)
        wrong=dict(box=box,score=.99,right=0)
        result=h.associate_detection(loc,[far,wrong,good])
        self.assertEqual(result['box'],box)
        self.assertEqual(result['source'],'wilor_hand_detector')
        self.assertIsNone(h.associate_detection(loc,[wrong]))

    def test_independent_detector_can_replace_bad_palm_anchors_but_not_wrong_wrist(self):
        loc=h.hand_box(self.base,'R');loc['refs']=(np.asarray(loc['refs'])+80).tolist()
        raw=copy.deepcopy(self.raw['R'])
        raw['localization']=dict(source='wilor_hand_detector',box=loc['box'],confidence=.9)
        value=h.observation(raw,'R',loc,'WILOR')
        self.assertEqual(value['localization']['source'],'wilor_hand_detector')
        raw['landmarks_px']=self.xy['L'].tolist()
        with self.assertRaises(ValueError):h.observation(raw,'R',loc,'WILOR')

    def test_small_or_uncertain_hand_does_not_force_prediction(self):
        for i in range(91, 112):self.base['wholebody'][i].update(x=.5, y=.5)
        self.base['landmarks'][15].update(x=.5, y=.5)
        with self.assertRaises(ValueError):h.hand_box(self.base, 'L')

    def test_failed_hand_and_body_preserved_selected_hand_changes_only(self):
        session = self.session()
        session.infer = lambda *args: [dict(self.raw['R'], joints_cv=np.zeros((21, 3)).tolist())]
        result, reports = h.refine(self.base, 'WILOR', session, sides=('R',))
        self.assertEqual(result['orientation'], self.base['orientation'])
        self.assertFalse(reports['R']['adopted'])
        result, reports = h.refine(self.base, 'WILOR', self.session(), sides=('L',))
        self.assertTrue(reports['L']['adopted'])
        self.assertEqual(result['orientation']['hands']['R'], self.base['orientation']['hands']['R'])
        for key in ('full_skeleton','landmarks','world_landmarks','body_candidates'):
            self.assertEqual(result[key], self.base[key])

    def test_single_lucky_crop_cannot_resurrect_invalid_default(self):
        good=copy.deepcopy(self.raw['R'])
        invalid=dict(good,joints_cv=np.zeros((21,3)).tolist())
        variants=dict(invalid,crop_candidates=[invalid,good])
        session=self.session();session.infer=lambda *args:[variants]
        result,report=h.refine(self.base,'WILOR',session,sides=('R',))
        self.assertFalse(report['R']['adopted'])
        self.assertEqual(result['orientation'],self.base['orientation'])

    def test_crop_choice_uses_independent_fingers_not_own_confidence(self):
        best=copy.deepcopy(self.raw['R']);worst=copy.deepcopy(best)
        # The same camera-space hand; only the independent reference fit differs.
        worst['landmarks_px']=(np.array(worst['landmarks_px'])+[18,0]).tolist()
        worst['crop_candidates']=[best,copy.deepcopy(worst)]
        session=self.session();session.infer=lambda *args:[worst]
        result,report=h.refine(self.base,'WILOR',session,sides=('R',))
        self.assertTrue(report['R']['adopted'])
        self.assertEqual(report['R']['crop_selection']['chosen'],1)
        np.testing.assert_allclose(result['orientation']['hands']['R']['world'],self.points)

    def run_job(self, backend='COMPARE', ready=True):
        self.request.write_text(json.dumps(dict(R=dict(backend=backend, gesture='NONE'))), encoding='utf8')
        with patch.object(jobs.hand3d, 'readiness', return_value=dict(ready=ready, missing=[] if ready else ['MANO_RIGHT.pkl'])):
            return jobs.run(self.input, self.request, self.model, self.output, session=self.session())

    def test_compare_saves_independent_candidates_without_picking_winner(self):
        result = self.run_job()
        self.assertEqual(result['orientation'], self.base['orientation'])
        for backend in h.BACKENDS:
            path = Path(result['hand_refinement']['candidates'][backend])
            candidate = jobs.pose_data.read_pose(path)
            self.assertEqual(candidate['orientation']['hands']['R']['source'], backend.lower()+'_hand3d')
            self.assertTrue(path.with_suffix('.report.json').is_file())
            for key in ('full_skeleton','landmarks','world_landmarks'):
                self.assertEqual(candidate[key], self.base[key])
        self.assertTrue(self.output.with_suffix('.html').is_file())
        self.assertFalse(result['hand_refinement']['hands']['R']['adopted'])

    def test_missing_assets_are_not_claimed_as_model_results(self):
        result = self.run_job(ready=False)
        self.assertEqual(result['orientation'], self.base['orientation'])
        for backend in h.BACKENDS:
            report = result['hand_refinement']['candidate_reports'][backend]['R']
            self.assertFalse(report['adopted'])
            self.assertIn('MANO_RIGHT.pkl', report['reason'])
            self.assertNotIn('source', report)

    def test_single_backend_runs_without_mediapipe(self):
        result = self.run_job('WILOR')
        self.assertTrue(result['hand_refinement']['hands']['R']['adopted'])
        self.assertEqual(result['orientation']['hands']['R']['source'], 'wilor_hand3d')

    def test_existing_results_and_inputs_cannot_be_overwritten(self):
        self.request.write_text(json.dumps(dict(R=dict(backend='WILOR'))), encoding='utf8')
        for path in (self.input, self.model, self.image):
            with self.assertRaises(ValueError):jobs.run(self.input, self.request, self.model, path)
        self.run_job()
        with self.assertRaises(ValueError):self.run_job()

    def test_unknown_or_mixed_backends_rejected(self):
        for req in (dict(R=dict(backend='BOGUS')), dict(R=dict(backend='WILOR'),L=dict(backend='HAMER'))):
            self.request.write_text(json.dumps(req), encoding='utf8')
            with self.assertRaises(ValueError):jobs.run(self.input,self.request,self.model,self.output)

    def test_texture_model_has_distinct_source_label(self):
        value=h.observation(self.raw['R'],'R',h.hand_box(self.base,'R'),'HAMER_TEXTURE')
        self.assertEqual(value['source'],'hamer_texture_hand3d')
        np.testing.assert_allclose(value['world'],self.points)

    def test_texture_compare_keeps_two_observations_without_applying(self):
        result=self.run_job('COMPARE_TEXTURE')
        self.assertEqual(set(result['hand_refinement']['candidates']),{'HAMER','HAMER_TEXTURE'})
        self.assertEqual(result['orientation'],self.base['orientation'])

    def test_missing_texture_weights_preserve_original_hand_source(self):
        result=self.run_job('HAMER_TEXTURE',ready=False)
        self.assertFalse(result['hand_refinement']['hands']['R']['adopted'])
        self.assertEqual(result['orientation'],self.base['orientation'])

    def video(self):
        import cv2
        path = self.root / 'source.avi'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 4, (64, 64))
        for _ in range(4):writer.write(np.zeros((64,64,3),np.uint8))
        writer.release()
        return path

    def test_video_reuses_one_session_and_records_hand_sources(self):
        import worker_video as video
        session = self.session()
        def detect(image, model, output, overlay, **kw):
            data = copy.deepcopy(self.base)
            data['pose_count'] = 1
            jobs.worker.write_json(output, data)
            return data
        with patch.object(h, 'readiness', return_value=dict(ready=True)), patch.object(h, 'Session', return_value=session) as factory:
            result = video.run(self.video(), self.root/'video_cache', self.model,
                backend='MEDIAPIPE', sample_fps=4, detect_fn=detect, hand3d_backend='WILOR')
        self.assertEqual(factory.call_count, 1)
        self.assertEqual(session.calls, [('WILOR',2)]*4)
        self.assertTrue(session.closed)
        self.assertEqual(result['hand3d_backend'], 'WILOR')
        for row in result['frames']:
            self.assertTrue(row['hand3d']['hands']['L']['adopted'])
            self.assertEqual(row['hand3d']['hands']['R']['source'], 'wilor_hand3d')

    def test_video_missing_assets_fails_before_creating_cache(self):
        import worker_video as video
        output = self.root / 'video_cache'
        with patch.object(h, 'readiness', return_value=dict(ready=False,missing=['MANO_RIGHT.pkl'])):
            with self.assertRaisesRegex(ValueError, 'MANO_RIGHT'):
                video.run(self.video(), output, self.model, hand3d_backend='HAMER')
        self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
