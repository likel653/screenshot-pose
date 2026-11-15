"""Shared hand reconstruction contract. No torch/Blender imports in this module."""
import copy
import json
from pathlib import Path
import queue
import subprocess
import threading
import numpy as np
try:
    from . import orientation_math as om, hand_geometry as hg
except ImportError:
    import orientation_math as om, hand_geometry as hg

BACKENDS = ('WILOR', 'HAMER')
OPTIONAL_BACKENDS = ('HAMER_TEXTURE',)
KNOWN_BACKENDS = BACKENDS + OPTIONAL_BACKENDS
EDGES = [(0, 1), (0, 5), (0, 9), (0, 13), (0, 17)] + [
    (b + i, b + i + 1) for b in (1, 5, 9, 13, 17) for i in range(3)]


def configuration(runtime):
    path = Path(runtime).resolve() / 'hand3d' / 'config.json'
    try:
        value = json.loads(path.read_text(encoding='utf8'))
    except (OSError, ValueError) as exc:
        raise ValueError('尚未准备 WiLoR / HaMeR 环境，请先准备三维手部模型') from exc
    if value.get('schema') != 'screenshot_pose/hand3d_config_v1':
        raise ValueError('三维手部环境配置格式错误')
    return path, value


def readiness(runtime, backend):
    if backend not in KNOWN_BACKENDS:
        raise ValueError('未知三维手部模型')
    try:
        path, cfg = configuration(runtime)
    except ValueError as exc:
        return dict(ready=False, missing=[str(exc)], backend=backend)
    model = cfg.get('models', {}).get(backend, {})
    missing = []
    if not cfg.get('environment', {}).get('imports_ok') or not cfg.get('environment', {}).get('cuda'):
        missing.append('已验证的 CUDA 手部环境')
    for label, name in [('Python', cfg.get('python')), ('MANO_RIGHT.pkl', cfg.get('mano')),
                        ('模型配置', model.get('config')), ('手形均值', model.get('mean_params')),
                        ('推理源码', str(Path(model.get('repo', '')) / model.get('source_module',backend.lower()) / 'models' / '__init__.py')),
                        ('网络权重', model.get('checkpoint'))]:
        if not name or not Path(name).is_file():
            missing.append(label)
    return dict(ready=not missing, missing=missing, backend=backend, config=str(path))


def hand_box(data, side):
    """Use the selected person's anatomical hand, never screen left/right."""
    if side not in ('L', 'R'):
        raise ValueError('无效左右手')
    w, h = data['image']['width'], data['image']['height']
    start, wrist, other = (91, 15, 16) if side == 'L' else (112, 16, 15)
    b = data.get('wholebody', [])
    refs = None
    # The image editor moves the current anchors, not the old 133-point fit.
    # After an edit, crop from the updated wrist and hand anchors.
    ids = (wrist, wrist + 2, wrist + 4, wrist + 6)
    edited = bool(set(ids).intersection(data.get('edited_indices', ())))
    if len(b) == 133 and not edited:
        refs = np.array([[v['x'] * w, v['y'] * h] for v in b[start:start + 21]], float)
        scores = np.array([v.get('confidence', 0) for v in b[start:start + 21]])
        selected = refs[(scores >= .35) & np.isfinite(refs).all(axis=1)]
    else:
        scores = np.zeros(21)
        selected = np.empty((0, 2))
    lm = data['landmarks']
    wp = np.array([lm[wrist]['x'] * w, lm[wrist]['y'] * h])
    op = np.array([lm[other]['x'] * w, lm[other]['y'] * h])
    elbow = wrist - 2
    ep = np.array([lm[elbow]['x'] * w, lm[elbow]['y'] * h])
    if not np.isfinite(wp).all():
        raise ValueError('手腕定位无效')
    if len(selected) < 3:
        selected = np.array([[lm[i]['x'] * w, lm[i]['y'] * h] for i in ids
                             if min(lm[i].get('visibility', 1), lm[i].get('presence', 1)) >= .35])
        refs = None
    if len(selected) < 3 or not np.isfinite(selected).all():
        raise ValueError('手部可靠定位点不足，未强行猜测手指')
    lo = np.minimum(selected.min(axis=0), wp)
    hi = np.maximum(selected.max(axis=0), wp)
    span = float(max(hi - lo))
    if span < 12:
        raise ValueError('手部像素不足（小于 12 像素），请提供更清晰参考图')
    if span > max(w, h) * .75:
        raise ValueError('手部定位范围异常，可能串到了其他关节')
    center = (lo + hi) / 2
    location = dict(box=[*lo.tolist(), *hi.tolist()], center=center.tolist(), span=span,
                wrist=wp.tolist(), other_wrist=op.tolist(),
                forearm_length=float(np.linalg.norm(wp-ep)),
                refs=refs.tolist() if refs is not None else None, scores=scores.tolist(),
                anchor_source='edited_hand_anchors' if edited else 'wholebody_hand_points',
                width=w, height=h, right=int(side == 'R'))
    if not edited and side in data.get('screenshot_wrist_disagreements', ()) and (
        data.get('pose_count', 1) == 1 or data.get('screenshot_roll_review')
    ):
        def pixels(indices):
            return [[lm[i]['x']*w,lm[i]['y']*h] for i in indices
                    if not lm[i].get('outside_frame') and min(lm[i]['visibility'],lm[i]['presence']) >= .5]
        head, arms = pixels((0,2,5,7,8)), pixels((elbow, other-2))
        if len(head) >= 3 and len(arms) == 2:
            location['regional_search'] = dict(head=head, elbows=arms, feet=pixels(range(27,33)))
    return location


def associate_detection(location, detections):
    """Associate an anatomical detection with the chosen person's wrist region."""
    wrist=np.asarray(location['wrist'],float)
    radius=max(24.,float(location.get('forearm_length',0))*.6)
    choices=[]
    for detection in detections:
        if detection['right']!=location['right'] or detection['score']<.4:continue
        box=np.asarray(detection['box'],float)
        if box.shape!=(4,) or not np.isfinite(box).all() or min(box[2:]-box[:2])<8:continue
        gap=float(np.linalg.norm(wrist-np.clip(wrist,box[:2],box[2:])))
        if gap>radius:continue
        span=float(max(box[2:]-box[:2]));center=(box[:2]+box[2:])/2
        distance=float(np.linalg.norm(wrist-center))
        if distance>max(span*1.3,radius):continue
        cost=gap/radius+.2*distance/max(span,1)-.1*detection['score']
        choices.append((cost,dict(source='wilor_hand_detector',box=box.tolist(),
            confidence=float(detection['score']),wrist_region_gap_px=gap)))
    return min(choices,key=lambda v:v[0])[1] if choices else None


REGIONAL_SOURCE = 'wilor_face_region'


def regional_detection(location, detections):
    """Use arm identity for a face-level hand, and exclude reported feet."""
    region = location['regional_search']
    elbow, other = np.asarray(region['elbows'])
    head = np.asarray(region['head']).mean(axis=0)
    feet = np.asarray(region['feet'])
    choices = []
    for candidate in detections:
        box = np.asarray(candidate['box'])
        span = max(box[2:]-box[:2])
        center = (box[:2]+box[2:])/2
        own, opposite = np.linalg.norm(center-elbow), np.linalg.norm(center-other)
        if candidate['score'] < .42 or min(box[2:]-box[:2]) < 12 or span > .18*location['width']:
            continue
        if own > 2.8*span or opposite-own < .8*span or np.linalg.norm(center-head) > 1.35*span:
            continue
        if len(feet) and np.min(np.linalg.norm(feet-center,axis=1)) < .8*span:
            continue
        choices.append(dict(source=REGIONAL_SOURCE, box=box.tolist(), confidence=candidate['score'],
                            detector_side_agrees=candidate['right']==location['right'],
                            wrist_region_gap_px=float(np.linalg.norm(np.asarray(location['wrist'])-center))))
    return choices[0] if len(choices) == 1 else None


def observation(raw, side, location, backend):
    """Original models output right-hand CV coordinates; undo left crop once."""
    if backend not in KNOWN_BACKENDS or side not in ('L', 'R'):
        raise ValueError('手部模型或左右手无效')
    if raw.get('device') != 'cuda':
        raise ValueError('三维手部模型没有确认 CUDA 推理')
    location=copy.deepcopy(location)
    localization=raw.get('localization')
    regional = localization and localization.get('source') == REGIONAL_SOURCE
    if localization:
        box=np.asarray(localization['box'],float)
        if box.shape!=(4,) or not np.isfinite(box).all() or min(box[2:]-box[:2])<8:
            raise ValueError('专用手部检测框无效')
        location.update(box=box.tolist(),span=float(max(box[2:]-box[:2])),refs=None)
    p = np.asarray(raw.get('joints_cv'), float).copy()
    xy = np.asarray(raw.get('landmarks_px'), float)
    if p.shape != (21, 3) or xy.shape != (21, 2) or not np.isfinite(p).all() or not np.isfinite(xy).all():
        raise ValueError('三维手部模型返回的 21 点无效')
    p[:, 0] *= 1 if side == 'R' else -1
    p = (p - p[0]) @ om.C_DOWN.T
    lengths = np.array([np.linalg.norm(p[b] - p[a]) for a, b in EDGES])
    palm = np.linalg.norm(p[9])
    if not .015 < palm < .25 or lengths.min() < palm * .015 or lengths.max() > palm * 2.5:
        raise ValueError('手部三维骨长异常')
    w, h = location['width'], location['height']
    wp, other = np.array(location['wrist']), np.array(location['other_wrist'])
    span = location['span']
    if location.get('regional_search') and not regional:
        feet = np.asarray(location['regional_search']['feet'])
        if len(feet) and np.min(np.linalg.norm(feet-xy[[0,9]].mean(axis=0),axis=1)) < .6*span:
            raise ValueError('预测手掌与已识别脚部冲突')
    wrist_error = float(np.linalg.norm(xy[0] - wp) / span)
    if regional:
        region = location.get('regional_search')
        if not region:
            raise ValueError('区域手部候选没有对应的手臂参考')
        palm_anchors = xy[[0,5,9,17]]
        wrist_error = float(np.max(np.linalg.norm(palm_anchors-np.clip(palm_anchors,box[:2],box[2:]),axis=1))/span)
        if wrist_error > .38:
            raise ValueError('区域手部预测偏离独立检测框')
        elbow, other_elbow = np.asarray(region['elbows'])
        if np.linalg.norm(xy[0]-elbow) > 2.8*span or np.linalg.norm(xy[0]-other_elbow)-np.linalg.norm(xy[0]-elbow) < .6*span:
            raise ValueError('区域手腕与手臂归属不一致')
    elif wrist_error > .45:
        raise ValueError('三维手部结果与所选手腕不匹配')
    if not regional and np.linalg.norm(other - wp) > span * .25 and np.linalg.norm(xy[0] - other) < np.linalg.norm(xy[0] - wp):
        raise ValueError('三维手部结果疑似左右手串位')
    if np.any(xy < [-.1 * w, -.1 * h]) or np.any(xy > [1.1 * w, 1.1 * h]):
        raise ValueError('预测手指超出图片范围')
    error = wrist_error
    if location['refs'] is not None:
        refs = np.asarray(location['refs'])
        # Palm anchors gate identity/alignment. Uncertain old fingertips must
        # not veto a better reconstruction of the fingers themselves.
        anchors = [i for i in (0, 5, 9, 17) if location['scores'][i] >= .35]
        if anchors:
            error = float(np.mean(np.linalg.norm(xy[anchors] - refs[anchors], axis=1)) / span)
            if error > .42:
                raise ValueError('三维手掌与原图定位不一致，请先校正手腕和手掌位置')
    result=dict(world=p.tolist(), frame=om.palm_frame(p).tolist(),
                landmarks=(xy / [w, h]).tolist(), curls=om.curls(p),
                source=backend.lower() + '_hand3d', anatomical_side=side,
                confidence=float(np.clip(.9 - error, .45, .9)),
                confidence_kind='geometric_gate_not_model_probability',
                reprojection_error=error, gesture_suggestion=hg.suggest(p),
                device='cuda', depth_available=True, box=location['box'],
                camera_axes='X_right_Y_away_Z_up', metric_scale='MANO_estimate_not_measurement')
    result['localization']=localization or dict(source=location.get('anchor_source','wholebody_hand_points'))
    if regional:
        result.update(confidence=min(.7,result['confidence']),error_metric='palm_outside_detector_box_fraction')
    if raw.get('input_sampling'):result['input_sampling']=raw['input_sampling']
    if raw.get('rest_joints_cv') is not None:
        try:from .hand_precision import mano_frames
        except ImportError:from hand_precision import mano_frames
        result.update(mano_frames(raw['rest_joints_cv'],raw['joint_rotations_cv'],side))
    return result


class Session:
    """One isolated CUDA process per job; reuses a model across video frames."""
    def __init__(self, runtime, timeout=240.,screenshot_precision=False):
        self.path, cfg = configuration(runtime)
        self.runtime = runtime
        self.timeout = timeout
        self.screenshot_precision=screenshot_precision
        self.events = queue.Queue()
        self.closed = False
        log = self.path.parent / 'runner.log'
        self.log = log.open('ab')
        try:
            self.process = subprocess.Popen([cfg['python'], str(Path(__file__).with_name('hand3d_runner.py')),
                '--config', str(self.path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            self.reader = threading.Thread(target=self._read, daemon=True)
            self.reader.start()
        except Exception:
            self.log.close()
            raise

    def _read(self):
        try:
            while True:
                line = self.process.stdout.readline(2_000_001)
                if not line or len(line) > 2_000_000:
                    break
                self.events.put(line)
        finally:
            self.events.put(None)

    def infer(self, backend, image, locations):
        state = readiness(self.runtime, backend)
        if not state['ready']:
            raise ValueError(backend + ' 缺少：' + '、'.join(state['missing']))
        if self.closed:
            raise RuntimeError('三维手部会话已关闭')
        request = dict(backend=backend, image=str(image), boxes=[v['box'] for v in locations],
                       right=[v['right'] for v in locations],locations=locations,
                       screenshot_precision=self.screenshot_precision,
                       original_image=getattr(self,'original_image',None))
        try:
            self.process.stdin.write((json.dumps(request, ensure_ascii=True) + '\n').encode('utf8'))
            self.process.stdin.flush()
            answer = self.events.get(timeout=self.timeout)
            if answer is None:
                raise RuntimeError('三维手部运行器退出，请查看 ' + str(self.path.parent / 'runner.log'))
            result = json.loads(answer)
            if not result.get('ok'):
                raise ValueError(result.get('error', '三维手部推理失败'))
            if len(result['hands']) != len(locations):
                raise ValueError('三维手部返回数量不匹配')
            return result['hands']
        except (queue.Empty, BrokenPipeError, OSError, json.JSONDecodeError) as exc:
            self.close()
            raise RuntimeError('三维手部推理超时或通信失败，请查看 runner.log') from exc

    def close(self):
        if self.closed:
            return
        self.closed = True
        if hasattr(self, 'process'):
            p = self.process
            try:
                p.stdin.close()
            except OSError:
                pass
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=5)
            self.reader.join(timeout=2)
            p.stdout.close()
        self.log.close()


def refine(data, backend, session, sides=('L', 'R')):
    """Explicit per-hand rejection preserves that hand; never rename a fallback."""
    result = copy.deepcopy(data)
    hands = result.setdefault('orientation', {}).setdefault('hands', {})
    reports, locations, valid = {}, [], []
    for side in sides:
        try:
            locations.append(hand_box(data, side))
            valid.append(side)
        except ValueError as exc:
            reports[side] = dict(adopted=False, reason=str(exc), backend=backend)
    if valid:
        try:
            raw = session.infer(backend, data['image']['path'], locations)
        except Exception as exc:
            for side in valid:
                reports[side] = dict(adopted=False, reason=str(exc), backend=backend)
        else:
            for side, loc, prediction in zip(valid, locations, raw):
                try:
                    try:from . import screenshot_quality as quality
                    except ImportError:import screenshot_quality as quality
                    variants=[prediction]+prediction.get('crop_candidates',[])
                    candidates=[];crop_reports=[]
                    for variant in variants:
                        try:
                            hand=observation(variant,side,loc,backend)
                            score=None if hand['localization']['source']==REGIONAL_SOURCE else quality.hand_reference_score(variant['landmarks_px'],loc)
                            candidates.append(hand)
                            crop_reports.append(dict(expansion=variant.get('crop_expansion',2.),valid=True,reference=score))
                        except ValueError as exc:
                            candidates.append(None)
                            crop_reports.append(dict(expansion=variant.get('crop_expansion',2.),valid=False,reason=str(exc)))
                    valid_indices=[i for i,c in enumerate(candidates) if c is not None]
                    if not valid_indices:raise ValueError(crop_reports[0].get('reason','手部候选均无效'))
                    if len(variants)>1 and candidates[0] is None and len(valid_indices)<2:
                        raise ValueError('多裁剪结果缺乏一致的手部定位，保留原观测')
                    scores=[crop_reports[i].get('reference') for i in valid_indices]
                    index=valid_indices[quality.hand_choice(scores)]
                    regional=candidates[index]['localization']['source']==REGIONAL_SOURCE
                    if regional:
                        anchors = np.array([variants[i]['landmarks_px'] for i in valid_indices])[:,[0,5,9,17]]
                        distances = np.linalg.norm(anchors[:,None]-anchors[None,:],axis=-1).mean(axis=-1)
                        span = max(np.subtract(candidates[index]['box'][2:],candidates[index]['box'][:2]))
                        support = (distances <= .24*span).sum(axis=1)
                        best = int(np.argmax(support))
                        if support[best] < 2:
                            raise ValueError('区域手部多尺度预测不一致')
                        index=valid_indices[best]
                    hd=candidates[index]
                    if len(variants)>1:
                        hd['crop_selection']=dict(chosen=index,candidates=crop_reports,
                            reason='regional_multi_crop_agreement' if regional else 'independent_2d_evidence' if index!=valid_indices[0] else 'keep_default_without_clear_evidence',
                            model_accuracy_validated=False)
                    hands[side] = hd
                    reports[side] = dict(adopted=True, source=hd['source'], backend=backend,
                                         confidence=hd['confidence'], has_21_points=True,
                                         localization=hd['localization'],
                                         input_sampling=hd.get('input_sampling'),native_rotations=hd.get('native_source')=='MANO',
                                         crop_selection=hd.get('crop_selection'),
                                         error_metric=hd.get('error_metric'),
                                         reprojection_error=hd['reprojection_error'])
                except ValueError as exc:
                    reports[side] = dict(adopted=False, reason=str(exc), backend=backend)
    result['hand3d_refinement'] = dict(backend=backend, hands=reports, body_preserved=True)
    return result, reports
