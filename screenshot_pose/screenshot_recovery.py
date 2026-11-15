"""Retry unusually tilted stills, and verify the recovered leg chains."""
import math

import numpy as np

try:
    from . import pose_data, worker
except ImportError:
    import pose_data
    import worker


def estimate_roll(image):
    import cv2

    sample = image.convert('L')
    sample.thumbnail((800, 800))
    size = max(sample.size)
    lines = cv2.HoughLinesP(cv2.Canny(np.asarray(sample), 70, 170), 1, np.pi/180,
                            45, minLineLength=.095*size, maxLineGap=.017*size)
    if lines is None:
        return None
    delta = lines[:, 0, 2:] - lines[:, 0, :2]
    angles = (np.rad2deg(np.arctan2(delta[:, 1], delta[:, 0])) + 45) % 90 - 45
    lengths = np.linalg.norm(delta, axis=1)
    bins = np.arange(-44, 45, 2)
    clusters = np.abs(angles[:, None] - bins) <= 4
    weights = lengths @ clusters
    best = int(np.argmax(weights))
    cluster = clusters[:, best]
    if cluster.sum() < 4 or weights[best] < .52*lengths.sum():
        return None
    angle = float(np.average(angles[cluster], weights=lengths[cluster]))
    return round(angle, 1) if 12 <= abs(angle) <= 41 else None


def leg_evidence(data, projected):
    """Check separation and side assignment against SAM's own image projection."""
    w, h = data['image']['width'], data['image']['height']
    indices = (25, 26, 27, 28)
    rows = [data['landmarks'][i] for i in indices]
    if any(p.get('outside_frame') or min(p['visibility'], p['presence']) < .5 for p in rows):
        return {}
    observed = np.array([[p['x']*w, p['y']*h] for p in rows])
    native = projected[[11, 12, 13, 14]]
    if (observed < 0).any() or (native < 0).any() or (observed > [w, h]).any() or (native > [w, h]).any():
        return {}
    a, b = observed[2] - observed[3], native[2] - native[3]
    sep, native_sep = np.linalg.norm(a), np.linalg.norm(b)
    if min(sep, native_sep) < w*.055:
        return {}
    alignment = float(a @ b / (sep*native_sep))
    direct = np.linalg.norm(observed[2:] - native[2:], axis=1)
    reversed_legs = alignment < -.65
    if reversed_legs:
        # Only reassign lower limbs when both hips still belong to the same
        # anatomical sides and the opposite ankles match much more closely.
        hips = np.array([[p['x']*w, p['y']*h] for p in data['landmarks'][23:25]])
        native_hips = projected[[9, 10]]
        hip_direct = np.linalg.norm(hips-native_hips, axis=1).sum()
        hip_swapped = np.linalg.norm(hips[::-1]-native_hips, axis=1).sum()
        matched = np.linalg.norm(observed[2:][::-1]-native[2:], axis=1)
        if hip_direct > w*.15 or hip_swapped-hip_direct < w*.03 or direct.sum()-matched.sum() < sep*.6:
            return {}
    else:
        matched = direct
    ankle_gap = float(matched.max())
    if abs(alignment) < .65 or ankle_gap > min(w*.12, sep*.4):
        return {}
    return dict(ankle_separation_px=float(sep), native_separation_px=float(native_sep),
                ankle_disagreement_px=ankle_gap, agreement=alignment, reassigned=reversed_legs)


def reassign_legs(data):
    landmarks, wholebody = data['landmarks'], data['wholebody']
    for left in (25, 27, 29, 31):
        landmarks[left], landmarks[left+1] = landmarks[left+1], landmarks[left]
    for left, right in ((13,14),(15,16),(17,20),(18,21),(19,22)):
        wholebody[left], wholebody[right] = wholebody[right], wholebody[left]
    data['body_candidates'] = [c for c in data.get('body_candidates', ())
                               if c['source'] == 'sam3d_mhr_depth']
    data.get('orientation', {}).get('feet', {}).clear()
    data.pop('foot_review_sides', None)


def update_wrists(data, reports):
    """Move only an independently recovered wrist and its hand anchors."""
    try:
        from .hand3d import REGIONAL_SOURCE
    except ImportError:
        from hand3d import REGIONAL_SOURCE

    recovered = []
    for side, report in reports.items():
        if not report.get('adopted') or (report.get('localization') or {}).get('source') != REGIONAL_SOURCE:
            continue
        hand = data['orientation']['hands'][side]
        points = hand['landmarks']
        if any(not (0 <= value <= 1) for i in (0, 4, 8, 20) for value in points[i]):
            continue
        wrist = 15 if side == 'L' else 16
        for index, joint in zip((wrist,wrist+2,wrist+4,wrist+6),(0,20,8,4)):
            x,y = points[joint]
            data['landmarks'][index].update(x=x,y=y,visibility=.7,presence=.7)
        recovered.append(side)
        report['wrist_updated'] = True
    if recovered:
        data['screenshot_recovered_wrists'] = recovered
        data.setdefault('backend_warnings', []).append(
            '已用独立检测和多尺度手部结果修正手腕位置；手腕前后深度沿用身体估计')
    return recovered


def recover(image, model, folder, detect, no_person, args, baseline, stage):
    """Use bounded retries only for a single target with strong camera-roll evidence."""
    if args['person_index'] or (baseline and baseline.get('pose_count', 1) != 1):
        return None
    im = worker.prepare_image(image)
    roll = estimate_roll(im)
    if roll is None:
        return None
    from PIL import Image

    stage('参考图倾斜，复核人物和双腿')
    attempts = []
    folder.mkdir()
    for index, angle in enumerate((roll, roll - math.copysign(8, roll))):
        out = folder/f'roll_review_{index}'
        out.mkdir()
        leveled = im.rotate(angle, Image.Resampling.BICUBIC, expand=True, fillcolor=(48, 48, 48))
        snapshot = out/'reference.png'
        worker.save_png(leveled, snapshot)
        try:
            data = detect(snapshot, model, out/'pose.json', out/'overlay.png',
                          **dict(args, confidence=min(args['confidence'], .3), reference_device='cpu'))
        except no_person:
            attempts.append(dict(angle=angle, accepted=False, reason='no_person'))
            continue
        evidence = data.get('screenshot_leg_evidence')
        single_target = data.get('screenshot_person_area_ratio', float('inf')) >= 1.65
        accepted = bool(evidence and single_target and not pose_data.feet_overlap(data))
        attempts.append(dict(angle=angle, accepted=accepted, evidence=evidence))
        if accepted:
            if evidence['reassigned']:
                reassign_legs(data)
            data['screenshot_roll_review'] = dict(angle_ccw=angle, estimated_roll=roll, trials=attempts,
                                                 reason='no_person' if baseline is None else 'overlapping_feet')
            data.setdefault('backend_warnings', []).append(
                f'参考图旋转 {angle:g}° 后重识别，已复核双脚归属；膝盖深度和鞋底方向仍可检查')
            return data
    return None
