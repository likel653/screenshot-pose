"""Conservative screenshot candidate decisions; model fidelity is not accuracy."""
import math
import numpy as np


def body_choice(native, landmarks, width):
    """Use image evidence and source-pose fidelity, with a native tie preference.

    All errors must come from the same body-only pipeline and initial rig state.
    A better pixel fit cannot silently discard a large amount of joint rotation.
    """
    width = float(width)
    if not math.isfinite(width) or width <= 0:
        raise ValueError('图片宽度无效')
    def valid(item):
        return item.get('valid', True) and all(math.isfinite(float(item[k])) for k in ('projection_px', 'rotation_deg')) and math.isfinite(float(item.get('score_px',item['projection_px'])))
    n, l = valid(native), valid(landmarks)
    if not n and not l:
        raise ValueError('两个身体候选均无有效评分')
    if not n: return 'landmark_fit', 'native_candidate_invalid'
    if not l: return 'full_source_rotation', 'landmark_candidate_invalid'
    gain = native['projection_px'] - landmarks['projection_px']
    # A 0.2%-width / 5% improvement is required before leaving the richer pose.
    significant = gain > max(width * .002, native['projection_px'] * .05)
    rotation_loss = landmarks['rotation_deg'] - native['rotation_deg']
    # The fidelity term has explicit units (pixels per degree), rather than
    # presenting either pixel error or predicted 3D pose as ground truth.
    score_gain = native.get('score_px',native['projection_px'])-landmarks.get('score_px',landmarks['projection_px']) - width * .0005 * max(0., rotation_loss)
    if significant and score_gain > width * .002:
        return 'landmark_fit', 'image_improvement_outweighs_source_rotation_loss'
    return 'full_source_rotation', 'preserve_source_rotation_without_clear_joint_evidence'


def hand_reference_score(predicted_px, location):
    """Independent confident finger observations, not the model's own projection."""
    refs = location.get('refs')
    if refs is None: return None
    points = np.asarray(predicted_px, float)
    refs = np.asarray(refs, float)
    confidence = np.asarray(location.get('scores', []), float)
    if points.shape != (21, 2) or refs.shape != (21, 2) or confidence.shape != (21,):
        return None
    tips = [i for i in (4, 8, 12, 16, 20) if confidence[i] >= .75]
    span = max(float(location['span']), 1.)
    # A whole-body prediction can assign two crossed fingers the same tip.
    # Their 2D positions cannot rank alternative 3D hand crops reliably.
    ambiguous = {i for i in tips for j in tips if i != j and np.linalg.norm(refs[i] - refs[j]) < span * .025}
    ids = [i for i in np.flatnonzero(confidence >= .75) if i not in ambiguous]
    tips = [i for i in tips if i not in ambiguous]
    if len(ids) < 8 or len(tips) < 3: return None
    errors = np.linalg.norm(points[ids] - refs[ids], axis=1) / span
    return dict(error=float(np.average(errors, weights=confidence[ids])),
                observed_joints=len(ids), observed_tips=len(tips),
                metric='independent_high_confidence_2d_not_3d_accuracy')


def hand_choice(scores):
    """Only change crop when independent evidence improves clearly. No averaging."""
    available = [i for i, item in enumerate(scores) if item is not None]
    if not available: return 0
    best = min(available, key=lambda i: scores[i]['error'])
    if scores[0] is None: return best
    if scores[0]['error'] - scores[best]['error'] > max(.02, scores[0]['error'] * .15):
        return best
    return 0
