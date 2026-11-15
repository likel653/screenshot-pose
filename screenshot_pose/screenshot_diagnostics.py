"""Measure evaluated target controls against the prediction, not human truth."""
import math
import numpy as np
from mathutils import Matrix
from . import orientation_math as om


def rotations(rig, mapping):
    return {role: rig.pose.bones[name].matrix.to_quaternion().copy()
            for role, name in mapping.items() if name in rig.pose.bones}


def rotation_error(rig, mapping, reference):
    actual = rotations(rig, mapping)
    if not reference or not actual: return 0., {}
    # Remove a common torso rotation; preserve articulation and axial twist.
    root = 'hips' if 'hips' in reference and 'hips' in actual else next(iter(reference))
    align = actual[root] @ reference[root].inverted()
    errors = {}
    for role,q in reference.items():
        if role not in actual or role==root:continue
        delta=(align @ q).rotation_difference(actual[role])
        # atan2 avoids float32 2*pi rounding and treats q and -q identically.
        errors[role]=math.degrees(2*math.atan2(math.sqrt(delta.x**2+delta.y**2+delta.z**2),abs(delta.w)))
    return float(np.mean(list(errors.values()))) if errors else 0., errors


def hands(rig, mapping, settings, data, calibration):
    from . import finger_pose
    result = {}
    for side in ('L', 'R'):
        observation = data.get('orientation', {}).get('hands', {}).get(side, {})
        if 'world' not in observation or side not in calibration['hands']: continue
        hand = rig.pose.bones.get(mapping.get('hand.' + side, ''))
        if not hand: continue
        source = np.asarray(observation['world'], float)
        if source.shape != (21, 3): continue
        source_palm = om.proper_rotation(observation['frame'])
        target_palm = (hand.matrix.to_3x3().normalized()
                       @ hand.bone.matrix_local.to_3x3().normalized().transposed()
                       @ Matrix(calibration['hands'][side]['frame']))
        to_source = source_palm @ np.asarray(target_palm).T
        fingers = {}; directions = []
        for fi, finger in enumerate(finger_pose.FINGERS):
            chain = [finger_pose.control(rig, finger, i, side) for i in (1, 2, 3)]
            if any(b is None for b in chain):
                fingers[finger] = dict(status='missing_target_chain'); continue
            predicted = np.diff(source[1 + fi * 4:5 + fi * 4], axis=0)
            actual = np.array([to_source @ np.array(b.tail - b.head) for b in chain])
            normalize = lambda v: v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-10)
            errors = np.degrees(np.arccos(np.clip(np.sum(normalize(predicted) * normalize(actual), axis=1), -1, 1)))
            source_flex = np.degrees(np.arccos(np.clip(np.sum(normalize(predicted)[:-1] * normalize(predicted)[1:], axis=1), -1, 1)))
            actual_flex = np.degrees(np.arccos(np.clip(np.sum(normalize(actual)[:-1] * normalize(actual)[1:], axis=1), -1, 1)))
            fingers[finger] = dict(direction_errors_deg=errors.tolist(), source_flex_deg=source_flex.tolist(),
                                   target_flex_deg=actual_flex.tolist(),
                                   target_points=[list(to_source @ np.array(b.head - hand.head)) for b in chain] + [list(to_source @ np.array(chain[-1].tail - hand.head))])
            directions.extend(errors.tolist())
        manual = bool(data.get('manual_hand_edits', {}).get(side)) or getattr(settings, 'hand_left' if side == 'L' else 'hand_right', 'NONE') != 'NONE'
        result[side] = dict(source=observation.get('source'), fingers=fingers,
            mean_direction_error_deg=float(np.mean(directions)) if directions else None,
            max_direction_error_deg=max(directions) if directions else None,
            metric='source_to_evaluated_target_not_recognition_accuracy',
            intentional_manual_or_partial_strength=manual or settings.strength * settings.hand_strength < .999,
            source_points=source.tolist())
    return result


def payload(rig,mapping,settings,data,result):
    calibration=__import__(__package__+'.orientation',fromlist=['calibration']).calibration(rig,mapping,settings)
    source=data.get('full_skeleton',{});roles=source.get('role_map',{})
    parents={'spine':'hips','chest':'spine','neck':'chest','head':'neck'}
    for side in ('L','R'):
        for child,parent in [('upper_arm','chest'),('forearm','upper_arm'),('hand','forearm'),('thigh','hips'),('shin','thigh'),('foot','shin')]:
            parents[child+'.'+side]=parent if parent in ('chest','hips') else parent+'.'+side
    rotation=np.asarray(result.get('fit',{}).get('camera',{}).get('rotation',np.eye(3)))
    target={r:(rotation@np.asarray(rig.matrix_world@rig.pose.bones[n].head)).tolist() for r,n in mapping.items() if n in rig.pose.bones and r in roles}
    body=dict(source={r:source['bones'][roles[r]]['pose_head'] for r in target},target=target,parents=parents) if target else None
    return dict(schema='screenshot_pose/transfer_diagnosis_v1',image=data['image'],target=rig.name,body=body,
        hands=hands(rig,mapping,settings,data,calibration),comparison=result.get('full_source_comparison'),
        source_joint_count=len(source.get('bones',{})),recognition_accuracy_validated=False)


def export(context,settings,result):
    import importlib,json,shutil,uuid
    from pathlib import Path
    import bpy
    from . import diagnostic_report
    addon=importlib.import_module(__package__)
    data=addon.pose_tools.effective_document(settings,addon.pose_data.read_pose(bpy.path.abspath(settings.json_path)))
    report=payload(settings.target,addon.pose_tools.mapping_for(settings),settings,data,result)
    report['frame']=context.scene.frame_current
    folder=addon.runtime_dir()/'diagnostics'/uuid.uuid4().hex;folder.mkdir(parents=True)
    source=Path(data['image']['path']);image='source'+source.suffix.lower()
    shutil.copy2(source,folder/image)
    (folder/'diagnosis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    path=folder/'diagnosis.html';path.write_text(diagnostic_report.page(report,image),encoding='utf8')
    return path
