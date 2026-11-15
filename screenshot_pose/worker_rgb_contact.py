"""Run the installed RGB MoCap contact solver in its own Python process."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation


JOINTS = (0, 1, 2, 4, 5, 7, 8)
BASIS = np.array(((1., 0., 0.), (0., 0., 1.), (0., -1., 0.)))
LEGS = ((1, 4, 7), (2, 5, 8))


def load_solver(root):
    sys.path.insert(0, str(root))
    from mocap import correction

    # Installed RGB MoCap and its source checkout place the same module in
    # different directories. Use the binary for this Python ABI only.
    tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
    candidates = [file for path in ('third_party/KimodoMotionCorrection/bin', 'prebuilt/motion-correction')
                  for file in (root / path).glob(f'_motion_correction.{tag}-*.pyd')]
    if not candidates:
        raise ValueError('找不到与所选 Python 版本匹配的 RGB MoCap 接触修正模块')
    spec = importlib.util.spec_from_file_location('_motion_correction', candidates[0])
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    if getattr(native, 'rgb_mocap_anatomical_version', 0) != 1:
        raise ValueError('RGB MoCap 接触修正模块版本不兼容')
    correction.backend = lambda: native
    return correction


def solve(data, correction):
    rest = data['rest'] @ BASIS.T
    points = data['points'] @ BASIS.T
    fps = float(data['fps'])
    n = len(points)
    leg_length = float(np.mean([np.linalg.norm(rest[b] - rest[a]) + np.linalg.norm(rest[c] - rest[b])
                                for a, b, c in LEGS]))
    if not np.isfinite(leg_length) or leg_length < 1e-5:
        raise ValueError('无法估计目标骨架的腿长')
    # Keep the RGB MoCap metric thresholds comparable on unusually scaled rigs.
    units = 1. if .55 <= leg_length <= 1.2 else .9 / leg_length
    rest *= units
    points *= units
    # This adapter supplies the pelvis and lower body. The solver's unused
    # upper-body slots remain passive, with the same root motion.
    delta = np.tile(np.eye(3), (n, 22, 1, 1))
    delta[:, JOINTS] = BASIS @ data['rotation'] @ data['rest_rotation'].swapaxes(-1, -2) @ BASIS.T
    local = delta.copy()
    for joint, parent in enumerate(correction.PARENTS[1:], 1):
        local[:, joint] = delta[:, parent].swapaxes(-1, -2) @ delta[:, joint]
    quat = Rotation.from_matrix(local.reshape(-1, 3, 3)).as_quat().reshape(n, 22, 4)[..., [3, 0, 1, 2]]
    root = points[:, 0].copy()
    fk, world = correction.fk(rest, root, quat)
    if np.linalg.norm(fk[:, JOINTS] - points[:, JOINTS], axis=-1).max() > .006:
        raise ValueError('腿部控制链有明显位移或缩放，当前动作无法安全进行支撑修正')

    ankles = [7, 7, 8, 8]
    sole_rest = rest[[7, 10, 8, 11]].copy()
    sole_rest[:, 1] = np.minimum(sole_rest[:, 1], np.repeat(rest[[10, 11], 1], 2)) - .018
    offsets = sole_rest - rest[ankles]

    def soles(p, r):
        return p[:, ankles] + np.einsum('tfij,fj->tfi', r[:, ankles], offsets)

    probes = soles(fk, world)
    floor = float(np.quantile(probes[..., 1].min(axis=1), .12))
    velocity = np.gradient(probes, 1. / fps, axis=0)
    speed = np.linalg.norm(velocity[..., [0, 2]], axis=-1)
    candidate = (abs(probes[..., 1] - floor) < .075) & (speed < .47) & (abs(velocity[..., 1]) < .50)
    if not candidate.any():
        return None, dict(contacts=0)
    fixed_root, fixed_quat, diagnostics = correction.correct_arrays(
        rest, root, quat, probes, np.where(candidate, .92, 0.), fps=fps, floor=floor,
    )
    if not diagnostics['contacts']:
        return None, dict(contacts=0)
    blend = np.ones(n)
    fade = min(n // 2, max(2, round(.25 * fps)))
    blend[:fade] = np.linspace(0, 1, fade)
    blend[-fade:] = blend[:fade][::-1]
    fixed_root = root + blend[:, None] * (fixed_root - root)
    solved = Rotation.from_quat(fixed_quat[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(n, 22, 3, 3)
    for joint in JOINTS[1:]:
        relative = Rotation.from_matrix(local[:, joint].swapaxes(-1, -2) @ solved[:, joint]).as_rotvec()
        local[:, joint] = local[:, joint] @ Rotation.from_rotvec(relative * blend[:, None]).as_matrix()
    final_quat = Rotation.from_matrix(local.reshape(-1, 3, 3)).as_quat().reshape(n, 22, 4)[..., [3, 0, 1, 2]]
    final, final_world = correction.fk(rest, fixed_root, final_quat)
    active = np.zeros((n, 2), dtype=bool)
    for segment in diagnostics['segments']:
        active[segment['start']:segment['end'], segment['foot']] = True
    pairs = active[:-1] & active[1:]
    speeds = []
    for track in (probes, soles(final, final_world)):
        centers = track.reshape(n, 2, 2, 3).mean(2)[..., [0, 2]]
        speeds.append(float(np.linalg.norm(np.diff(centers, axis=0), axis=-1)[pairs].mean() * fps / units) if pairs.any() else 0.)
    if not pairs.any() or speeds[0] < 1e-5 or speeds[1] >= .98 * speeds[0]:
        return None, dict(contacts=int(diagnostics['contacts']), no_improvement=True, speed=speeds)
    desired = BASIS.T @ final_world[:, JOINTS] @ BASIS @ data['rest_rotation']
    results = dict(points=(final[:, JOINTS] @ BASIS) / units, rotations=desired)
    report = dict(contacts=int(diagnostics['contacts']), segments=diagnostics['segments'],
                  speed=speeds, comparable_steps=int(pairs.sum()), frames=n, fps=fps)
    return results, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    correction = load_solver(args.root)
    with np.load(args.folder / 'input.npz', allow_pickle=False) as data:
        results, report = solve(data, correction)
    if results is not None:
        np.savez_compressed(args.folder / 'result.npz', **results)
    (args.folder / 'report.json').write_text(json.dumps(report, default=lambda value: value.item()), encoding='utf8')


if __name__ == '__main__':
    main()
