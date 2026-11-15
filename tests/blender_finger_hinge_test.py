"""Run with Blender in background mode, without an asset fixture."""
import math
import sys
from pathlib import Path

from mathutils import Quaternion, Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from screenshot_pose.finger_pose import hinge_angle, limit_targets


def assert_rotation(actual, expected):
    delta = actual.rotation_difference(expected)
    assert min(delta.angle, abs(2 * math.pi - delta.angle)) < 1e-4


base = Quaternion((0, 0, 1), .45)
rest = Quaternion((0, 1, 0), .25)
axis = Vector((1, 0, .5)).normalized()
along = Vector((0, 1, 0))
hinges = {
    'mid': ('base', rest, axis, (-.15, 1.9)),
    'tip': ('mid', rest, axis, (-.25, 1.5)),
}
identity = Quaternion()
initial = {'base': identity, 'mid': rest, 'tip': rest @ rest}
# Twist the shaft without changing flexion. The next joint is bent too far.
raw_mid = base @ rest @ Quaternion(axis, .8) @ Quaternion(along, .6)
raw_tip = raw_mid @ rest @ Quaternion(axis, 2.)
for strength in (1., .4):
    targets = {'base': base, 'mid': raw_mid, 'tip': raw_tip}
    limit_targets(targets, initial, hinges, strength)
    assert_rotation(targets['base'], identity.slerp(base, strength))
    assert_rotation(targets['mid'], targets['base'] @ rest @ Quaternion(axis, .8 * strength))
    assert_rotation(targets['tip'], targets['mid'] @ rest @ Quaternion(axis, 1.5 * strength))

# A pure sideways deflection must contribute no curl.
sideways = Quaternion(axis.cross(along), .6)
assert abs(hinge_angle(sideways, axis)) < 1e-6

print('FINGER_HINGE_TEST_PASS', flush=True)
