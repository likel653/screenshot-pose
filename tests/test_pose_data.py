"""Coordinate contract tests without importing Blender or the add-on package."""
import copy
import importlib.util
import math
from pathlib import Path
import unittest


DATA_PATH = Path(__file__).resolve().parents[1] / "screenshot_pose" / "pose_data.py"
spec = importlib.util.spec_from_file_location("pose_data_for_tests", DATA_PATH)
pose_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pose_data)


class PoseCoordinateTests(unittest.TestCase):
    def test_crop_requires_low_confidence_feet_and_both_knees_at_bottom(self):
        data = pose_data.template_pose()
        data['source'] = 'sam3d_mhr'
        for index in (25, 26, 27, 28, 31, 32):
            data['landmarks'][index]['y'] = .97
        for index in (27, 28, 31, 32):
            data['world_landmarks'][index]['visibility'] = .2
        self.assertTrue(pose_data.lower_body_cropped(data))

        edited = dict(data, edited_indices=[27])
        self.assertFalse(pose_data.lower_body_cropped(edited))
        full_body = copy.deepcopy(data)
        full_body['landmarks'][25]['y'] = .72
        self.assertFalse(pose_data.lower_body_cropped(full_body))
        data['world_landmarks'][27]['visibility'] = .8
        self.assertFalse(pose_data.lower_body_cropped(data))

    def assertPointAlmostEqual(self, first, second):
        for axis, (a, b) in enumerate(zip(first, second)):
            self.assertAlmostEqual(a, b, places=10, msg=f"axis {axis}")

    def test_image_coordinates_keep_equal_pixel_distances_equal(self):
        # A 100-pixel horizontal or vertical segment must remain equal in 3D,
        # even for a tall screenshot whose normalized x/y units differ.
        data = pose_data.template_pose(width=400, height=800)
        data["landmarks"][11].update(x=.5, y=.25, z=0)
        data["landmarks"][13].update(x=.75, y=.25, z=0)
        data["landmarks"][15].update(x=.75, y=.375, z=0)
        joints, _ = pose_data.to_joints(data, mode="IMAGE")
        upper = math.dist(joints["shoulder.L"], joints["elbow.L"])
        lower = math.dist(joints["elbow.L"], joints["wrist.L"])
        self.assertAlmostEqual(upper, .25)
        self.assertAlmostEqual(upper, lower)

    def test_world_coordinates_do_not_depend_on_image_aspect(self):
        data = pose_data.template_pose(width=400, height=800)
        changed = copy.deepcopy(data)
        changed["image"].update(width=1600, height=900)
        original, _ = pose_data.to_joints(data, mode="WORLD")
        resized, _ = pose_data.to_joints(changed, mode="WORLD")
        self.assertEqual(original, resized)

    def test_camera_axes_and_pelvis_center(self):
        data = pose_data.template_pose()
        points = data["world_landmarks"]
        points[23].update(x=1, y=2, z=3)
        points[24].update(x=3, y=2, z=3)
        points[11].update(x=5, y=-1, z=-2)
        joints, _ = pose_data.to_joints(data, depth=.5)
        self.assertPointAlmostEqual(joints["pelvis"], (0, 0, 0))
        self.assertPointAlmostEqual(joints["hip.L"], (-1, 0, 0))
        self.assertPointAlmostEqual(joints["hip.R"], (1, 0, 0))
        # Image right is +X, closer to camera is -Y, image up is +Z.
        self.assertPointAlmostEqual(joints["shoulder.L"], (3, -2.5, 3))

    def test_zero_depth_flattens_only_camera_depth(self):
        data = pose_data.template_pose()
        data["world_landmarks"][13]["z"] = -.8
        full, _ = pose_data.to_joints(data)
        flat, _ = pose_data.to_joints(data, depth=0)
        for name in full:
            self.assertEqual(flat[name][1], 0)
            self.assertEqual(flat[name][0], full[name][0])
            self.assertEqual(flat[name][2], full[name][2])

    def test_mirror_reflects_and_swaps_anatomical_sides_and_confidence(self):
        data = pose_data.template_pose()
        data["world_landmarks"][13].update(x=.7, y=-.25, z=-.4,
                                           visibility=.1, presence=.8)
        original, low = pose_data.to_joints(data)
        mirrored, mirrored_low = pose_data.to_joints(data, mirror=True)
        self.assertIn("elbow.L", low)
        self.assertIn("elbow.R", mirrored_low)
        self.assertNotIn("elbow.L", mirrored_low)
        p = original["elbow.L"]
        self.assertPointAlmostEqual(mirrored["elbow.R"], (-p[0], p[1], p[2]))
        self.assertPointAlmostEqual(mirrored["pelvis"], original["pelvis"])
        self.assertAlmostEqual(math.dist(original["shoulder.L"], original["elbow.L"]),
                               math.dist(mirrored["shoulder.R"], mirrored["elbow.R"]))

    def test_yaw_is_radians_and_rotates_about_vertical_axis(self):
        data = pose_data.template_pose()
        original, _ = pose_data.to_joints(data)
        turned, _ = pose_data.to_joints(data, yaw=math.pi / 2)
        for name, point in original.items():
            self.assertPointAlmostEqual(turned[name], (-point[1], point[0], point[2]))

    def test_nonfinite_coordinates_and_bool_are_rejected(self):
        for source in ("landmarks", "world_landmarks"):
            for value in (float("nan"), float("inf"), -float("inf"), True):
                with self.subTest(source=source, value=value):
                    data = pose_data.template_pose()
                    data[source][13]["x"] = value
                    with self.assertRaises(ValueError):
                        pose_data.to_joints(data)

    def test_invalid_confidence_and_image_dimensions_are_rejected(self):
        for key, value in (("visibility", float("nan")), ("presence", float("inf")),
                           ("visibility", -.1), ("presence", 1.1)):
            with self.subTest(key=key, value=value):
                data = pose_data.template_pose()
                data["world_landmarks"][13][key] = value
                with self.assertRaises(ValueError):
                    pose_data.validate(data)
        for value in (0, -1, 1.5, True, 100001):
            with self.subTest(width=value):
                data = pose_data.template_pose()
                data["image"]["width"] = value
                with self.assertRaises(ValueError):
                    pose_data.validate(data)


if __name__ == "__main__":
    unittest.main()
