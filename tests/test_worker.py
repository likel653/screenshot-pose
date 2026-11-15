"""Focused worker checks: no model or MediaPipe installation is needed."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

WORKER_PATH = Path(__file__).resolve().parents[1] / "screenshot_pose" / "worker.py"
spec = importlib.util.spec_from_file_location("pose_worker_for_tests", WORKER_PATH)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
try:
    from PIL import Image
except ImportError:
    Image = None


def result_fixture(count=1):
    def points():
        return [SimpleNamespace(x=.3 + i * .01, y=.5, z=-.02,
                                visibility=.8, presence=.9) for i in range(33)]
    return SimpleNamespace(pose_landmarks=[points() for _ in range(count)],
                           pose_world_landmarks=[points() for _ in range(count)])


class WorkerTests(unittest.TestCase):
    def test_json_unicode_and_atomic_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "姿势 结果.json"
            worker.write_json(path, {"path": "人物 截图.png"})
            self.assertEqual(json.loads(path.read_text("utf-8"))["path"], "人物 截图.png")
            before = path.read_bytes()

            def fail(temp):
                temp.write_text("partial")
                raise OSError("disk error")

            with self.assertRaises(OSError):
                worker.atomic_write(path, fail)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(folder).iterdir()), [path])
            with self.assertRaises(ValueError):
                worker.write_json(path, {"x": float("nan")})
            self.assertEqual(path.read_bytes(), before)

    def test_invalid_options(self):
        for index, maximum, confidence in [(-1, 4, .5), (4, 4, .5), (0, 0, .5),
                                            (0, 4, float("nan")), (0, 4, 1.1)]:
            with self.subTest(index=index, maximum=maximum, confidence=confidence):
                with self.assertRaises(worker.WorkerError):
                    worker.validate_options(index, maximum, confidence)

    def test_no_person_and_wrong_person_are_distinct(self):
        args = (Path("processed.png"), Path("original.png"), (800, 600))
        with self.assertRaisesRegex(worker.WorkerError, "没有识别到人物"):
            worker.make_document(result_fixture(0), *args, 0)
        with self.assertRaisesRegex(worker.WorkerError, "只识别到 1 人"):
            worker.make_document(result_fixture(1), *args, 1)

    def test_document_preserves_world_units_and_coordinate_orientation(self):
        data = worker.make_document(result_fixture(2), Path("处理图.png"),
                                    Path("原图.png"), (600, 800), 1)
        self.assertEqual(data["pose_count"], 2)
        self.assertEqual(data["person_index"], 1)
        self.assertEqual(data["image"]["width"], 600)
        self.assertEqual(data["image"]["original_path"], "原图.png")
        self.assertEqual(len(data["world_landmarks"]), 33)
        self.assertEqual(data["world_landmarks"][0]["z"], -.02)
        self.assertEqual(data["landmarks"][0]["x"], .3)

    def test_invalid_coordinate_rejected(self):
        result = result_fixture()
        result.pose_world_landmarks[0][5].z = float("inf")
        with self.assertRaisesRegex(worker.WorkerError, "无效坐标"):
            worker.make_document(result, Path("a.png"), Path("b.png"), (80, 60), 0)

    def test_output_cannot_overwrite_source(self):
        path = Path("source.png").resolve()
        with self.assertRaisesRegex(worker.WorkerError, "不能覆盖"):
            worker._check_paths(path, Path("model.task").resolve(),
                                Path("out.json").resolve(), path, Path("input.png").resolve())

    def test_missing_model_gives_nonzero_exit_and_no_output(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "out.json"
            with contextlib.redirect_stderr(io.StringIO()) as stderr:
                code = worker.main(["--image", str(Path(folder) / "image.png"),
                                    "--model", str(Path(folder) / "missing.task"),
                                    "--output", str(destination)])
            self.assertEqual(code, 2)
            self.assertIn("找不到本地姿势模型", stderr.getvalue())
            self.assertFalse(destination.exists())

    @unittest.skipIf(Image is None, "Pillow not installed")
    def test_exif_orientation_rgb_size_and_unicode_path(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "人物 横向截图.jpg"
            image = Image.new("RGB", (2000, 1000), "red")
            exif = image.getexif()
            exif[274] = 6  # Rotate 90 degrees clockwise for display.
            image.save(path, exif=exif)
            prepared = worker.prepare_image(path)
            self.assertEqual(prepared.size, (800, 1600))
            self.assertEqual(prepared.mode, "RGB")
            saved = Path(folder) / "processed.png"
            worker.save_png(prepared, saved)
            with Image.open(saved) as readback:
                self.assertEqual(readback.size, (800, 1600))
                self.assertNotIn(274, readback.getexif())

    @unittest.skipIf(Image is None, "Pillow not installed")
    def test_overlay_retains_dimensions(self):
        data = worker.make_document(result_fixture(), Path("a"), Path("b"), (400, 600), 0)
        data["landmarks"][15]["visibility"] = .01
        image = Image.new("RGB", (400, 600), "white")
        overlay = worker.draw_overlay(image, data["landmarks"], .5)
        self.assertEqual(overlay.size, image.size)
        self.assertNotEqual(overlay.tobytes(), image.tobytes())

    @unittest.skipIf(Image is None, "Pillow not installed")
    def test_clipboard_file_list_rejected(self):
        with patch("PIL.ImageGrab.grabclipboard", return_value=["C:/screenshot.png"]):
            with self.assertRaisesRegex(worker.WorkerError, "文件列表"):
                worker.save_clipboard("clipboard.png")


if __name__ == "__main__":
    unittest.main()
