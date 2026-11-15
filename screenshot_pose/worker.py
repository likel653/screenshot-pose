"""Local image-to-pose worker, run by an external Python (never Blender's bpy).

No model downloads or image uploads are performed here. Input coordinates refer
to the EXIF-corrected, RGB image saved beside the JSON as ``<stem>.input.png``.
Importing this module needs only the standard library; runtime imports are lazy.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

SCHEMA = "screenshot_pose/v1"
MAX_IMAGE_SIDE = 1600
LEFT = {1, 2, 3, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31}
RIGHT = {4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32}
CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
)


class WorkerError(ValueError):
    """An actionable error suitable for Blender's status message."""


def _pillow():
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise WorkerError("缺少 Pillow，请先运行插件的识别环境安装。") from exc
    return Image, ImageOps


def prepare_image(path: str | Path):
    """Return a detached RGB PIL image, oriented correctly and at most 1600 px."""
    Image, ImageOps = _pillow()
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise WorkerError(f"找不到输入图片：{path}")
    try:
        with Image.open(path) as source:
            oriented = ImageOps.exif_transpose(source)
            # Composite transparency over white, rather than interpreting hidden
            # RGB values as an opaque black screenshot background.
            if oriented.mode in ("RGBA", "LA") or "transparency" in oriented.info:
                rgba = oriented.convert("RGBA")
                result = Image.new("RGB", rgba.size, "white")
                result.paste(rgba, mask=rgba.getchannel("A"))
            else:
                result = oriented.convert("RGB")
            result.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
            return result
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise WorkerError(f"图片无法读取，请选择 PNG/JPG/WebP 等有效图片：{exc}") from exc


def atomic_write(path: str | Path, writer) -> Path:
    """Write in the destination directory, replace only after the writer succeeds."""
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temp_path = Path(temp_name)
    try:
        writer(temp_path)
        # Windows readers/antivirus can briefly hold the destination without
        # FILE_SHARE_DELETE. Keep the old complete file and retry the rename.
        delays=(.01,.02,.04,.08,.12)
        for attempt in range(len(delays)+1):
            try:
                os.replace(temp_path, path)
                break
            except PermissionError as exc:
                if os.name!='nt' or getattr(exc,'winerror',None) not in (5,32,33) or attempt==len(delays):
                    raise
                time.sleep(delays[attempt])
    finally:
        temp_path.unlink(missing_ok=True)
    return path


def write_json(path: str | Path, data: dict) -> Path:
    # Serializing before opening the file also keeps old output intact on NaN.
    content = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    return atomic_write(path, lambda temp: temp.write_text(content, encoding="utf-8"))


def write_progress(path: str | Path, data: dict) -> bool:
    """Progress is telemetry, never the commit marker or a reason to abort work."""
    try:
        write_json(path,data)
        return True
    except OSError:
        return False


def save_png(image, path: str | Path) -> Path:
    return atomic_write(path, lambda temp: image.save(temp, format="PNG"))


def save_clipboard(output: str | Path) -> Path:
    Image, ImageOps = _pillow()
    try:
        from PIL import ImageGrab
        content = ImageGrab.grabclipboard()
    except (OSError, NotImplementedError) as exc:
        raise WorkerError(f"当前系统无法读取剪贴板图片：{exc}") from exc
    if isinstance(content, list):
        raise WorkerError("剪贴板里是文件列表。请复制截图本身，或使用插件的图片文件选择器。")
    if not isinstance(content, Image.Image):
        raise WorkerError("剪贴板中没有图片，请先截图或复制图片。")
    if Path(output).suffix.lower() != ".png":
        raise WorkerError("剪贴板图片的输出路径必须以 .png 结尾。")
    return save_png(ImageOps.exif_transpose(content).convert("RGB"), output)


def validate_options(person_index: int, max_poses: int, confidence: float):
    if isinstance(max_poses, bool) or not isinstance(max_poses, int) or not 1 <= max_poses <= 16:
        raise WorkerError("最多识别人数必须是 1 至 16 的整数。")
    if isinstance(person_index, bool) or not isinstance(person_index, int) or not 0 <= person_index < max_poses:
        raise WorkerError("人物序号从 0 开始，且必须小于最多识别人数。")
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise WorkerError("识别置信度必须是 0 至 1 之间的有限数值。")


def _landmark_dict(point: Any, fallback: dict | None = None) -> dict:
    values = {axis: float(getattr(point, axis)) for axis in ("x", "y", "z")}
    if not all(math.isfinite(value) for value in values.values()):
        raise WorkerError("识别结果含无效坐标，请换一张更清晰的图片。")
    for key in ("visibility", "presence"):
        value = getattr(point, key, None)
        if value is None:
            value = (fallback or {}).get(key, 1.0)
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise WorkerError("识别结果含无效关节置信度。")
        values[key] = value
    return values


def make_document(result: Any, image_path: Path, original_path: Path,
                  size: tuple[int, int], person_index: int) -> dict:
    count = len(result.pose_landmarks)
    if count == 0:
        raise WorkerError("没有识别到人物。请裁剪到单个人物、包含主要四肢，并尝试降低识别置信度；卡通截图也可以改用手动校正。")
    if not 0 <= person_index < count:
        raise WorkerError(f"只识别到 {count} 人，人物序号应为 0 至 {count - 1}。")
    if len(result.pose_world_landmarks) != count:
        raise WorkerError("模型没有返回完整的三维骨骼数据。")
    points, world = result.pose_landmarks[person_index], result.pose_world_landmarks[person_index]
    if len(points) != 33 or len(world) != 33:
        raise WorkerError("模型返回的关节点数量不正确，预期为 33。")
    normalized = [_landmark_dict(point) for point in points]
    world_points = [_landmark_dict(point, normalized[index]) for index, point in enumerate(world)]
    return {
        "schema": SCHEMA,
        "source": "mediapipe_pose_landmarker",
        "image": {"path": str(image_path), "original_path": str(original_path),
                  "width": int(size[0]), "height": int(size[1])},
        "person_index": person_index,
        "pose_count": count,
        "landmarks": normalized,
        "world_landmarks": world_points,
    }


def draw_overlay(image, points: list[dict], confidence: float, person_index: int = 0):
    """Blue = subject's left, red = right, yellow = uncertain/out of view."""
    from PIL import ImageDraw
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    width, height = preview.size
    radius = max(3, round(min(width, height) / 150))
    thickness = max(2, radius // 2)
    xy = [(p["x"] * (width - 1), p["y"] * (height - 1)) for p in points]
    reliable = [min(p["visibility"], p["presence"]) >= confidence and
                0 <= p["x"] <= 1 and 0 <= p["y"] <= 1 for p in points]
    left_color, right_color = (63, 166, 255), (255, 91, 108)
    uncertain, central = (255, 203, 57), (225, 239, 246)
    for a, b in CONNECTIONS:
        if points[a].get("outside_frame") or points[b].get("outside_frame"):
            continue
        color = (left_color if a in LEFT and b in LEFT else
                 right_color if a in RIGHT and b in RIGHT else central)
        if not (reliable[a] and reliable[b]):
            color = uncertain
        draw.line((xy[a], xy[b]), fill=color, width=thickness)
    for index, (x, y) in enumerate(xy):
        if points[index].get("outside_frame"):
            continue
        # Out-of-frame landmarks are projected to the border for visibility.
        x, y = max(radius, min(width - 1 - radius, x)), max(radius, min(height - 1 - radius, y))
        color = left_color if index in LEFT else right_color if index in RIGHT else central
        bounds = (x - radius, y - radius, x + radius, y + radius)
        if reliable[index]:
            draw.ellipse(bounds, fill=color, outline=(20, 25, 32), width=1)
        else:
            draw.ellipse(bounds, outline=uncertain, width=thickness)
            draw.line((x - radius, y - radius, x + radius, y + radius), fill=uncertain, width=1)
    label = f"Person {person_index} | L: blue | R: red | low confidence: yellow"
    box = draw.textbbox((7, 7), label)
    draw.rectangle((0, 0, min(width, box[2] + 7), box[3] + 7), fill=(17, 23, 32))
    draw.text((7, 7), label, fill=(255, 255, 255))
    return preview


def _check_paths(image: Path, model: Path, output: Path, overlay: Path | None, processed: Path):
    destinations = [output, processed] + ([overlay] if overlay else [])
    normalized = [os.path.normcase(str(p)) for p in destinations]
    if len(set(normalized)) != len(normalized):
        raise WorkerError("JSON、预处理图片和预览图片必须使用不同的输出路径。")
    sources = {os.path.normcase(str(image)), os.path.normcase(str(model))}
    if sources.intersection(normalized):
        raise WorkerError("输出路径不能覆盖输入图片或识别模型。")


def detect(image_path: str | Path, model_path: str | Path, output: str | Path,
           overlay: str | Path | None = None, person_index: int = 0,
           max_poses: int = 4, confidence: float = 0.5) -> dict:
    validate_options(person_index, max_poses, confidence)
    original = Path(image_path).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    overlay = Path(overlay).expanduser().resolve() if overlay else None
    processed = output.with_name(output.stem + ".input.png")
    _check_paths(original, model_path, output, overlay, processed)
    if not model_path.is_file():
        raise WorkerError(f"找不到本地姿势模型：{model_path}。请先安装识别环境或选择 .task 模型。")
    image = prepare_image(original)
    # MediaPipe imports Matplotlib. Keep its incidental font cache in the
    # process user's temporary directory, not a possibly read-only profile.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "screenshot_pose_mpl"))
    try:
        import numpy as np
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision
    except (ImportError, OSError) as exc:
        raise WorkerError(f"姿势识别依赖无法加载，请使用 Python 3.12/3.13 重新安装独立识别环境：{exc}") from exc
    # Bytes avoid native library path-encoding failures on Windows Chinese paths.
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=model_path.read_bytes(),
                                 delegate=BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=max_poses,
        min_pose_detection_confidence=confidence,
        min_pose_presence_confidence=confidence,
        min_tracking_confidence=confidence,
        output_segmentation_masks=False,
    )
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=np.ascontiguousarray(np.asarray(image), dtype=np.uint8))
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)
    data = make_document(result, processed, original, image.size, person_index)
    # JSON is the commit marker: callers consume it only after a zero exit code.
    save_png(image, processed)
    if overlay:
        save_png(draw_overlay(image, data["landmarks"], confidence, person_index), overlay)
    write_json(output, data)
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="本地截图姿势识别；不会上传图片或自动下载模型。")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image", help="输入截图路径")
    mode.add_argument("--clipboard", action="store_true", help="将剪贴板图片保存到 --output PNG")
    parser.add_argument("--model", help="本地 MediaPipe Pose Landmarker .task 模型")
    parser.add_argument("--output", required=True, help="姿势 JSON，或剪贴板模式的 PNG 输出路径")
    parser.add_argument("--overlay", help="可选的关节点预览 PNG")
    parser.add_argument("--person-index", type=int, default=0)
    parser.add_argument("--max-poses", type=int, default=4)
    parser.add_argument("--confidence", type=float, default=0.5)
    args = parser.parse_args(argv)
    if args.image and not args.model:
        parser.error("图片识别模式必须指定 --model 本地模型。")
    if args.clipboard and (args.model or args.overlay):
        parser.error("--clipboard 仅保存图片，请勿同时指定 --model 或 --overlay。")
    try:
        if args.clipboard:
            saved = save_clipboard(args.output)
            print(json.dumps({"output": str(saved)}, ensure_ascii=False))
        else:
            data = detect(args.image, args.model, args.output, args.overlay,
                          args.person_index, args.max_poses, args.confidence)
            print(json.dumps({"output": str(Path(args.output).resolve()),
                              "pose_count": data["pose_count"],
                              "person_index": data["person_index"]}, ensure_ascii=False))
        return 0
    except (WorkerError, OSError, RuntimeError, ValueError) as exc:
        print(f"截图姿势识别失败：{exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"截图姿势识别失败（{type(exc).__name__}）：{exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
