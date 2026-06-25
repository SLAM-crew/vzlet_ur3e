#!/usr/bin/env python3
"""
YOLO OpenCV debug preview with live confidence sliders.

Input:
  - image file: displays a static preview with boxes
  - video file / camera index: displays a live preview with boxes

Output:
  - OpenCV window only
  - does not save jpg/png/video outputs

Examples:
  python yolo_debug_preview.py --source image.jpg --model best.pt
  python yolo_debug_preview.py --source video.mp4 --model best.pt
  python yolo_debug_preview.py --source 0 --model best.pt

Controls:
  q / ESC : quit
  Space   : pause/resume video
  n       : step one frame when paused
  r       : restart video
"""

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
WINDOW_NAME = "YOLO debug preview"
SLIDER_MIN_CONF = "min conf x100"
SLIDER_MAX_CONF = "max conf x100"
SLIDER_MIN_AREA = "min area px"
SLIDER_CLASS_ID = "class id (-1 all)"


def noop(_value: int) -> None:
    pass


def parse_source(source: str) -> Union[str, int]:
    """
    Treat a numeric source like '0' as a camera index.
    Otherwise use it as a filesystem path.
    """
    if source.isdigit():
        return int(source)
    return source


def is_image_source(source: Union[str, int]) -> bool:
    if isinstance(source, int):
        return False
    return Path(source).suffix.lower() in IMAGE_SUFFIXES


def clamp_slider_value(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def make_window(initial_min_conf: float, initial_max_conf: float, initial_min_area: int) -> None:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    cv2.createTrackbar(
        SLIDER_MIN_CONF,
        WINDOW_NAME,
        clamp_slider_value(int(round(initial_min_conf * 100.0)), 0, 100),
        100,
        noop,
    )
    cv2.createTrackbar(
        SLIDER_MAX_CONF,
        WINDOW_NAME,
        clamp_slider_value(int(round(initial_max_conf * 100.0)), 0, 100),
        100,
        noop,
    )
    cv2.createTrackbar(
        SLIDER_MIN_AREA,
        WINDOW_NAME,
        max(0, int(initial_min_area)),
        20000,
        noop,
    )

    # OpenCV sliders cannot be negative, so:
    #   0  -> all classes
    #   1  -> class id 0
    #   2  -> class id 1
    #   ...
    cv2.createTrackbar(
        SLIDER_CLASS_ID,
        WINDOW_NAME,
        0,
        100,
        noop,
    )


def read_filters() -> Tuple[float, float, int, Optional[int]]:
    min_conf = cv2.getTrackbarPos(SLIDER_MIN_CONF, WINDOW_NAME) / 100.0
    max_conf = cv2.getTrackbarPos(SLIDER_MAX_CONF, WINDOW_NAME) / 100.0

    if min_conf > max_conf:
        min_conf, max_conf = max_conf, min_conf

    min_area = cv2.getTrackbarPos(SLIDER_MIN_AREA, WINDOW_NAME)

    raw_class_slider = cv2.getTrackbarPos(SLIDER_CLASS_ID, WINDOW_NAME)
    class_id = None if raw_class_slider == 0 else raw_class_slider - 1

    return min_conf, max_conf, min_area, class_id


def pick_color(class_id: int) -> Tuple[int, int, int]:
    """
    Deterministic BGR color per class.
    """
    rng = np.random.default_rng(class_id + 12345)
    color = rng.integers(40, 255, size=3).tolist()
    return int(color[0]), int(color[1]), int(color[2])


def draw_label(
    image: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    color: Tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1

    x, y = origin
    y = max(18, y)

    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(
        image,
        (x, y - th - baseline - 4),
        (x + tw + 6, y + baseline),
        color,
        -1,
    )
    cv2.putText(
        image,
        text,
        (x + 3, y - 4),
        font,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def draw_info_panel(
    image: np.ndarray,
    lines: Iterable[str],
    x: int = 10,
    y: int = 24,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    line_height = 22

    for i, line in enumerate(lines):
        yy = y + i * line_height
        cv2.putText(
            image,
            line,
            (x + 1, yy + 1),
            font,
            scale,
            (0, 0, 0),
            thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            line,
            (x, yy),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def render_debug_frame(
    frame: np.ndarray,
    model: YOLO,
    names: Dict[int, str],
    imgsz: Optional[int],
) -> np.ndarray:
    min_conf, max_conf, min_area, class_id_filter = read_filters()

    # Use conf=0.001 to get almost everything, then filter manually with sliders.
    # This lets the slider range control both lower and upper confidence after inference.
    predict_kwargs = {
        "source": frame,
        "verbose": False,
        "conf": 0.001,
    }
    if imgsz is not None:
        predict_kwargs["imgsz"] = imgsz

    results = model.predict(**predict_kwargs)
    debug = frame.copy()

    kept = 0
    total = 0

    if results:
        result = results[0]
        boxes = getattr(result, "boxes", None)

        if boxes is not None:
            for box in boxes:
                total += 1

                cls_id = int(box.cls.item()) if getattr(box, "cls", None) is not None else -1
                conf = float(box.conf.item()) if getattr(box, "conf", None) is not None else 0.0

                x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().tolist()
                x1 = float(x1)
                y1 = float(y1)
                x2 = float(x2)
                y2 = float(y2)

                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                area = w * h

                if conf < min_conf or conf > max_conf:
                    continue
                if area < float(min_area):
                    continue
                if class_id_filter is not None and cls_id != class_id_filter:
                    continue

                kept += 1

                pt1 = (int(round(x1)), int(round(y1)))
                pt2 = (int(round(x2)), int(round(y2)))
                center = (int(round(0.5 * (x1 + x2))), int(round(0.5 * (y1 + y2))))

                color = pick_color(cls_id)
                cv2.rectangle(debug, pt1, pt2, color, 2)
                cv2.drawMarker(
                    debug,
                    center,
                    color,
                    markerType=cv2.MARKER_CROSS,
                    markerSize=12,
                    thickness=2,
                )

                class_name = names.get(cls_id, str(cls_id))
                label = f"{class_name} id={cls_id} conf={conf:.2f} area={area:.0f}"
                draw_label(debug, label, (pt1[0], pt1[1] - 6), color)

    class_text = "all" if class_id_filter is None else str(class_id_filter)
    draw_info_panel(
        debug,
        [
            f"kept/total: {kept}/{total}",
            f"conf range: {min_conf:.2f}..{max_conf:.2f}",
            f"min area: {min_area}px",
            f"class id: {class_text}",
            "q/ESC quit | Space pause | n step | r restart",
        ],
    )

    return debug


def run_image(source: str, model: YOLO, names: Dict[int, str], imgsz: Optional[int]) -> None:
    frame = cv2.imread(source, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Could not read image: {source}")

    while True:
        debug = render_debug_frame(frame, model, names, imgsz)
        cv2.imshow(WINDOW_NAME, debug)

        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q")):
            break


def run_video(source: Union[str, int], model: YOLO, names: Dict[int, str], imgsz: Optional[int]) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video/camera source: {source}")

    paused = False
    last_frame = None

    while True:
        if not paused or last_frame is None:
            ok, frame = cap.read()

            if not ok:
                # For files, loop back to the start. For cameras, keep trying.
                if isinstance(source, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()

                if not ok:
                    key = cv2.waitKey(30) & 0xFF
                    if key in (27, ord("q")):
                        break
                    continue

            last_frame = frame

        debug = render_debug_frame(last_frame, model, names, imgsz)
        cv2.imshow(WINDOW_NAME, debug)

        key = cv2.waitKey(1 if not paused else 30) & 0xFF

        if key in (27, ord("q")):
            break
        if key == ord(" "):
            paused = not paused
        elif key == ord("n"):
            paused = True
            ok, frame = cap.read()
            if ok:
                last_frame = frame
        elif key == ord("r"):
            if isinstance(source, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                last_frame = None

    cap.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenCV YOLO debug preview with lower/upper confidence sliders."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to image/video, or camera index like 0.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to Model, e.g. best.pt.",
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.25,
        help="Initial lower confidence threshold. Default: 0.25.",
    )
    parser.add_argument(
        "--max-conf",
        type=float,
        default=1.0,
        help="Initial upper confidence threshold. Default: 1.0.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=0,
        help="Initial minimum bbox area in pixels. Default: 0.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Optional YOLO inference image size.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    source = parse_source(args.source)

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(str(model_path))
    names = model.names if isinstance(model.names, dict) else {}

    make_window(args.min_conf, args.max_conf, args.min_area)

    try:
        if is_image_source(source):
            run_image(str(source), model, names, args.imgsz)
        else:
            run_video(source, model, names, args.imgsz)
    finally:
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
