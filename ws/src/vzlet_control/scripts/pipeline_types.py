from pathlib import Path
from typing import Final

PIPELINE_ROOT = Path(__file__).resolve().parents[4]

DEFAULT_CSV_FILE: Final[str] = str(PIPELINE_ROOT / "recorded_poses_floor.csv")
DEFAULT_YOLO_MODEL_FILE: Final[str] = str(PIPELINE_ROOT / "models" / "vzlet_ver4.pt")
# DEFAULT_YOLO_MODEL_FILE: Final[str] = str(PIPELINE_ROOT / "models" / "vzlet_ver2_ncnn_model")

# Vision Tracking Methods
CIRCLE_DETECTION_CONTOUR: Final[str] = "contour"
CIRCLE_DETECTION_YOLO: Final[str] = "yolo"
CIRCLE_DETECTION_MASK: Final[str] = "mask"

CIRCLE_DETECTION_METHODS: Final[set[str]] = {
    CIRCLE_DETECTION_CONTOUR,
    CIRCLE_DETECTION_YOLO,
    CIRCLE_DETECTION_MASK,
}


INITIAL_ZONE: Final[str] = "zone1"
FINAL_ZONE: Final[str] = "zone2"

ACTION_PICK: Final[str] = "pick"
ACTION_PLACE: Final[str] = "place"