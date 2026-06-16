from pathlib import Path
from typing import Final

PIPELINE_ROOT = Path(__file__).resolve().parents[4]

ZONE_CSV_FILE: Final[str] = str(PIPELINE_ROOT / "zone_poses_floor.csv")
DEFAULT_YOLO_MODEL_FILE: Final[str] = str(PIPELINE_ROOT / "models" / "vzlet_ver5.pt")

INITIAL_ZONE: Final[str] = "00"
FINAL_ZONE: Final[str] = "zone2"

ACTION_PICK: Final[str] = "pick"
ACTION_PLACE: Final[str] = "place"