"""Central configuration for the local hand-tracking prototype."""

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"


@dataclass(frozen=True)
class AppConfig:
    """Settings shared by the camera and hand-tracking applications."""

    camera_index: int = 0
    max_hands: int = 2
    window_title: str = "Detector de gestos - landmarks"
    model_path: Path = field(
        default_factory=lambda: MODEL_DIR / "hand_landmarker.task"
    )
    min_hand_detection_confidence: float = 0.5
    min_hand_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    gesture_stable_frames: int = 4
    gesture_release_frames: int = 5


GESTURE_TO_KEY: dict[str, str] = {
    "bode": "q",
    "cachorro": "w",
    "javali": "e",
    "palma_abierta": "r",
    "cavalo": "d",
    "passaro": "f",
}
GESTURE_DISPLAY_NAMES: dict[str, str] = {
    "bode": "Bode",
    "cachorro": "Cachorro",
    "javali": "Javali",
    "palma_abierta": "Palma abierta",
    "cavalo": "Cavalo",
    "passaro": "Pássaro",
}


GESTURE_HAND_COUNTS = {label: 1 if label == "palma_abierta" else 2 for label in GESTURE_TO_KEY}
