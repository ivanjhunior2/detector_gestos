"""MediaPipe Tasks wrapper for hand tracking on webcam frames."""

from __future__ import annotations

from pathlib import Path
from types import TracebackType

import cv2
import mediapipe as mp
import numpy as np

from gesture_detector.config import AppConfig


class HandTracker:
    """Run MediaPipe Hand Landmarker in synchronous video mode."""

    def __init__(self, config: AppConfig) -> None:
        model_path = Path(config.model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                "No se encontró el modelo de MediaPipe: "
                f"{model_path}. Descarga el modelo Hand Landmarker oficial y "
                "guárdalo con ese nombre dentro de la carpeta models."
            )

        vision = mp.tasks.vision
        options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=config.max_hands,
            min_hand_detection_confidence=config.min_hand_detection_confidence,
            min_hand_presence_confidence=config.min_hand_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def detect(
        self, frame_bgr: np.ndarray, timestamp_ms: int
    ) -> mp.tasks.vision.HandLandmarkerResult:
        """Return hand landmarks for a BGR frame and monotonic timestamp."""
        timestamp_ms = max(timestamp_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        return self._landmarker.detect_for_video(image, timestamp_ms)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> HandTracker:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
