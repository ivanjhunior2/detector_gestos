"""Small resource-safe wrapper around an OpenCV camera capture."""

from __future__ import annotations

from types import TracebackType

import cv2
import numpy as np


class CameraError(RuntimeError):
    """Raised when the webcam cannot be opened or stops returning frames."""


class Camera:
    """Own a webcam connection and release it reliably on exit."""

    def __init__(self, camera_index: int = 0) -> None:
        self.camera_index = camera_index
        self._capture: cv2.VideoCapture | None = None

    def __enter__(self) -> Camera:
        self._capture = cv2.VideoCapture(self.camera_index)
        if not self._capture.isOpened():
            self.release()
            raise CameraError(
                f"No se pudo abrir la cámara con índice {self.camera_index}. "
                "Comprueba que esté conectada y que Windows permita el acceso."
            )
        return self

    def read(self) -> np.ndarray:
        """Return a mirror-view BGR frame or raise a useful error."""
        if self._capture is None:
            raise CameraError("La cámara no está abierta.")

        success, frame = self._capture.read()
        if not success or frame is None:
            raise CameraError("La cámara dejó de entregar fotogramas.")
        # Present the feed like a mirror: moving left then appears left on screen.
        return cv2.flip(frame, 1)

    def release(self) -> None:
        """Release the capture safely; repeated calls are allowed."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()
