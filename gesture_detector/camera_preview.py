"""Camera-only preview entry point, kept separate from MediaPipe."""

import logging

import cv2

from gesture_detector.camera import Camera, CameraError
from gesture_detector.config import AppConfig
from gesture_detector.window import window_is_open


LOGGER = logging.getLogger(__name__)


def run_camera_preview(config: AppConfig | None = None) -> int:
    """Show the webcam feed until Q or Escape is pressed."""
    settings = config or AppConfig()

    try:
        with Camera(settings.camera_index) as camera:
            while True:
                frame = camera.read()
                cv2.imshow("Prueba de cámara", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27) or not window_is_open("Prueba de cámara"):
                    return 0
    except CameraError as error:
        LOGGER.error("%s", error)
        return 1
    finally:
        cv2.destroyAllWindows()
