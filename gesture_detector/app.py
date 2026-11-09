"""Application loops for hand tracking and rule-based gesture typing."""

import logging
import time

import cv2

from gesture_detector.camera import Camera, CameraError
from gesture_detector.config import AppConfig, GESTURE_DISPLAY_NAMES, GESTURE_TO_KEY
from gesture_detector.event_gate import GestureEventGate
from gesture_detector.gesture_rules import GestureRuleRecognizer
from gesture_detector.hand_tracker import HandTracker
from gesture_detector.keyboard import KeyboardController
from gesture_detector.visualizer import draw_hand_landmarks
from gesture_detector.window import window_is_open


LOGGER = logging.getLogger(__name__)


def run_hand_tracking(config: AppConfig | None = None) -> int:
    """Open the camera, detect up to two hands, and display their landmarks."""
    settings = config or AppConfig()

    try:
        with HandTracker(settings) as tracker, Camera(settings.camera_index) as camera:
            while True:
                frame = camera.read()
                timestamp_ms = time.monotonic_ns() // 1_000_000
                result = tracker.detect(frame, timestamp_ms)
                annotated = draw_hand_landmarks(frame, result)
                cv2.imshow(settings.window_title, annotated)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27) or not window_is_open(settings.window_title):
                    return 0
    except (CameraError, FileNotFoundError, RuntimeError) as error:
        LOGGER.error("No se pudo iniciar el detector: %s", error)
        return 1
    finally:
        cv2.destroyAllWindows()


def run_gesture_typing(
    send_keys: bool = False,
    config: AppConfig | None = None,
) -> int:
    """Apply geometric gesture rules and emit one keypress per stable pose."""
    settings = config or AppConfig()
    gate = GestureEventGate(
        stable_frames=settings.gesture_stable_frames,
        release_frames=settings.gesture_release_frames,
    )
    keyboard = KeyboardController(enabled=send_keys)
    recognizer = GestureRuleRecognizer()
    last_action = "Sin eventos"

    try:
        with HandTracker(settings) as tracker, Camera(settings.camera_index) as camera:
            while True:
                frame = camera.read()
                timestamp_ms = time.monotonic_ns() // 1_000_000
                result = tracker.detect(frame, timestamp_ms)
                predicted_label = recognizer.classify(result, frame.shape)
                event = gate.update(predicted_label)
                if event is not None:
                    last_action = keyboard.press_for_gesture(event)
                    print(last_action)

                if predicted_label is None:
                    prediction_text = "Gesto: sin regla coincidente"
                else:
                    display_name = GESTURE_DISPLAY_NAMES[predicted_label]
                    key_name = GESTURE_TO_KEY[predicted_label].upper()
                    prediction_text = f"Gesto: {display_name} -> {key_name}"

                mode_text = (
                    "TECLAS ACTIVAS"
                    if send_keys
                    else "SIMULACION: no envia teclas"
                )
                gate_text = "Listo" if gate.armed else "Gesto activo"
                status_lines = (
                    mode_text,
                    prediction_text,
                    f"Estado: {gate_text} | Ultima accion: {last_action}",
                    "Enfoca primero el editor o la aplicacion destino.",
                )
                annotated = draw_hand_landmarks(frame, result, status_lines)
                cv2.imshow(settings.window_title, annotated)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27) or not window_is_open(settings.window_title):
                    return 0
    except (CameraError, FileNotFoundError, RuntimeError, OSError, ValueError) as error:
        LOGGER.error("No se pudo iniciar la escritura por gestos: %s", error)
        return 1
    finally:
        cv2.destroyAllWindows()
