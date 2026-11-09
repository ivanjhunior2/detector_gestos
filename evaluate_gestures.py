"""Measure each gesture in both orientations and save replayable landmarks."""

import argparse
from collections import Counter
from datetime import datetime
from itertools import groupby
import json
from pathlib import Path
import time

import cv2

from gesture_detector.camera import Camera
from gesture_detector.config import (
    AppConfig, GESTURE_DISPLAY_NAMES, GESTURE_HAND_COUNTS, GESTURE_TO_KEY,
)
from gesture_detector.event_gate import GestureEventGate
from gesture_detector.gesture_rules import GestureRuleRecognizer
from gesture_detector.hand_tracker import HandTracker
from gesture_detector.visualizer import draw_hand_landmarks
from gesture_detector.window import window_is_open


WINDOW = "Evaluacion de gestos"
PREPARE_SECONDS = 3
HOLD_SECONDS = 2


def serialize_landmarks(hands):
    return [[[point.x, point.y, point.z] for point in hand] for hand in hands]


def evaluate(repetitions: int, output: Path) -> None:
    config = AppConfig()
    recognizer = GestureRuleRecognizer()
    attempts = [
        (label, orientation, repetition)
        for label in GESTURE_TO_KEY
        for repetition in range(1, repetitions + 1)
        for orientation in ("habitual", "invertida")
    ]
    results: dict[tuple[str, str], Counter] = {}
    with Camera(config.camera_index) as camera, HandTracker(config) as tracker, output.open(
        "x", encoding="utf-8"
    ) as report:
        for attempt_index, (expected, orientation, repetition) in enumerate(attempts, 1):
            key = GESTURE_TO_KEY[expected]
            expected_hands = GESTURE_HAND_COUNTS[expected]
            pose_hint = orientation
            if expected_hands == 1:
                pose_hint = "mano izquierda" if orientation == "habitual" else "mano derecha"
            title = f"{GESTURE_DISPLAY_NAMES[expected]} ({key.upper()}) | {pose_hint} | {repetition}/{repetitions}"
            frames = []
            events = []
            gate = GestureEventGate(config.gesture_stable_frames, config.gesture_release_frames)
            started = None
            while True:
                frame = camera.read()
                now = time.monotonic_ns()
                result = tracker.detect(frame, now // 1_000_000)
                elapsed = None if started is None else (now - started) / 1e9
                if elapsed is not None and elapsed >= PREPARE_SECONDS + HOLD_SECONDS:
                    break
                if elapsed is None:
                    prompt = "ESPACIO: iniciar | ESC o X: salir"
                elif elapsed < PREPARE_SECONDS:
                    prompt = f"Prepara el sello: {PREPARE_SECONDS - int(elapsed)}"
                else:
                    prompt = "Manten el sello"
                    prediction = recognizer.classify(result, frame.shape)
                    event = gate.update(prediction)
                    if event is not None:
                        events.append(event)
                    frames.append({
                        "elapsed_ms": round((elapsed - PREPARE_SECONDS) * 1000),
                        "prediction": prediction,
                        "landmarks": serialize_landmarks(result.hand_landmarks),
                        "world_landmarks": serialize_landmarks(result.hand_world_landmarks),
                    })
                annotated = draw_hand_landmarks(frame, result, (title, prompt))
                cv2.imshow(WINDOW, annotated)
                pressed = cv2.waitKey(1) & 0xFF
                if pressed in (27, ord("q")) or not window_is_open(WINDOW):
                    return
                if pressed == ord(" ") and started is None:
                    started = now

            def has_expected_hands(sample):
                return len(sample["landmarks"]) == expected_hands

            tracked_frames = sum(map(has_expected_hands, frames))
            longest_tracking = max((
                sum(1 for _ in group)
                for tracked, group in groupby(frames, has_expected_hands)
                if tracked
            ), default=0)
            if events == [expected]:
                outcome = "correcto"
            elif events:
                outcome = "pulsacion_incorrecta_o_repetida"
            elif longest_tracking < config.gesture_stable_frames:
                outcome = "seguimiento_insuficiente"
            else:
                outcome = "sin_pulsacion"
            record = {
                "attempt": attempt_index, "expected": expected, "key": key,
                "orientation": orientation, "repetition": repetition,
                "outcome": outcome, "events": events, "frame_shape": list(frame.shape),
                "expected_hands": expected_hands, "tracked_frames": tracked_frames,
                "longest_tracking_streak": longest_tracking,
                "frames": frames,
            }
            report.write(json.dumps(record, ensure_ascii=False) + "\n")
            report.flush()
            results.setdefault((key, orientation), Counter())[outcome] += 1
            print(f"{title}: {outcome} ({tracked_frames}/{len(frames)} fotogramas con {expected_hands} mano(s))", flush=True)

    for (key, orientation), counts in results.items():
        print(f"{key.upper()} {orientation}: {counts['correcto']}/{sum(counts.values())} correctos")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intentos", type=int, default=3, help="Intentos por sello y orientación")
    args = parser.parse_args()
    if args.intentos < 1:
        parser.error("--intentos debe ser al menos 1")
    output = Path(f"evaluacion_gestos_{datetime.now():%Y%m%d_%H%M%S_%f}.jsonl")
    print(f"Informe de puntos de las manos: {output.resolve()}", flush=True)
    try:
        evaluate(args.intentos, output)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
