"""Drawing helpers for displaying MediaPipe hand landmarks."""

import cv2
import numpy as np


# MediaPipe's 21-point hand topology, expressed as landmark index pairs.
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


def _to_pixel(point, width: int, height: int) -> tuple[int, int]:
    x = min(max(int(point.x * width), 0), width - 1)
    y = min(max(int(point.y * height), 0), height - 1)
    return x, y


def draw_hand_landmarks(
    frame: np.ndarray,
    result,
    status_lines: tuple[str, ...] = (),
) -> np.ndarray:
    """Draw hand skeletons, handedness labels, and optional status text."""
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    hands = result.hand_landmarks
    handedness = result.handedness

    for hand_index, landmarks in enumerate(hands):
        points = [_to_pixel(point, width, height) for point in landmarks]

        for start, end in HAND_CONNECTIONS:
            cv2.line(annotated, points[start], points[end], (70, 210, 90), 2)

        for point in points:
            cv2.circle(annotated, point, 4, (40, 80, 255), -1)

        label = "Mano"
        if hand_index < len(handedness) and handedness[hand_index]:
            label = handedness[hand_index][0].category_name or label

        wrist_x, wrist_y = points[0]
        cv2.putText(
            annotated,
            label,
            (wrist_x + 8, max(wrist_y - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    status = f"Mano detectada: {'si' if hands else 'no'}"
    cv2.putText(
        annotated,
        status,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        "Q o Esc: salir",
        (12, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    for index, line in enumerate(status_lines):
        y_position = 86 + index * 25
        cv2.putText(
            annotated,
            line,
            (12, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (30, 30, 30),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            line,
            (12, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return annotated
