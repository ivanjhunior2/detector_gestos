"""Geometric rules for recognizing hand seals and an open palm.

These are deliberately conservative first-pass rules. The reference sheet is a
stylized drawing, so the thresholds need calibration against the live webcam.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, degrees

import numpy as np


@dataclass(frozen=True)
class HandPose:
    """Normalized geometric description of one detected hand."""

    center: np.ndarray
    wrist: np.ndarray
    palm_axis: np.ndarray
    palm_scale: float
    extended: tuple[bool | None, bool | None, bool | None, bool | None]
    finger_tips: np.ndarray
    index_tip: np.ndarray
    middle_tip: np.ndarray
    index_direction: np.ndarray

    palm_score: float = 0.0

    @property
    def horizontality(self) -> float:
        return abs(float(self.palm_axis[0]))


class GestureRuleRecognizer:
    """Classify single-hand and two-hand poses."""

    EXTENDED_ANGLE_DEGREES = 150.0
    CURLED_ANGLE_DEGREES = 132.0
    MIN_SCORE = 0.60
    MIN_LEAD = 0.12

    def classify(self, result, frame_shape: tuple[int, ...]) -> str | None:
        """Return the best supported gesture when it clearly leads."""
        hand_count = len(result.hand_landmarks)
        if hand_count not in (1, 2):
            return None

        height, width = frame_shape[:2]
        if width <= 0 or height <= 0:
            return None

        world_hands = getattr(result, "hand_world_landmarks", ())
        poses = [
            self._measure_pose(
                landmarks, width, height,
                world_hands[index] if index < len(world_hands) else (),
            )
            for index, landmarks in enumerate(result.hand_landmarks)
        ]
        valid_poses = [pose for pose in poses if pose is not None]
        if len(valid_poses) != hand_count:
            return None
        if hand_count == 1:
            hand = valid_poses[0]
            if hand.palm_score >= self.MIN_SCORE:
                return "palma_abierta"
            return None
        # Screen order makes pair rules independent of MediaPipe's handedness order.
        left, right = sorted(valid_poses, key=lambda pose: float(pose.center[0]))
        pair_scale = (left.palm_scale + right.palm_scale) / 2.0
        if pair_scale <= 1.0:
            return None

        center_gap = float(np.linalg.norm(right.center - left.center)) / pair_scale
        height_gap = abs(float(left.center[1] - right.center[1])) / pair_scale
        index_gap = float(np.linalg.norm(left.index_tip - right.index_tip)) / pair_scale
        middle_gap = float(np.linalg.norm(left.middle_tip - right.middle_tip)) / pair_scale
        hands = (left, right)
        upper, lower = sorted(hands, key=lambda pose: float(pose.center[1]))
        upper_fingers = self._openness(upper)
        lower_fingers = self._openness(lower)

        nearby = self._evidence(center_gap, 1.90, 1.30)
        downward = min(self._evidence(hand.palm_axis[1], 0.08, 0.52) for hand in hands)
        curled = min(1.0 - self._openness(hand).mean() for hand in hands)

        # Both triangular seals share an apex. The middle fingers decide which.
        pointing_up = min(self._evidence(-hand.index_direction[1], 0.10, 0.62) for hand in hands)
        apex = min(nearby, pointing_up, self._evidence(index_gap, 1.18, 0.55))
        converging = self._evidence(min(left.index_direction[0], -right.index_direction[0]), -0.15, 0.28)
        fingers = np.array([self._openness(hand) for hand in hands])
        middle_open = float(fingers[:, 1].mean())
        outer_folded = float(1.0 - fingers[:, 2:].mean())
        index_open = float(fingers[:, 0].mean())
        apex *= 0.45 + 0.55 * outer_folded
        lower_folded = float(1.0 - lower_fingers.mean())

        scores = {
            "bode": nearby * max(self._bode_score(raised, support, pair_scale)
                                 for raised, support in (hands, hands[::-1])),
            "javali": min(nearby, downward) * (
                0.55 * curled + 0.25 * self._evidence(height_gap, 1.12, 0.44)
                + 0.20 * self._evidence(index_gap, 1.74, 0.90)
            ),
            "cachorro": min(nearby, self._evidence(upper.horizontality, 0.26, 0.76),
                            self._evidence(lower_folded, 0.15, 0.65)) * (
                0.45 * float(upper_fingers[:2].mean()) + 0.35 * lower_folded
                + 0.20 * self._evidence(height_gap, 0.04, 0.36)
            ),
            "cavalo": apex * (
                0.55 * (1.0 - middle_open) + 0.20 * outer_folded
                + 0.15 * converging + 0.10 * index_open
            ),
            "passaro": apex * (
                0.55 * middle_open + 0.15 * outer_folded + 0.10 * converging
                + 0.10 * index_open + 0.10 * self._evidence(middle_gap, 1.32, 0.62)
            ),
        }
        best, runner_up, *_ = sorted(scores, key=scores.__getitem__, reverse=True)
        if scores[best] >= self.MIN_SCORE and scores[best] - scores[runner_up] >= self.MIN_LEAD:
            return best
        return None

    def _bode_score(self, raised: HandPose, support: HandPose, scale: float) -> float:
        tip_height = (raised.center[1] - raised.index_tip[1]) / raised.palm_scale
        tip_lead = (support.index_tip[1] - raised.index_tip[1]) / scale
        base_gap = np.linalg.norm(support.finger_tips - raised.center, axis=1).min() / scale
        support_folded = 1.0 - self._openness(support).mean()
        return min(
            self._evidence(-raised.index_direction[1], 0.15, 0.76),
            self._evidence(tip_height, 0.22, 0.66),
            self._evidence(tip_lead, 0.05, 0.48),
        ) * (0.70 * self._evidence(base_gap, 1.65, 0.70) + 0.30 * support_folded)

    @staticmethod
    def _openness(hand: HandPose) -> np.ndarray:
        return np.array([0.5 if finger is None else float(finger) for finger in hand.extended])

    @staticmethod
    def _evidence(value: float, weak: float, strong: float) -> float:
        """Grade a geometric cue from zero evidence to a strong match."""
        return max(0.0, min(1.0, (value - weak) / (strong - weak)))

    def _measure_pose(
        self, landmarks, width: int, height: int, world_landmarks=()
    ) -> HandPose | None:
        points = np.asarray(
            [(landmark.x * width, landmark.y * height) for landmark in landmarks],
            dtype=np.float32,
        )
        if points.shape != (21, 2) or not np.isfinite(points).all():
            return None

        wrist = points[0]
        middle_mcp = points[9]
        palm_scale = float(np.linalg.norm(middle_mcp - wrist))
        if palm_scale < 1.0:
            return None

        joint_points = np.asarray(
            [(point.x, point.y, point.z) for point in world_landmarks], dtype=np.float32
        )
        if joint_points.shape != (21, 3) or not np.isfinite(joint_points).all():
            joint_points = points

        palm_axis = self._unit(middle_mcp - wrist)
        center = np.mean(points[[0, 5, 9, 13, 17]], axis=0)
        finger_indices = ((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20))
        extended = tuple(
            self._finger_is_extended(joint_points, mcp, pip, tip)
            for mcp, pip, tip in finger_indices
        )
        return HandPose(
            center=center,
            wrist=wrist,
            palm_axis=palm_axis,
            palm_scale=palm_scale,
            extended=extended,
            finger_tips=points[[8, 12, 16, 20]],
            index_tip=points[8],
            middle_tip=points[12],
            index_direction=self._unit(points[8] - points[5]),
            palm_score=self._open_palm_score(points, palm_axis, palm_scale),
        )

    def _open_palm_score(self, points: np.ndarray, palm_axis: np.ndarray, palm_scale: float) -> float:
        # Fingertip reach allows slightly bent fingers without calling a folded
        # finger open. The thumb must still stick out on the index side.
        reach = (points[[8, 12, 16, 20]] - points[[5, 9, 13, 17]]) @ palm_axis / palm_scale
        thumb_side = self._unit(points[5] - points[17])
        thumb_offset = points[4] - points[5]
        return min(
            self._evidence(reach.min(), 0.10, 0.30),
            self._evidence(reach.mean(), 0.23, 0.46),
            self._evidence(thumb_offset @ thumb_side / palm_scale, 0.06, 0.20),
            self._evidence(np.linalg.norm(thumb_offset) / palm_scale, 0.20, 0.38),
        )

    def _finger_is_extended(
        self,
        points: np.ndarray,
        mcp_index: int,
        pip_index: int,
        tip_index: int,
    ) -> bool | None:
        angle = self._joint_angle(
            points[mcp_index], points[pip_index], points[tip_index]
        )
        if angle is None:
            return None
        if angle >= self.EXTENDED_ANGLE_DEGREES:
            return True
        if angle <= self.CURLED_ANGLE_DEGREES:
            return False
        return None

    @staticmethod
    def _unit(vector: np.ndarray) -> np.ndarray:
        length = float(np.linalg.norm(vector))
        if length < 1e-6:
            return np.zeros_like(vector)
        return vector / length

    @staticmethod
    def _joint_angle(
        first: np.ndarray,
        joint: np.ndarray,
        last: np.ndarray,
    ) -> float | None:
        first_vector = first - joint
        last_vector = last - joint
        denominator = float(np.linalg.norm(first_vector) * np.linalg.norm(last_vector))
        if denominator < 1e-6:
            return None
        cosine = float(np.dot(first_vector, last_vector) / denominator)
        return degrees(acos(float(np.clip(cosine, -1.0, 1.0))))
