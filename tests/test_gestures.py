import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from gesture_detector.config import AppConfig, GESTURE_TO_KEY
from gesture_detector.event_gate import GestureEventGate
from gesture_detector.gesture_rules import GestureRuleRecognizer, HandPose


def hand(center, wrist, axis, extended, index, middle, direction):
    tips = np.array([index, middle, middle, middle], dtype=float)
    return HandPose(
        np.array(center), np.array(wrist), np.array(axis), 10.0, extended,
        tips, tips[0], tips[1], np.array(direction),
    )


def reflect(pose):
    reflection = np.array([-1, 1])
    return replace(pose, **{
        name: getattr(pose, name) * reflection
        for name in ("center", "wrist", "palm_axis", "finger_tips",
                     "index_tip", "middle_tip", "index_direction")
    })


FOLDED = (False,) * 4
INDEX = (True, False, False, False)
TWO_FINGERS = (True, True, False, False)
FIXTURES = {
    "bode": (
        hand((-3, 0), (-3, 9), (0, -1), INDEX, (-3, -14), (-3, -4), (0, -1)),
        hand((3, 2), (6, 10), (0, -1), FOLDED, (-2, 1), (3, 3), (-1, 0)),
    ),
    "javali": (
        hand((-4, 0), (-4, -10), (0, 1), FOLDED, (-3, 5), (-4, 5), (0, 1)),
        hand((4, 0), (4, -10), (0, 1), FOLDED, (3, 5), (4, 5), (0, 1)),
    ),
    "cachorro": (
        hand((-3, -4), (-13, -4), (1, 0), (True,) * 4, (8, -4), (8, -3), (1, 0)),
        hand((3, 4), (12, 4), (-1, 0), FOLDED, (1, 5), (2, 5), (0, 1)),
    ),
    "cavalo": (
        hand((-5, 0), (-10, 8), (.6, -.8), INDEX, (-1, -8), (-4, -1), (.6, -.8)),
        hand((5, 0), (10, 8), (-.6, -.8), INDEX, (1, -8), (4, -1), (-.6, -.8)),
    ),
    "passaro": (
        hand((-5, 0), (-10, 8), (.6, -.8), TWO_FINGERS, (-1, -8), (-2, -7), (.6, -.8)),
        hand((5, 0), (10, 8), (-.6, -.8), TWO_FINGERS, (1, -8), (2, -7), (-.6, -.8)),
    ),
}


class GestureEvaluationTest(unittest.TestCase):
    def test_open_palm_requires_one_hand_and_an_open_thumb_on_either_side(self):
        recognizer = GestureRuleRecognizer()
        coordinates = [(50, 75), (43, 67), (37, 62), (29, 54), (21, 46)]
        for x in (40, 50, 58, 66):
            coordinates.extend((x, y) for y in (50, 42, 36, 30))
        self.assertEqual(GESTURE_TO_KEY["palma_abierta"], "r")
        self.assertEqual(GESTURE_TO_KEY["cachorro"], "w")
        for mirrored in (False, True):
            points = [SimpleNamespace(x=(100 - x if mirrored else x) / 100, y=y / 100)
                      for x, y in coordinates]
            folded_thumb = list(points)
            folded_thumb[4] = points[5]
            self.assertEqual(recognizer.classify(SimpleNamespace(hand_landmarks=[points]), (100, 100)), "palma_abierta")
            self.assertIsNone(recognizer.classify(SimpleNamespace(hand_landmarks=[folded_thumb]), (100, 100)))
            self.assertIsNone(recognizer.classify(SimpleNamespace(hand_landmarks=[points, points]), (100, 100)))

    def test_open_palm_tolerates_a_slightly_bent_finger_but_not_a_folded_one(self):
        coordinates = [(50, 75), (43, 67), (37, 62), (28, 56), (33, 43)]
        for x in (40, 50, 58, 66):
            coordinates.extend((x, y) for y in (50, 42, 36, 30))
        coordinates[20] = (70, 43)
        recognizer = GestureRuleRecognizer()

        def classify(points):
            landmarks = [SimpleNamespace(x=x / 100, y=y / 100) for x, y in points]
            return recognizer.classify(SimpleNamespace(hand_landmarks=[landmarks]), (100, 100))

        self.assertEqual(classify(coordinates), "palma_abierta")
        coordinates[20] = (66, 53)
        self.assertIsNone(classify(coordinates))

    def test_bode_recognizes_an_elevated_tip_with_uncertain_finger_angle(self):
        raised, support = FIXTURES["bode"]
        raised = replace(raised, extended=(False, None, False, None), index_direction=np.array([.8, -.6]))
        recognizer = GestureRuleRecognizer()
        with patch.object(recognizer, "_measure_pose", side_effect=lambda pose, *_: pose):
            for pair in ((raised, support), (reflect(raised), reflect(support))):
                result = SimpleNamespace(hand_landmarks=pair)
                self.assertEqual(recognizer.classify(result, (480, 640, 3)), "bode")

    def test_gate_confirms_a_new_gesture_in_four_frames_and_does_not_repeat(self):
        config = AppConfig()
        gate = GestureEventGate(config.gesture_stable_frames, config.gesture_release_frames)
        self.assertEqual([gate.update("bode") for _ in range(5)], [None, None, None, "bode", None])
        self.assertEqual([gate.update("palma_abierta") for _ in range(5)], [None, None, None, "palma_abierta", None])

    def test_each_seal_in_both_orientations_and_input_orders(self):
        recognizer = GestureRuleRecognizer()
        config = AppConfig()
        self.assertEqual(set(FIXTURES), set(GESTURE_TO_KEY) - {"palma_abierta"})
        with patch.object(recognizer, "_measure_pose", side_effect=lambda pose, *_: pose):
            for expected, hands in FIXTURES.items():
                for mirrored in (False, True):
                    pair = tuple(map(reflect, hands)) if mirrored else hands
                    for ordered in (pair, pair[::-1]):
                        with self.subTest(gesture=expected, mirrored=mirrored, reversed=ordered is not pair):
                            gate = GestureEventGate(config.gesture_stable_frames, config.gesture_release_frames)
                            result = SimpleNamespace(hand_landmarks=ordered)
                            label = recognizer.classify(result, (480, 640, 3))
                            self.assertEqual(label, expected)
                            events = [gate.update(label) for _ in range(2 * config.gesture_stable_frames)]
                            self.assertEqual([event for event in events if event], [expected])

    def test_other_seals_tolerate_geometry_or_finger_variation(self):
        upper, lower = FIXTURES["cachorro"]
        approximate = {
            "cachorro": (replace(upper, palm_axis=np.array([.64, -.77]),
                                 extended=(True, None, True, None)), lower),
            "javali": tuple(replace(pose, palm_axis=np.array([.9, .42]),
                                    extended=(False, False, None, None)) for pose in FIXTURES["javali"]),
            "cavalo": (FIXTURES["cavalo"][0], replace(FIXTURES["cavalo"][1],
                                                     extended=(True, None, False, False))),
            "passaro": (FIXTURES["passaro"][0], replace(FIXTURES["passaro"][1],
                                                       extended=(True, None, False, False))),
        }
        recognizer = GestureRuleRecognizer()
        with patch.object(recognizer, "_measure_pose", side_effect=lambda pose, *_: pose):
            for expected, pair in approximate.items():
                for hands in (pair, tuple(map(reflect, pair))):
                    with self.subTest(gesture=expected):
                        result = SimpleNamespace(hand_landmarks=hands)
                        self.assertEqual(recognizer.classify(result, (480, 640, 3)), expected)

    def test_unclear_middle_fingers_do_not_choose_between_cavalo_and_passaro(self):
        recognizer = GestureRuleRecognizer()
        hands = [replace(pose, extended=(True, None, False, False)) for pose in FIXTURES["passaro"]]
        with patch.object(recognizer, "_measure_pose", side_effect=lambda pose, *_: pose):
            self.assertIsNone(recognizer.classify(SimpleNamespace(hand_landmarks=hands), (480, 640, 3)))

    def test_each_seal_rejects_either_hand_on_its_own(self):
        recognizer = GestureRuleRecognizer()
        with patch.object(recognizer, "_measure_pose", side_effect=lambda pose, *_: pose):
            for expected, hands in FIXTURES.items():
                for index, pose in enumerate(hands):
                    with self.subTest(gesture=expected, hand=index):
                        result = SimpleNamespace(hand_landmarks=[pose])
                        self.assertIsNone(recognizer.classify(result, (480, 640, 3)))
