import unittest

from gesture_detector.config import GESTURE_TO_KEY
from gesture_detector.event_gate import GestureEventGate
from gesture_detector.sequence import GestureSequence


class GestureSequenceTest(unittest.TestCase):
    def test_completes_once_and_can_be_repeated(self):
        sequence = GestureSequence()
        for start in (0.0, 5.0):
            self.assertFalse(sequence.update("q", start))
            self.assertEqual(sequence.progress, ("q",))
            self.assertFalse(sequence.update(None, start + 0.5))
            self.assertFalse(sequence.update("w", start + 1))
            self.assertEqual(sequence.progress, ("q", "w"))
            self.assertTrue(sequence.update("e", start + 4))
            self.assertEqual(sequence.progress, ())
            self.assertFalse(sequence.update("e", start + 4.1))

    def test_out_of_order_input_cancels_and_q_restarts(self):
        sequence = GestureSequence()
        for key, now in (("q", 0), ("e", 1), ("w", 2), ("e", 3)):
            self.assertFalse(sequence.update(key, now))
        self.assertEqual(sequence.progress, ())
        for key, now in (("q", 5), ("w", 6), ("q", 7), ("w", 8)):
            self.assertFalse(sequence.update(key, now))
        self.assertTrue(sequence.update("e", 10))

    def test_expiration_on_idle_frame_or_late_key(self):
        for key in (None, "e"):
            with self.subTest(key=key):
                sequence = GestureSequence()
                sequence.update("q", 0)
                sequence.update("w", 2)
                self.assertFalse(sequence.update(key, 4.01))
                self.assertEqual(sequence.progress, ())

    def test_held_poses_produce_one_completion(self):
        gate = GestureEventGate()
        sequence = GestureSequence()
        results = []
        frames = [label for label in ("bode", "cachorro", "javali") for _ in range(8)]
        for frame, label in enumerate(frames):
            confirmed = gate.update(label)
            results.append(sequence.update(GESTURE_TO_KEY.get(confirmed), frame / 30))
        self.assertEqual(sum(results), 1)
