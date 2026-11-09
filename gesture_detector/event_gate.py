"""One-shot gesture events with stable-frame debounce and release re-arming."""

from dataclasses import dataclass


@dataclass
class GestureEventGate:
    """Require a stable pose; re-arm after release or a different stable pose."""

    stable_frames: int = 4
    release_frames: int = 5
    armed: bool = True
    _candidate: str | None = None
    _candidate_count: int = 0
    _released_count: int = 0
    _release_label: str | None = None
    _last_triggered: str | None = None

    def update(self, label: str | None) -> str | None:
        """Return one event for a stable pose, never repeat while it is held."""
        if label is None:
            self._candidate = None
            self._candidate_count = 0
            if not self.armed:
                self._release_label = None
                self._released_count += 1
                if self._released_count >= self.release_frames:
                    self.armed = True
                    self._released_count = 0
                    self._release_label = None
                    self._last_triggered = None
            return None

        if not self.armed:
            if label == self._last_triggered:
                self._released_count = 0
                self._release_label = None
                return None

            # Require a different pose to persist before re-arming; a one-frame
            # classification wobble must not repeat the previous key.
            if label == self._release_label:
                self._released_count += 1
            else:
                self._release_label = label
                self._released_count = 1

            switch_frames = min(self.release_frames, self.stable_frames)
            if self._released_count < switch_frames:
                return None

            self.armed = True
            self._last_triggered = None
            self._released_count = 0
            self._release_label = None
            self._candidate = label
            self._candidate_count = switch_frames
            if self._candidate_count >= self.stable_frames:
                self.armed = False
                self._last_triggered = label
                self._candidate = None
                self._candidate_count = 0
                return label
            return None

        if label == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = label
            self._candidate_count = 1

        if self._candidate_count >= self.stable_frames:
            self.armed = False
            self._last_triggered = label
            self._candidate = None
            self._candidate_count = 0
            return label
        return None
