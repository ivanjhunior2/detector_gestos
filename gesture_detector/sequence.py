"""Recognize a timed sequence of confirmed gesture keys."""

from dataclasses import dataclass, field


SEQUENCE = ("q", "w", "e")


@dataclass
class GestureSequence:
    """Emit one completion for Q → W → E within the configured time limit."""

    timeout_seconds: float = 4.0
    _matched: int = field(default=0, init=False)
    _deadline: float | None = field(default=None, init=False)

    @property
    def progress(self) -> tuple[str, ...]:
        return SEQUENCE[:self._matched]

    def update(self, key: str | None, now: float) -> bool:
        """Consume a confirmed key or check for expiration using monotonic time."""
        if self._deadline is not None and now > self._deadline:
            self.reset()

        if key is None:
            return False
        if key == SEQUENCE[0]:
            self._matched = 1
            self._deadline = now + self.timeout_seconds
            return False
        if self._matched == 0 or key != SEQUENCE[self._matched]:
            self.reset()
            return False

        self._matched += 1
        if self._matched == len(SEQUENCE):
            self.reset()
            return True
        return False

    def reset(self) -> None:
        self._matched = 0
        self._deadline = None
