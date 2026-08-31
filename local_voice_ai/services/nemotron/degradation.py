"""Track unrecoverable speech-model failures without ML dependencies."""

from __future__ import annotations


class DegradationTracker:
    """Decide when repeated inference errors require a process restart."""

    _FATAL_ERROR_MARKERS = (
        "cuda out of memory",
        "no attribute 'replay'",
        "cuda error",
        "device-side assert",
        "cublas",
        "illegal memory access",
    )

    def __init__(
        self,
        *,
        fatal_error_types: tuple[type[BaseException], ...] = (),
        max_consecutive_failures: int = 3,
    ) -> None:
        self._fatal_error_types = fatal_error_types
        self.max_consecutive_failures = max_consecutive_failures
        self.degraded: str | None = None
        self.consecutive_failures = 0

    def is_fatal(self, error: BaseException) -> bool:
        if isinstance(error, self._fatal_error_types):
            return True
        message = str(error).lower()
        return any(marker in message for marker in self._FATAL_ERROR_MARKERS)

    def note_failure(self, error: BaseException) -> bool:
        """Record an error and return whether the model became degraded."""
        self.consecutive_failures += 1
        if self.degraded is not None:
            return False
        if self.is_fatal(error):
            self.degraded = f"unrecoverable inference error: {error}"
        elif self.consecutive_failures >= self.max_consecutive_failures:
            self.degraded = (
                f"{self.consecutive_failures} consecutive transcription failures; "
                f"last error: {error}"
            )
        else:
            return False
        return True

    def note_success(self) -> None:
        """Clear transient failures without clearing a degraded state."""
        self.consecutive_failures = 0
