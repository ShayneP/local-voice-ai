"""Tests for STT degradation tracking without importing the ML runtime."""

from __future__ import annotations

import pytest

from local_voice_ai.services.nemotron.degradation import DegradationTracker


class CudaOutOfMemoryError(RuntimeError):
    """Lightweight stand-in for the injected Torch exception type."""


@pytest.fixture
def tracker() -> DegradationTracker:
    return DegradationTracker(fatal_error_types=(CudaOutOfMemoryError,))


class TestFatalClassification:
    @pytest.mark.parametrize(
        "message",
        [
            "CUDA out of memory. Tried to allocate 14.00 MiB",
            "'NoneType' object has no attribute 'replay'",
            "CUDA error: an illegal memory access was encountered",
            "cuBLAS error",
        ],
    )
    def test_fatal_markers(
        self,
        tracker: DegradationTracker,
        message: str,
    ) -> None:
        assert tracker.is_fatal(RuntimeError(message)) is True

    @pytest.mark.parametrize(
        "message",
        [
            "Failed to process audio: unknown format",
            "Empty audio file",
            "invalid sample rate",
        ],
    )
    def test_ordinary_errors_are_not_fatal(
        self,
        tracker: DegradationTracker,
        message: str,
    ) -> None:
        assert tracker.is_fatal(ValueError(message)) is False

    def test_injected_cuda_oom_type_is_fatal(
        self,
        tracker: DegradationTracker,
    ) -> None:
        assert tracker.is_fatal(CudaOutOfMemoryError("boom")) is True


class TestDegradation:
    def test_single_fatal_error_degrades_immediately(
        self,
        tracker: DegradationTracker,
    ) -> None:
        tracker.note_failure(RuntimeError("CUDA out of memory"))

        assert tracker.degraded is not None

    def test_single_ordinary_error_does_not_degrade(
        self,
        tracker: DegradationTracker,
    ) -> None:
        tracker.note_failure(ValueError("bad audio"))

        assert tracker.degraded is None

    def test_repeated_ordinary_errors_degrade(
        self,
        tracker: DegradationTracker,
    ) -> None:
        for _ in range(tracker.max_consecutive_failures):
            tracker.note_failure(ValueError("bad audio"))

        assert tracker.degraded is not None

    def test_success_resets_the_run(self, tracker: DegradationTracker) -> None:
        for _ in range(tracker.max_consecutive_failures - 1):
            tracker.note_failure(ValueError("bad audio"))
            tracker.note_success()

        assert tracker.degraded is None
        assert tracker.consecutive_failures == 0

    def test_degradation_is_sticky(self, tracker: DegradationTracker) -> None:
        tracker.note_failure(RuntimeError("CUDA out of memory"))
        first = tracker.degraded

        tracker.note_success()

        assert tracker.degraded == first
