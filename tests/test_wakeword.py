"""Tests for the wake word sliding-window buffering.

The window/hop constants mirror livekit-wakeword's reference listener: a 2 s
rolling window at 16 kHz, scored every 80 ms (1280 samples). The invariants:
no scoring until the window has filled once, then one score per hop.
"""

from __future__ import annotations

import numpy as np

from local_voice_ai.wakeword import HOP_SAMPLES, SAMPLE_RATE, WINDOW_SAMPLES, SlidingWindow


def _frames(total: int, frame: int = 160):
    """Yield int16-scale float frames like rtc.AudioStream (10 ms at 16 kHz)."""
    remaining = total
    while remaining > 0:
        n = min(frame, remaining)
        remaining -= n
        yield np.zeros(n, dtype=np.float32)


class TestSlidingWindow:
    def test_no_score_until_window_full(self) -> None:
        w = SlidingWindow()
        hops = sum(w.push(f) for f in _frames(WINDOW_SAMPLES - 160))
        assert hops == 0

    def test_scores_once_per_hop_after_full(self) -> None:
        w = SlidingWindow()
        for f in _frames(WINDOW_SAMPLES):
            w.push(f)
        # one second of additional audio → SAMPLE_RATE / HOP_SAMPLES hops
        hops = sum(w.push(f) for f in _frames(SAMPLE_RATE))
        assert hops == SAMPLE_RATE // HOP_SAMPLES

    def test_window_holds_newest_audio(self) -> None:
        w = SlidingWindow()
        w.push(np.zeros(WINDOW_SAMPLES, dtype=np.float32))
        marker = np.full(HOP_SAMPLES, 0.5, dtype=np.float32)
        w.push(marker)
        assert (w.window[-HOP_SAMPLES:] == 0.5).all()
        assert (w.window[:-HOP_SAMPLES] == 0.0).all()

    def test_oversized_push_keeps_tail(self) -> None:
        w = SlidingWindow()
        big = np.arange(WINDOW_SAMPLES + 5, dtype=np.float32)
        w.push(big)
        assert (w.window == big[-WINDOW_SAMPLES:]).all()

    def test_empty_push_is_noop(self) -> None:
        w = SlidingWindow()
        assert w.push(np.array([], dtype=np.float32)) is False


class TestModelIntegration:
    def test_real_model_loads_and_scores(self) -> None:
        # livekit-wakeword is a first-class dependency; make sure a window of
        # silence produces a well-formed (empty-model) prediction API shape.
        from livekit.wakeword import WakeWordModel

        model = WakeWordModel(models=[])  # no classifier → empty scores
        assert model.predict(np.zeros(WINDOW_SAMPLES, dtype=np.float32)) == {}
