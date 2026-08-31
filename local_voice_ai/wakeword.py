"""Wake word gating for the voice agent.

When WAKE_WORD=1, the agent joins the room with its audio input disabled and
runs livekit-wakeword (a small ONNX classifier over frozen speech embeddings)
on the participant's microphone. Only after the wake phrase is heard does the
agent start listening — the LiveKit-blessed edge pattern from
https://github.com/livekit-examples/hello-wakeword, folded into the agent
since our "edge device" is a browser tab.

The buffering mirrors the reference listener: a rolling ~2 s window at 16 kHz,
scored every 80 ms hop. Detection scores are logged at DEBUG so the threshold
can be calibrated against a real microphone (WAKE_WORD_THRESHOLD).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from livekit import rtc

logger = logging.getLogger("wakeword")

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 2 * SAMPLE_RATE  # the classifier consumes ~2 s windows
HOP_SAMPLES = 1280  # 80 ms between scores, as in the reference listener


class SlidingWindow:
    """Rolling fixed-size audio window; signals when a new hop is ready.

    ``push`` returns True when at least HOP_SAMPLES of new audio arrived since
    the last score AND the window has filled once — the moments ``window``
    should be scored.
    """

    def __init__(self) -> None:
        self._buf = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        self._since_hop = 0
        self._filled = 0

    def push(self, samples: np.ndarray) -> bool:
        n = len(samples)
        if n == 0:
            return False
        if n >= WINDOW_SAMPLES:
            self._buf[:] = samples[-WINDOW_SAMPLES:]
        else:
            self._buf[:-n] = self._buf[n:]
            self._buf[-n:] = samples
        self._filled = min(self._filled + n, WINDOW_SAMPLES)
        self._since_hop += n
        if self._since_hop >= HOP_SAMPLES and self._filled >= WINDOW_SAMPLES:
            self._since_hop = 0
            return True
        return False

    @property
    def window(self) -> np.ndarray:
        return self._buf


async def wait_for_wake_word(
    participant: rtc.Participant,
    model_path: str,
    threshold: float,
) -> float:
    """Block until the wake word is heard on ``participant``'s microphone.

    Returns the detection score. Raises if the model can't be loaded — the
    caller decides whether to fail open (start listening) or crash.
    """
    from livekit import rtc  # deferred: heavy import, agent-process only
    from livekit.wakeword import WakeWordModel

    model = WakeWordModel(models=[model_path])
    window = SlidingWindow()

    stream = rtc.AudioStream.from_participant(
        participant=participant,
        track_source=rtc.TrackSource.SOURCE_MICROPHONE,
        sample_rate=SAMPLE_RATE,
        num_channels=1,
    )
    logger.info("listening for wake word (threshold=%.2f)", threshold)
    try:
        async for event in stream:
            samples = np.frombuffer(event.frame.data, dtype=np.int16)
            if not window.push(samples.astype(np.float32) / 32768.0):
                continue
            scores = model.predict(window.window)
            score = max(scores.values()) if scores else 0.0
            if score > 0.05:  # only log when something phrase-like is happening
                logger.debug("wake word score: %.3f", score)
            if score >= threshold:
                logger.info("wake word detected (score=%.3f)", score)
                return score
    finally:
        await stream.aclose()
    raise RuntimeError("audio stream ended before wake word was detected")
