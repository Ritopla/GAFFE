"""
Eye Aspect Ratio + Blink Dynamics (EAR/BD) — Proxy for AU7/AU43.

Computes:
  - EAR: instantaneous eye openness (Soukupová & Čech, 2016)
  - Blink detection: frame-level blink state machine
  - Blink rate: blinks per minute over a sliding window
  - Blink duration: average duration of recent blinks
  - EAR stress score: composite of lid tightening + blink rate + blink speed
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from detection.landmarks import LEFT_EYE_INDICES, RIGHT_EYE_INDICES
from detection.config import StressConfig


def _dist(p1: np.ndarray, p2: np.ndarray) -> float:
    """Euclidean distance between two 2D points."""
    return float(np.linalg.norm(p1 - p2))


def compute_ear(landmarks: np.ndarray) -> float:
    """
    Compute the Eye Aspect Ratio averaged over both eyes.

    Formula (per eye):
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 · ||p1 - p4||)

    Where p1..p6 are the 6 eye contour landmarks in order:
        p1=outer, p2=upper-lat, p3=upper-med,
        p4=inner, p5=lower-med, p6=lower-lat

    Args:
        landmarks: (478, 2) array of (x, y) landmark coordinates.

    Returns:
        Averaged EAR value (float). Typical range: 0.0–0.40
    """
    ear_values = []
    for indices in (LEFT_EYE_INDICES, RIGHT_EYE_INDICES):
        p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in indices]

        vertical_1 = _dist(p2, p6)
        vertical_2 = _dist(p3, p5)
        horizontal = _dist(p1, p4)

        if horizontal < 1e-6:
            ear_values.append(0.0)
            continue

        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        ear_values.append(ear)

    return float(np.mean(ear_values))


# ──────────────────────────────────────────────────────────────
# Blink Detector — state machine for blink event tracking
# ──────────────────────────────────────────────────────────────


@dataclass
class BlinkEvent:
    """A single detected blink."""
    start_frame: int
    end_frame: int
    duration_frames: int

    def duration_ms(self, fps: float) -> float:
        """Duration in milliseconds."""
        return (self.duration_frames / fps) * 1000.0 if fps > 0 else 0.0


class BlinkDetector:
    """
    Detects blink events from a stream of EAR values.

    Uses a simple state machine:
      OPEN → (EAR < threshold) → CLOSING → (EAR >= threshold) → register blink → OPEN
    """

    def __init__(self, config: StressConfig):
        self.config = config
        self._in_blink = False
        self._blink_start_frame: int = 0
        self._current_frame: int = 0

        # Sliding window of recent blink events
        max_blinks = int(config.blink_window_seconds * 10)  # generous upper bound
        self._blink_history: deque[BlinkEvent] = deque(maxlen=max(max_blinks, 100))

    def update(self, ear: float, frame_id: int) -> bool:
        """
        Process a new EAR value and return whether the current frame is a blink.

        Args:
            ear: Current Eye Aspect Ratio.
            frame_id: Current frame number.

        Returns:
            True if this frame is part of a blink event.
        """
        self._current_frame = frame_id
        is_blink_frame = False

        if not self._in_blink:
            if ear < self.config.ear_blink_threshold:
                # Transition: OPEN → CLOSING
                self._in_blink = True
                self._blink_start_frame = frame_id
                is_blink_frame = True
        else:
            if ear >= self.config.ear_blink_threshold:
                # Transition: CLOSING → OPEN (blink completed)
                duration = frame_id - self._blink_start_frame
                if duration >= 1:  # Ignore single-frame noise
                    event = BlinkEvent(
                        start_frame=self._blink_start_frame,
                        end_frame=frame_id,
                        duration_frames=duration,
                    )
                    self._blink_history.append(event)
                self._in_blink = False
            else:
                is_blink_frame = True

        return is_blink_frame

    def _get_recent_blinks(self, frame_id: int) -> list[BlinkEvent]:
        """Get blinks within the sliding window."""
        window_frames = int(self.config.blink_window_seconds * self.config.fps)
        cutoff = frame_id - window_frames
        return [b for b in self._blink_history if b.end_frame >= cutoff]

    def get_blink_rate(self, frame_id: int) -> float:
        """
        Compute blinks per minute over the sliding window.

        Returns 0.0 if not enough data has been collected yet.
        """
        recent = self._get_recent_blinks(frame_id)
        window_frames = int(self.config.blink_window_seconds * self.config.fps)

        # How many frames have we actually observed?
        actual_frames = min(frame_id + 1, window_frames)
        if actual_frames < self.config.fps:  # Need at least 1 second
            return 0.0

        actual_seconds = actual_frames / self.config.fps
        return (len(recent) / actual_seconds) * 60.0

    def get_avg_blink_duration_ms(self, frame_id: int) -> float:
        """Average blink duration (ms) over the sliding window."""
        recent = self._get_recent_blinks(frame_id)
        if not recent:
            return 0.0
        durations = [b.duration_ms(self.config.fps) for b in recent]
        return float(np.mean(durations))


# ──────────────────────────────────────────────────────────────
# EAR Stress Score — composite of lid tightening + blink dynamics
# ──────────────────────────────────────────────────────────────


def compute_ear_stress(
    ear: float,
    blink_rate: float,
    avg_blink_duration_ms: float,
    config: StressConfig,
) -> float:
    """
    Compute a composite stress score from eye metrics.

    Components (all ∈ [0, 1]):
      1. Lid tightening: how much EAR is below baseline (AU7)
      2. Blink rate: deviation above population baseline
      3. Blink speed: how fast blinks are (faster = more stress)

    Args:
        ear: Current Eye Aspect Ratio (smoothed).
        blink_rate: Blinks per minute.
        avg_blink_duration_ms: Average blink duration in ms.
        config: StressConfig.

    Returns:
        EAR stress score ∈ [0, 1].
    """
    # 1) Lid tightening score
    # Low EAR → high score (AU7 active)
    if ear >= config.ear_open_baseline:
        lid_score = 0.0
    elif ear <= config.ear_blink_threshold:
        lid_score = 1.0
    else:
        lid_score = (config.ear_open_baseline - ear) / (
            config.ear_open_baseline - config.ear_blink_threshold
        )

    # 2) Blink rate score
    # Higher than baseline → stress
    if blink_rate <= config.blink_rate_baseline:
        rate_score = 0.0
    elif blink_rate >= config.blink_rate_stress:
        rate_score = 1.0
    else:
        rate_score = (blink_rate - config.blink_rate_baseline) / (
            config.blink_rate_stress - config.blink_rate_baseline
        )

    # 3) Blink speed score
    # Faster blinks → higher stress score
    if avg_blink_duration_ms <= 0.0:
        speed_score = 0.0  # Not enough data
    elif avg_blink_duration_ms <= config.blink_duration_fast_ms:
        speed_score = 1.0  # Very fast → stress
    elif avg_blink_duration_ms >= config.blink_duration_slow_ms:
        speed_score = 0.0  # Slow → fatigue, not stress
    else:
        # Linear interpolation: fast=1.0, slow=0.0
        speed_score = (config.blink_duration_slow_ms - avg_blink_duration_ms) / (
            config.blink_duration_slow_ms - config.blink_duration_fast_ms
        )

    # Composite
    ear_stress = (
        config.ear_sub_weight_level * lid_score
        + config.ear_sub_weight_rate * rate_score
        + config.ear_sub_weight_duration * speed_score
    )
    return float(np.clip(ear_stress, 0.0, 1.0))
