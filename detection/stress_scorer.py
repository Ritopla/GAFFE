"""
StressScorer — Main orchestrator for frame-by-frame stress analysis.

Processes a stream of landmark arrays and produces a structured result
per frame, including all intermediate metrics and the composite stress score.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from detection.config import StressConfig
from detection.landmarks import FACE_TOP, FACE_BOTTOM
from detection.metrics.brow_furrow import compute_bfi
from detection.metrics.eye_aspect import (
    compute_ear,
    compute_ear_stress,
    BlinkDetector,
)
from detection.metrics.brow_asymmetry import (
    compute_brow_asymmetry,
    compute_bad,
    BrowDynamicsTracker,
)


def _dist(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1 - p2))


class SmootherBuffer:
    """Simple moving-average smoother for a scalar signal."""

    def __init__(self, window: int):
        self._window = max(window, 1)
        self._buffer: deque[float] = deque(maxlen=self._window)

    def push(self, value: float) -> float:
        """Push a value and return the smoothed output."""
        self._buffer.append(value)
        return float(np.mean(self._buffer))


class StressScorer:
    """
    Main pipeline class. Stateful: maintains temporal trackers
    for blink detection and brow dynamics.

    Usage:
        scorer = StressScorer(config)
        for frame_id, landmarks in enumerate(landmark_stream):
            result = scorer.process_frame(landmarks, frame_id)
            # result is a dict ready for JSON serialization
    """

    def __init__(self, config: StressConfig | None = None):
        self.config = config or StressConfig()
        self._blink_detector = BlinkDetector(self.config)
        self._brow_tracker = BrowDynamicsTracker(self.config)

        # Smoothers for per-frame jitter reduction
        w = self.config.smoothing_window_frames
        self._bfi_smoother = SmootherBuffer(w)
        self._ear_smoother = SmootherBuffer(w)

    def process_frame(
        self,
        landmarks: np.ndarray,
        frame_id: int,
        timestamp_ms: float | None = None,
    ) -> dict[str, Any]:
        """
        Analyze a single frame's landmarks and return the full stress report.

        Args:
            landmarks: (478, 2) or (478, 3) array of landmark coordinates.
                       If 3D, only x and y are used.
            frame_id: Sequential frame number (0-indexed).
            timestamp_ms: Optional timestamp in milliseconds.

        Returns:
            Dictionary with all metrics and composite stress score.
        """
        # Ensure 2D
        lm = np.asarray(landmarks, dtype=np.float64)
        if lm.ndim == 2 and lm.shape[1] > 2:
            lm = lm[:, :2]

        # Compute face height (used by multiple metrics)
        face_height = _dist(lm[FACE_TOP], lm[FACE_BOTTOM])

        # ── 1. BFI ────────────────────────────────────────────
        bfi_result = compute_bfi(lm, self.config)
        bfi_raw = bfi_result["value"]
        bfi_smooth = self._bfi_smoother.push(bfi_raw)

        # ── 2. EAR + Blink Dynamics ───────────────────────────
        ear_raw = compute_ear(lm)
        ear_smooth = self._ear_smoother.push(ear_raw)

        is_blink = self._blink_detector.update(ear_smooth, frame_id)
        blink_rate = self._blink_detector.get_blink_rate(frame_id)
        avg_blink_dur = self._blink_detector.get_avg_blink_duration_ms(frame_id)

        ear_stress_score = compute_ear_stress(
            ear_smooth, blink_rate, avg_blink_dur, self.config
        )

        # ── 3. BAD (Brow Asymmetry + Dynamics) ────────────────
        asymmetry = compute_brow_asymmetry(lm, face_height)
        self._brow_tracker.update(bfi_smooth, asymmetry)

        bfi_variance = self._brow_tracker.get_bfi_variance()
        mean_asymmetry = self._brow_tracker.get_mean_asymmetry()

        bad_score = compute_bad(mean_asymmetry, bfi_variance, self.config)

        # ── Composite Stress Score ────────────────────────────
        stress_score = (
            self.config.weight_bfi * bfi_smooth
            + self.config.weight_ear * ear_stress_score
            + self.config.weight_bad * bad_score
        )
        stress_score = float(np.clip(stress_score, 0.0, 1.0))
        stress_level = self.config.classify_stress(stress_score)

        # ── Build output ──────────────────────────────────────
        result: dict[str, Any] = {
            "frame_id": frame_id,
            "face_detected": True,
            "metrics": {
                "bfi": {
                    "value": round(bfi_smooth, 4),
                    "components": bfi_result["components"],
                },
                "ear": {
                    "value": round(ear_smooth, 4),
                    "is_blink": is_blink,
                    "blink_rate_per_min": round(blink_rate, 2),
                    "avg_blink_duration_ms": round(avg_blink_dur, 1),
                    "ear_stress_score": round(ear_stress_score, 4),
                },
                "bad": {
                    "value": round(bad_score, 4),
                    "components": {
                        "asymmetry": round(mean_asymmetry, 4),
                        "bfi_variance": round(bfi_variance, 6),
                    },
                },
            },
            "stress": {
                "score": round(stress_score, 4),
                "level": stress_level,
                "weights_used": {
                    "bfi": self.config.weight_bfi,
                    "ear": self.config.weight_ear,
                    "bad": self.config.weight_bad,
                },
            },
        }

        if timestamp_ms is not None:
            result["timestamp_ms"] = round(timestamp_ms, 1)

        return result

    @staticmethod
    def no_face_result(
        frame_id: int, timestamp_ms: float | None = None
    ) -> dict[str, Any]:
        """Return a placeholder result when no face is detected."""
        result: dict[str, Any] = {
            "frame_id": frame_id,
            "face_detected": False,
            "metrics": None,
            "stress": None,
        }
        if timestamp_ms is not None:
            result["timestamp_ms"] = round(timestamp_ms, 1)
        return result
