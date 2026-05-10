"""
Brow Asymmetry + Dynamics (BAD) — Composite metric for stress.

Captures two signals that the static BFI alone misses:
  1. Left/right brow asymmetry: involuntary micro-asymmetries
     increase under stress (genuine expressions are more symmetric).
  2. Temporal variance of BFI: stress produces intermittent furrowing
     (fluctuations), while concentration is typically steady.
"""

from collections import deque

import numpy as np

from detection.landmarks import (
    BROW_INNER_LEFT,
    BROW_INNER_RIGHT,
    BROW_MID_LEFT,
    BROW_MID_RIGHT,
    EYELID_TOP_LEFT,
    EYELID_TOP_RIGHT,
    FACE_TOP,
    FACE_BOTTOM,
)
from detection.config import StressConfig


def _dist(p1: np.ndarray, p2: np.ndarray) -> float:
    """Euclidean distance between two 2D points."""
    return float(np.linalg.norm(p1 - p2))


def compute_brow_asymmetry(landmarks: np.ndarray, face_height: float) -> float:
    """
    Compute instantaneous brow asymmetry.

    Measures the absolute difference between left and right
    brow-to-eyelid distances, normalized by face height.

    Args:
        landmarks: (478, 2) array.
        face_height: Distance from forehead to chin.

    Returns:
        Asymmetry value (float, >= 0). Typical resting: 0.00–0.03.
    """
    if face_height < 1e-6:
        return 0.0

    bed_left = _dist(landmarks[BROW_MID_LEFT], landmarks[EYELID_TOP_LEFT])
    bed_right = _dist(landmarks[BROW_MID_RIGHT], landmarks[EYELID_TOP_RIGHT])

    # Also include inner brow vertical positions
    # (captures AU4 asymmetry — one brow lowered more than the other)
    inner_left_y = landmarks[BROW_INNER_LEFT][1]
    inner_right_y = landmarks[BROW_INNER_RIGHT][1]
    inner_diff = abs(inner_left_y - inner_right_y) / face_height

    bed_diff = abs(bed_left - bed_right) / face_height

    # Combined asymmetry
    return float((bed_diff + inner_diff) / 2.0)


class BrowDynamicsTracker:
    """
    Tracks temporal dynamics of BFI for the BAD metric.

    Maintains a sliding window of recent BFI values to compute:
      - BFI variance: how much the furrowing fluctuates
      - Mean asymmetry: average asymmetry over the window
    """

    def __init__(self, config: StressConfig):
        self.config = config
        window_frames = max(int(config.bad_window_seconds * config.fps), 10)
        self._bfi_history: deque[float] = deque(maxlen=window_frames)
        self._asymmetry_history: deque[float] = deque(maxlen=window_frames)

    def update(self, bfi_value: float, asymmetry: float) -> None:
        """Record a new BFI value and asymmetry measurement."""
        self._bfi_history.append(bfi_value)
        self._asymmetry_history.append(asymmetry)

    def get_bfi_variance(self) -> float:
        """Variance of BFI over the sliding window."""
        if len(self._bfi_history) < 3:
            return 0.0
        return float(np.var(self._bfi_history))

    def get_mean_asymmetry(self) -> float:
        """Mean asymmetry over the sliding window."""
        if not self._asymmetry_history:
            return 0.0
        return float(np.mean(self._asymmetry_history))


def compute_bad(
    asymmetry: float,
    bfi_variance: float,
    config: StressConfig,
) -> float:
    """
    Compute the BAD (Brow Asymmetry + Dynamics) stress score.

    Args:
        asymmetry: Mean brow asymmetry (from BrowDynamicsTracker).
        bfi_variance: Variance of BFI (from BrowDynamicsTracker).
        config: StressConfig.

    Returns:
        BAD score ∈ [0, 1].
    """
    # Normalize asymmetry to [0, 1]
    if asymmetry <= config.bad_asymmetry_rest_max:
        asym_score = 0.0
    elif asymmetry >= config.bad_asymmetry_stress:
        asym_score = 1.0
    else:
        asym_score = (asymmetry - config.bad_asymmetry_rest_max) / (
            config.bad_asymmetry_stress - config.bad_asymmetry_rest_max
        )

    # Normalize BFI variance to [0, 1]
    if bfi_variance <= config.bad_variance_rest_max:
        var_score = 0.0
    elif bfi_variance >= config.bad_variance_stress:
        var_score = 1.0
    else:
        var_score = (bfi_variance - config.bad_variance_rest_max) / (
            config.bad_variance_stress - config.bad_variance_rest_max
        )

    # Composite
    bad = config.bad_gamma * asym_score + config.bad_delta * var_score
    return float(np.clip(bad, 0.0, 1.0))
