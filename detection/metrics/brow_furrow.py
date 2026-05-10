"""
Brow Furrow Index (BFI) — Geometric proxy for AU4 (Brow Lowerer).

Measures the degree of eyebrow furrowing by combining:
  - IED: inter-eyebrow distance (horizontal convergence)
  - BED: brow-to-eyelid distance (vertical lowering)

Both are normalized against stable face dimensions to be
scale-invariant and not require individual baseline calibration.
"""

import numpy as np

from detection.landmarks import (
    BROW_INNER_LEFT,
    BROW_INNER_RIGHT,
    BROW_MID_LEFT,
    BROW_MID_RIGHT,
    EYELID_TOP_LEFT,
    EYELID_TOP_RIGHT,
    FACE_LEFT_CONTOUR,
    FACE_RIGHT_CONTOUR,
    FACE_TOP,
    FACE_BOTTOM,
)
from detection.config import StressConfig


def _dist(p1: np.ndarray, p2: np.ndarray) -> float:
    """Euclidean distance between two 2D points."""
    return float(np.linalg.norm(p1 - p2))


def _get_face_dimensions(landmarks: np.ndarray) -> tuple[float, float]:
    """
    Compute face width and height from contour landmarks.

    Returns:
        (face_width, face_height)
    """
    face_width = _dist(
        landmarks[FACE_LEFT_CONTOUR], landmarks[FACE_RIGHT_CONTOUR]
    )
    face_height = _dist(landmarks[FACE_TOP], landmarks[FACE_BOTTOM])
    return face_width, face_height


def compute_bfi(
    landmarks: np.ndarray, config: StressConfig
) -> dict:
    """
    Compute the Brow Furrow Index from facial landmarks.

    Args:
        landmarks: (478, 2) array of (x, y) landmark coordinates.
        config: StressConfig with BFI parameters.

    Returns:
        Dictionary with:
          - "value": BFI score ∈ [0, 1]
          - "components": {"ied_normalized", "bed_normalized"}
    """
    face_width, face_height = _get_face_dimensions(landmarks)

    # Guard against degenerate detections
    if face_width < 1e-6 or face_height < 1e-6:
        return {"value": 0.0, "components": {"ied_normalized": 0.0, "bed_normalized": 0.0}}

    # ── IED: Inter-Eyebrow Distance ──────────────────────────
    ied = _dist(landmarks[BROW_INNER_LEFT], landmarks[BROW_INNER_RIGHT])
    ied_ratio = ied / face_width  # Typically 0.15–0.25 at rest

    # Normalize to [0, 1]: 0 = resting, 1 = maximum furrow
    # When furrow increases, IED decreases → invert
    ied_norm = np.clip(
        (config.bfi_ied_rest_max - ied_ratio)
        / (config.bfi_ied_rest_max - config.bfi_ied_rest_min),
        0.0,
        1.0,
    )

    # ── BED: Brow-to-Eyelid Distance ─────────────────────────
    bed_left = _dist(landmarks[BROW_MID_LEFT], landmarks[EYELID_TOP_LEFT])
    bed_right = _dist(landmarks[BROW_MID_RIGHT], landmarks[EYELID_TOP_RIGHT])
    bed_mean = (bed_left + bed_right) / 2.0
    bed_ratio = bed_mean / face_height  # Typically 0.04–0.10 at rest

    # When furrow increases, BED decreases → invert
    bed_norm = np.clip(
        (config.bfi_bed_rest_max - bed_ratio)
        / (config.bfi_bed_rest_max - config.bfi_bed_rest_min),
        0.0,
        1.0,
    )

    # ── Composite BFI ─────────────────────────────────────────
    bfi = config.bfi_alpha * ied_norm + config.bfi_beta * bed_norm
    bfi = float(np.clip(bfi, 0.0, 1.0))

    return {
        "value": bfi,
        "components": {
            "ied_normalized": round(float(ied_norm), 4),
            "bed_normalized": round(float(bed_norm), 4),
        },
    }
