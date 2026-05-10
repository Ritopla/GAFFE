"""
Configuration for the GAFFE stress detection pipeline.

All thresholds, weights, and tunable parameters are centralized here
to make the system easy to calibrate without modifying computation logic.
"""

from dataclasses import dataclass, field


@dataclass
class StressConfig:
    """Configuration for stress scoring pipeline."""

    # ── Metric weights (must sum to 1.0) ──────────────────────
    weight_bfi: float = 0.45   # Brow Furrow Index
    weight_ear: float = 0.35   # Eye Aspect Ratio + Blink Dynamics
    weight_bad: float = 0.20   # Brow Asymmetry + Dynamics

    # ── BFI parameters ────────────────────────────────────────
    # Component weights within BFI (alpha + beta = 1.0)
    bfi_alpha: float = 0.5     # Weight for IED (inter-eyebrow distance)
    bfi_beta: float = 0.5      # Weight for BED (brow-eyelid distance)

    # Population-based neutral ranges for normalization
    # IED / face_width ratio: typical resting range
    bfi_ied_rest_min: float = 0.15
    bfi_ied_rest_max: float = 0.25
    # BED / face_height ratio: typical resting range
    bfi_bed_rest_min: float = 0.04
    bfi_bed_rest_max: float = 0.10

    # ── EAR parameters ────────────────────────────────────────
    # Eye Aspect Ratio thresholds
    ear_open_baseline: float = 0.30    # Typical open-eye EAR
    ear_blink_threshold: float = 0.20  # Below this → blink candidate
    ear_closed_threshold: float = 0.05 # Below this → confirmed closed

    # Blink dynamics
    blink_rate_baseline: float = 17.5  # Population mean blinks/min
    blink_rate_stress: float = 25.0    # Above this → stress indicator
    blink_duration_fast_ms: float = 150.0  # Below → rapid blink (stress)
    blink_duration_slow_ms: float = 300.0  # Above → slow blink (fatigue)

    # Sliding window for blink rate (in seconds)
    blink_window_seconds: float = 15.0

    # Sub-weights within EAR stress score
    ear_sub_weight_level: float = 0.35     # Chronic lid tightening
    ear_sub_weight_rate: float = 0.40      # Blink rate
    ear_sub_weight_duration: float = 0.25  # Blink speed

    # ── BAD parameters ────────────────────────────────────────
    # Component weights within BAD (gamma + delta = 1.0)
    bad_gamma: float = 0.4    # Weight for asymmetry
    bad_delta: float = 0.6    # Weight for BFI variance

    # Population thresholds
    bad_asymmetry_rest_max: float = 0.03  # Max normal asymmetry
    bad_asymmetry_stress: float = 0.08    # Asymmetry indicating stress
    bad_variance_rest_max: float = 0.02   # Max normal BFI variance
    bad_variance_stress: float = 0.06     # Variance indicating stress

    # Sliding window for temporal dynamics (in seconds)
    bad_window_seconds: float = 3.0

    # ── Stress classification thresholds ──────────────────────
    stress_low_max: float = 0.25
    stress_moderate_max: float = 0.50
    stress_high_max: float = 0.75
    # Above stress_high_max → VERY_HIGH

    # ── Temporal smoothing ────────────────────────────────────
    # Moving average window for per-frame metric smoothing
    smoothing_window_frames: int = 5

    # ── Video parameters ──────────────────────────────────────
    fps: float = 30.0  # Default, overridden by actual video FPS

    def classify_stress(self, score: float) -> str:
        """Classify a stress score into a human-readable level."""
        if score <= self.stress_low_max:
            return "LOW"
        elif score <= self.stress_moderate_max:
            return "MODERATE"
        elif score <= self.stress_high_max:
            return "HIGH"
        else:
            return "VERY_HIGH"
