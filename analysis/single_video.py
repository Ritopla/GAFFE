"""
Single-video stress analysis from GAFFE JSON output.

Reads a _gaffe.json file produced by demo.py and computes:
  - Summary statistics (mean, std, median, percentiles, CI)
  - Temporal profile with windowed segments and trend detection
  - Peak / valley moment detection
  - Per-metric contribution breakdown
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# ──────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────


def load_gaffe_json(path: str | Path) -> dict:
    """Load and validate a GAFFE JSON output file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GAFFE JSON file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Basic validation
    required_keys = {"metadata", "summary", "frames"}
    if not required_keys.issubset(data.keys()):
        missing = required_keys - data.keys()
        raise ValueError(f"Invalid GAFFE JSON — missing keys: {missing}")

    return data


def _extract_detected_frames(data: dict) -> list[dict]:
    """Extract only frames where a face was detected."""
    return [f for f in data["frames"] if f.get("face_detected", False)]


# ──────────────────────────────────────────────────────────────
# Summary statistics
# ──────────────────────────────────────────────────────────────


def compute_summary_stats(data: dict) -> dict[str, Any]:
    """
    Compute comprehensive summary statistics from GAFFE data.

    Returns a dictionary with:
      - score stats (mean, std, median, min, max, percentiles, 95% CI)
      - level distribution
      - face detection rate
    """
    frames = _extract_detected_frames(data)
    total_frames = len(data["frames"])

    if not frames:
        return {
            "total_frames": total_frames,
            "detected_frames": 0,
            "detection_rate": 0.0,
            "stress_score": None,
            "level_distribution": None,
        }

    scores = np.array([f["stress"]["score"] for f in frames])
    levels = [f["stress"]["level"] for f in frames]

    # 95% confidence interval for the mean
    n = len(scores)
    se = float(np.std(scores, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    z = 1.96  # 95% CI
    mean_val = float(np.mean(scores))

    level_counts = {}
    for level in ["LOW", "MODERATE", "HIGH", "VERY_HIGH"]:
        count = levels.count(level)
        level_counts[level] = {
            "count": count,
            "percentage": round(count / n * 100, 2),
        }

    return {
        "total_frames": total_frames,
        "detected_frames": n,
        "detection_rate": round(n / total_frames * 100, 2),
        "stress_score": {
            "mean": round(mean_val, 4),
            "std": round(float(np.std(scores, ddof=1)), 4) if n > 1 else 0.0,
            "median": round(float(np.median(scores)), 4),
            "min": round(float(np.min(scores)), 4),
            "max": round(float(np.max(scores)), 4),
            "p25": round(float(np.percentile(scores, 25)), 4),
            "p75": round(float(np.percentile(scores, 75)), 4),
            "p90": round(float(np.percentile(scores, 90)), 4),
            "ci_95_lower": round(max(0.0, mean_val - z * se), 4),
            "ci_95_upper": round(min(1.0, mean_val + z * se), 4),
        },
        "level_distribution": level_counts,
    }


# ──────────────────────────────────────────────────────────────
# Temporal profile (windowed segments)
# ──────────────────────────────────────────────────────────────


def _detect_trend(scores: np.ndarray) -> str:
    """
    Detect the linear trend of a score sequence.

    Returns one of: 'rising', 'falling', 'stable', 'volatile'.
    """
    if len(scores) < 3:
        return "stable"

    # Simple linear regression slope
    x = np.arange(len(scores), dtype=np.float64)
    slope = float(np.polyfit(x, scores, 1)[0])

    # Coefficient of variation as volatility measure
    cv = float(np.std(scores) / np.mean(scores)) if np.mean(scores) > 1e-6 else 0.0

    if cv > 0.3:
        return "volatile"
    elif slope > 0.005:
        return "rising"
    elif slope < -0.005:
        return "falling"
    else:
        return "stable"


def compute_temporal_profile(
    data: dict, window_seconds: float = 5.0
) -> dict[str, Any]:
    """
    Split the video into temporal segments and analyze each.

    Args:
        data: Loaded GAFFE JSON.
        window_seconds: Duration of each analysis window.

    Returns:
        Dictionary with segments and overall temporal stats.
    """
    fps = data["metadata"].get("video_fps", 30.0)
    frames = _extract_detected_frames(data)

    if not frames:
        return {"segments": [], "overall_trend": "unknown"}

    window_frames = max(int(window_seconds * fps), 1)
    scores = np.array([f["stress"]["score"] for f in frames])
    frame_ids = [f["frame_id"] for f in frames]

    segments = []
    for start_idx in range(0, len(frames), window_frames):
        end_idx = min(start_idx + window_frames, len(frames))
        seg_scores = scores[start_idx:end_idx]
        seg_frame_start = frame_ids[start_idx]
        seg_frame_end = frame_ids[end_idx - 1]

        start_s = seg_frame_start / fps
        end_s = (seg_frame_end + 1) / fps

        segments.append({
            "segment_id": len(segments),
            "start_s": round(start_s, 2),
            "end_s": round(end_s, 2),
            "frame_range": [seg_frame_start, seg_frame_end],
            "n_frames": len(seg_scores),
            "mean_stress": round(float(np.mean(seg_scores)), 4),
            "std_stress": round(float(np.std(seg_scores)), 4),
            "max_stress": round(float(np.max(seg_scores)), 4),
            "trend": _detect_trend(seg_scores),
        })

    overall_trend = _detect_trend(scores)

    return {
        "window_seconds": window_seconds,
        "n_segments": len(segments),
        "segments": segments,
        "overall_trend": overall_trend,
    }


# ──────────────────────────────────────────────────────────────
# Peak / valley detection
# ──────────────────────────────────────────────────────────────


def detect_peaks(
    data: dict,
    top_n: int = 5,
    smoothing_window: int = 15,
) -> dict[str, Any]:
    """
    Find peak and valley moments in the stress time series.

    Uses a smoothed version of the score to avoid noise peaks.

    Args:
        data: Loaded GAFFE JSON.
        top_n: Number of peaks/valleys to return.
        smoothing_window: Moving average window for smoothing.

    Returns:
        Dictionary with peak and valley moments.
    """
    fps = data["metadata"].get("video_fps", 30.0)
    frames = _extract_detected_frames(data)

    if len(frames) < smoothing_window:
        return {"peaks": [], "valleys": []}

    scores = np.array([f["stress"]["score"] for f in frames])
    frame_ids = [f["frame_id"] for f in frames]

    # Smooth the signal
    kernel = np.ones(smoothing_window) / smoothing_window
    smoothed = np.convolve(scores, kernel, mode="same")

    # Find local extrema (simple approach: compare with neighbors)
    peaks = []
    valleys = []
    margin = smoothing_window // 2

    for i in range(margin, len(smoothed) - margin):
        window = smoothed[i - margin : i + margin + 1]
        if smoothed[i] == np.max(window):
            peaks.append((i, smoothed[i]))
        elif smoothed[i] == np.min(window):
            valleys.append((i, smoothed[i]))

    # Sort and take top N
    peaks.sort(key=lambda x: x[1], reverse=True)
    valleys.sort(key=lambda x: x[1])

    def _build_moment(idx: int, score: float, frame: dict) -> dict:
        """Build a moment description with the dominant metric."""
        metrics = frame.get("metrics", {})
        dominant = "unknown"
        if metrics:
            contributions = {}
            if "bfi" in metrics:
                contributions["bfi"] = metrics["bfi"]["value"] * 0.45
            if "ear" in metrics:
                contributions["ear"] = metrics["ear"]["ear_stress_score"] * 0.35
            if "bad" in metrics:
                contributions["bad"] = metrics["bad"]["value"] * 0.20
            if contributions:
                dominant = max(contributions, key=contributions.get)

        return {
            "frame_id": frame_ids[idx],
            "timestamp_s": round(frame_ids[idx] / fps, 2),
            "score_smoothed": round(float(score), 4),
            "score_raw": round(float(scores[idx]), 4),
            "dominant_metric": dominant,
        }

    return {
        "peaks": [
            _build_moment(idx, score, frames[idx])
            for idx, score in peaks[:top_n]
        ],
        "valleys": [
            _build_moment(idx, score, frames[idx])
            for idx, score in valleys[:top_n]
        ],
    }


# ──────────────────────────────────────────────────────────────
# Metric contribution breakdown
# ──────────────────────────────────────────────────────────────


def compute_metric_breakdown(data: dict) -> dict[str, Any]:
    """
    Analyze the contribution of each metric to the composite stress score.

    For each metric, computes:
      - Raw mean value
      - Weighted contribution to composite score
      - Percentage of total contribution
    """
    frames = _extract_detected_frames(data)

    if not frames:
        return {}

    bfi_values = []
    ear_stress_values = []
    bad_values = []

    for f in frames:
        m = f["metrics"]
        bfi_values.append(m["bfi"]["value"])
        ear_stress_values.append(m["ear"]["ear_stress_score"])
        bad_values.append(m["bad"]["value"])

    bfi_arr = np.array(bfi_values)
    ear_arr = np.array(ear_stress_values)
    bad_arr = np.array(bad_values)

    # Weighted contributions (using the default weights from config)
    # These are read from the first frame's weights_used field
    weights = frames[0]["stress"].get("weights_used", {})
    w_bfi = weights.get("bfi", 0.45)
    w_ear = weights.get("ear", 0.35)
    w_bad = weights.get("bad", 0.20)

    bfi_contrib = float(np.mean(bfi_arr)) * w_bfi
    ear_contrib = float(np.mean(ear_arr)) * w_ear
    bad_contrib = float(np.mean(bad_arr)) * w_bad
    total_contrib = bfi_contrib + ear_contrib + bad_contrib

    def _pct(val: float) -> float:
        return round(val / total_contrib * 100, 2) if total_contrib > 0 else 0.0

    # EAR sub-components
    ear_values = [f["metrics"]["ear"]["value"] for f in frames]
    blink_rates = [f["metrics"]["ear"]["blink_rate_per_min"] for f in frames]
    blink_durations = [f["metrics"]["ear"]["avg_blink_duration_ms"] for f in frames]
    # Filter out zero blink durations (no data yet)
    valid_blink_durations = [d for d in blink_durations if d > 0]

    # BAD sub-components
    asymmetries = [f["metrics"]["bad"]["components"]["asymmetry"] for f in frames]
    bfi_variances = [f["metrics"]["bad"]["components"]["bfi_variance"] for f in frames]

    # BFI sub-components
    ied_values = [f["metrics"]["bfi"]["components"]["ied_normalized"] for f in frames]
    bed_values = [f["metrics"]["bfi"]["components"]["bed_normalized"] for f in frames]

    return {
        "bfi": {
            "raw_mean": round(float(np.mean(bfi_arr)), 4),
            "raw_std": round(float(np.std(bfi_arr)), 4),
            "weighted_contribution": round(bfi_contrib, 4),
            "contribution_pct": _pct(bfi_contrib),
            "components": {
                "ied_normalized_mean": round(float(np.mean(ied_values)), 4),
                "bed_normalized_mean": round(float(np.mean(bed_values)), 4),
            },
        },
        "ear": {
            "raw_mean": round(float(np.mean(ear_arr)), 4),
            "raw_std": round(float(np.std(ear_arr)), 4),
            "weighted_contribution": round(ear_contrib, 4),
            "contribution_pct": _pct(ear_contrib),
            "sub_metrics": {
                "ear_value_mean": round(float(np.mean(ear_values)), 4),
                "blink_rate_mean": round(float(np.mean(blink_rates)), 2),
                "blink_rate_max": round(float(np.max(blink_rates)), 2),
                "avg_blink_duration_ms": (
                    round(float(np.mean(valid_blink_durations)), 1)
                    if valid_blink_durations
                    else 0.0
                ),
            },
        },
        "bad": {
            "raw_mean": round(float(np.mean(bad_arr)), 4),
            "raw_std": round(float(np.std(bad_arr)), 4),
            "weighted_contribution": round(bad_contrib, 4),
            "contribution_pct": _pct(bad_contrib),
            "components": {
                "asymmetry_mean": round(float(np.mean(asymmetries)), 4),
                "bfi_variance_mean": round(float(np.mean(bfi_variances)), 6),
            },
        },
    }


# ──────────────────────────────────────────────────────────────
# Full single-video analysis
# ──────────────────────────────────────────────────────────────


def analyze_single_video(
    gaffe_json_path: str | Path,
    window_seconds: float = 5.0,
    top_n_peaks: int = 5,
) -> dict[str, Any]:
    """
    Run the full analysis pipeline on a single GAFFE JSON file.

    Args:
        gaffe_json_path: Path to the _gaffe.json file.
        window_seconds: Duration of temporal segments.
        top_n_peaks: Number of peak/valley moments to detect.

    Returns:
        Complete analysis dictionary ready for JSON serialization.
    """
    data = load_gaffe_json(gaffe_json_path)

    analysis = {
        "source_file": str(Path(gaffe_json_path).name),
        "video_metadata": data["metadata"],
        "summary": compute_summary_stats(data),
        "temporal_profile": compute_temporal_profile(data, window_seconds),
        "peaks_and_valleys": detect_peaks(data, top_n_peaks),
        "metric_breakdown": compute_metric_breakdown(data),
    }

    return analysis


def save_analysis(analysis: dict, output_path: str | Path) -> None:
    """Save analysis results to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
