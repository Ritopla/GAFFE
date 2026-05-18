"""
Batch analysis for GAFFE — process a dataset of videos, group by label,
and produce within-dataset statistical comparisons.

Handles:
  - Discovery and processing of video files via demo.py
  - Label extraction from filenames (truth/lie)
  - Per-group aggregation (mean, std, CI per metric)
  - Statistical tests: Mann-Whitney U, Cohen's d, rank-biserial correlation
  - Dataset-level summary report with optional charts
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from analysis.single_video import (
    analyze_single_video,
    load_gaffe_json,
    save_analysis,
)


# ──────────────────────────────────────────────────────────────
# Label extraction
# ──────────────────────────────────────────────────────────────


def extract_label(filename: str) -> str | None:
    """
    Extract truth/lie label from a filename.

    Supports common naming conventions:
      - 'trial_truth_013.mp4'  → 'truth'
      - 'AN_WILTY_EP16_lie18.mp4' → 'lie'
      - 'subject01_truthful.mp4' → 'truth'
      - 'subject01_deceptive.mp4' → 'lie'

    Returns 'truth', 'lie', or None if no label can be detected.
    """
    lower = filename.lower()

    lie_keywords = ["_lie", "lie_", "deceptive", "deceiving", "lie."]
    truth_keywords = ["_truth", "truth_", "truthful", "genuine", "truth."]

    for kw in lie_keywords:
        if kw in lower:
            return "lie"
    for kw in truth_keywords:
        if kw in lower:
            return "truth"

    return None


# ──────────────────────────────────────────────────────────────
# Statistical tests
# ──────────────────────────────────────────────────────────────


def _cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size between two groups.

    Uses pooled standard deviation. Returns 0.0 if both groups have zero variance.
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0

    var1 = float(np.var(group1, ddof=1))
    var2 = float(np.var(group2, ddof=1))

    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return 0.0

    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def _mann_whitney_u(
    group1: np.ndarray, group2: np.ndarray
) -> dict[str, float]:
    """
    Perform a Mann-Whitney U test (non-parametric).

    Uses scipy if available, otherwise falls back to a simple implementation.
    Returns dict with U statistic, p-value, and rank-biserial correlation.
    """
    try:
        from scipy.stats import mannwhitneyu
        u_stat, p_value = mannwhitneyu(
            group1, group2, alternative="two-sided"
        )
    except ImportError:
        # Simplified fallback (no p-value computation)
        combined = np.concatenate([group1, group2])
        ranks = np.argsort(np.argsort(combined)).astype(float) + 1
        n1 = len(group1)
        r1 = np.sum(ranks[:n1])
        u_stat = r1 - n1 * (n1 + 1) / 2
        p_value = float("nan")  # Can't compute without scipy

    n1, n2 = len(group1), len(group2)
    # Rank-biserial correlation: r = 1 - (2U)/(n1*n2)
    r_rb = 1.0 - (2.0 * u_stat) / (n1 * n2) if n1 * n2 > 0 else 0.0

    return {
        "u_statistic": round(float(u_stat), 2),
        "p_value": round(float(p_value), 6) if not np.isnan(p_value) else None,
        "rank_biserial_r": round(float(r_rb), 4),
    }


def _effect_size_label(d: float) -> str:
    """Cohen's d interpretation."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def compare_groups(
    truth_scores: np.ndarray,
    lie_scores: np.ndarray,
) -> dict[str, Any]:
    """
    Statistical comparison between truth and lie groups.

    Returns a dict with:
      - Group descriptive stats
      - Cohen's d effect size
      - Mann-Whitney U test results
    """
    if len(truth_scores) == 0 or len(lie_scores) == 0:
        return {"error": "One or both groups are empty"}

    d = _cohens_d(lie_scores, truth_scores)
    mwu = _mann_whitney_u(lie_scores, truth_scores)

    z = 1.96

    def _stats(arr: np.ndarray) -> dict:
        n = len(arr)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        se = std / np.sqrt(n) if n > 0 else 0.0
        return {
            "n": n,
            "mean": round(mean, 4),
            "std": round(std, 4),
            "median": round(float(np.median(arr)), 4),
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
            "ci_95_lower": round(max(0.0, mean - z * se), 4),
            "ci_95_upper": round(min(1.0, mean + z * se), 4),
        }

    return {
        "truth": _stats(truth_scores),
        "lie": _stats(lie_scores),
        "effect_size": {
            "cohens_d": round(d, 4),
            "interpretation": _effect_size_label(d),
            "direction": (
                "lie > truth" if d > 0
                else "truth > lie" if d < 0
                else "equal"
            ),
        },
        "mann_whitney_u": mwu,
        "significant_p005": (
            mwu["p_value"] is not None and mwu["p_value"] < 0.05
        ),
    }


# ──────────────────────────────────────────────────────────────
# Batch processing core
# ──────────────────────────────────────────────────────────────


@dataclass
class VideoResult:
    """Result for a single video in the batch."""
    filename: str
    label: str | None
    gaffe_json_path: Path
    analysis: dict


def discover_videos(
    directory: str | Path,
    extensions: tuple[str, ...] = (".mp4", ".avi", ".mkv", ".mov", ".webm"),
) -> list[Path]:
    """Find all video files in a directory."""
    directory = Path(directory)
    videos = []
    for ext in extensions:
        videos.extend(directory.glob(f"*{ext}"))
    return sorted(videos)


def discover_gaffe_jsons(directory: str | Path) -> list[Path]:
    """Find all existing _gaffe.json files in a directory."""
    directory = Path(directory)
    return sorted(directory.glob("*_gaffe.json"))


def run_demo_on_video(
    video_path: Path,
    output_dir: Path | None = None,
    model_path: Path | None = None,
    python_path: str = sys.executable,
) -> Path | None:
    """
    Run detection/detect.py on a single video file to produce a GAFFE JSON.

    The script is invoked as a module (``-m detection.detect``) so that
    package-relative imports work correctly regardless of the caller's cwd.
    The working directory is always set to the project root (two levels up
    from this file), which is required for the module resolution to succeed.

    If output_dir is given the JSON is written there instead of next to the
    video — useful when the video lives on a read-only mount (e.g. Google Drive).

    Returns the path to the generated JSON, or None on failure.
    """
    if output_dir is not None:
        gaffe_json = output_dir / f"{video_path.stem}_gaffe.json"
    else:
        gaffe_json = video_path.with_name(f"{video_path.stem}_gaffe.json")

    # Skip if already processed
    if gaffe_json.exists():
        return gaffe_json

    # Project root — needed as cwd so `python -m detection.detect` resolves.
    project_root = Path(__file__).parent.parent

    # Invoke as a module, not as a bare script, so package imports work.
    cmd = [
        python_path, "-m", "detection.detect",
        str(video_path),
        "--no-display",
        "--output", str(gaffe_json),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(project_root),
        )
        if result.returncode != 0:
            # Print full stderr so errors are visible without truncation.
            print(f"  Error processing {video_path.name}:\n{result.stderr}",
                  file=sys.stderr)
            return None
    except subprocess.TimeoutExpired:
        print(f"  Timeout processing {video_path.name}", file=sys.stderr)
        return None

    return gaffe_json if gaffe_json.exists() else None


def batch_analyze(
    directory: str | Path,
    output_dir: str | Path | None = None,
    process_videos: bool = False,
    window_seconds: float = 5.0,
    generate_charts: bool = False,
    chart_format: str = "png",
    chart_dpi: int = 150,
    summary_only: bool = True,
) -> dict[str, Any]:
    """
    Analyze all GAFFE JSON files in a directory.

    If process_videos=True, first runs detect.py on any unprocessed video files.

    Args:
        directory:      Path containing video files and/or _gaffe.json files.
        output_dir:     Writable directory for outputs (JSON, charts). If None,
                        defaults to the input directory (requires write access).
        process_videos: If True, run detect.py on unprocessed videos first.
        window_seconds: Temporal window for per-video analysis.
        generate_charts: Generate per-video dashboard charts.
        chart_format:   Chart image format ('png', 'pdf', 'svg').
        chart_dpi:      Chart resolution in DPI.
        summary_only:   If True (default), generate only the summary dashboard
                        per video. If False, generate all individual charts.

    Returns:
        Complete batch analysis dict with per-video results and group comparison.
    """
    directory = Path(directory)
    out_dir = Path(output_dir) if output_dir else directory
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    # Step 1: Optionally process raw videos
    if process_videos:
        videos = discover_videos(directory)
        print(f"Found {len(videos)} video files in {directory}")
        for i, vid in enumerate(videos):
            gaffe_path = out_dir / f"{vid.stem}_gaffe.json"
            if gaffe_path.exists():
                print(f"  [{i+1}/{len(videos)}] {vid.name} — already processed")
            else:
                print(f"  [{i+1}/{len(videos)}] {vid.name} — processing...")
                run_demo_on_video(vid, output_dir=out_dir)

    # Step 2: Discover GAFFE JSONs (check both source and output dirs)
    gaffe_files = discover_gaffe_jsons(out_dir)
    if not gaffe_files and out_dir != directory:
        gaffe_files = discover_gaffe_jsons(directory)
    if not gaffe_files:
        return {"error": "No _gaffe.json files found", "directory": str(directory)}

    print(f"\nAnalyzing {len(gaffe_files)} GAFFE JSON files...")

    # Step 3: Analyze each file
    results: list[VideoResult] = []
    for gf in gaffe_files:
        video_stem = gf.stem.replace("_gaffe", "")
        label = extract_label(gf.name)

        try:
            analysis = analyze_single_video(gf, window_seconds)
        except Exception as e:
            print(f"  Warning: Failed to analyze {gf.name}: {e}", file=sys.stderr)
            continue

        # Generate per-video charts if requested
        if generate_charts:
            try:
                from analysis.charts import generate_all_charts
                data = load_gaffe_json(gf)
                charts_dir = out_dir / "charts"
                generate_all_charts(
                    data, analysis, charts_dir,
                    fmt=chart_format, dpi=chart_dpi,
                    summary_only=summary_only,
                )
                print(f"  ✓ {gf.name} (label={label or '?'}) — charts generated")
            except Exception as e:
                print(f"  ✓ {gf.name} (label={label or '?'}) — charts failed: {e}")
        else:
            print(f"  ✓ {gf.name} (label={label or '?'})")

        results.append(VideoResult(
            filename=gf.name,
            label=label,
            gaffe_json_path=gf,
            analysis=analysis,
        ))

    # Filter out videos where face detection failed entirely
    valid_results = [
        r for r in results
        if r.analysis.get("summary", {}).get("stress_score") is not None
        and r.analysis.get("metric_breakdown")
    ]

    # Step 4: Group by label
    truth_results = [r for r in valid_results if r.label == "truth"]
    lie_results = [r for r in valid_results if r.label == "lie"]
    unlabeled = [r for r in valid_results if r.label is None]

    # Step 5: Aggregate per group
    def _aggregate_group(group: list[VideoResult]) -> dict[str, Any]:
        if not group:
            return {"n_videos": 0}

        stress_means = np.array([
            r.analysis["summary"]["stress_score"]["mean"]
            for r in group
        ])
        bfi_means = np.array([
            r.analysis["metric_breakdown"]["bfi"]["raw_mean"]
            for r in group
        ])
        ear_means = np.array([
            r.analysis["metric_breakdown"]["ear"]["raw_mean"]
            for r in group
        ])
        bad_means = np.array([
            r.analysis["metric_breakdown"]["bad"]["raw_mean"]
            for r in group
        ])

        z = 1.96

        def _summary(arr: np.ndarray) -> dict:
            n = len(arr)
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
            se = std / np.sqrt(n) if n > 0 else 0.0
            return {
                "mean": round(mean, 4),
                "std": round(std, 4),
                "median": round(float(np.median(arr)), 4),
                "min": round(float(np.min(arr)), 4),
                "max": round(float(np.max(arr)), 4),
                "ci_95_lower": round(max(0.0, mean - z * se), 4),
                "ci_95_upper": round(min(1.0, mean + z * se), 4),
            }

        return {
            "n_videos": len(group),
            "videos": [r.filename for r in group],
            "stress_score": _summary(stress_means),
            "metrics": {
                "bfi": _summary(bfi_means),
                "ear_stress": _summary(ear_means),
                "bad": _summary(bad_means),
            },
        }

    truth_agg = _aggregate_group(truth_results)
    lie_agg = _aggregate_group(lie_results)

    # Step 6: Statistical comparison (truth vs lie)
    comparison = None
    metric_comparisons = {}
    if truth_results and lie_results:
        truth_stress = np.array([
            r.analysis["summary"]["stress_score"]["mean"]
            for r in truth_results
        ])
        lie_stress = np.array([
            r.analysis["summary"]["stress_score"]["mean"]
            for r in lie_results
        ])
        comparison = compare_groups(truth_stress, lie_stress)

        # Per-metric comparison
        for metric_key, extract_fn in [
            ("bfi", lambda r: r.analysis["metric_breakdown"]["bfi"]["raw_mean"]),
            ("ear_stress", lambda r: r.analysis["metric_breakdown"]["ear"]["raw_mean"]),
            ("bad", lambda r: r.analysis["metric_breakdown"]["bad"]["raw_mean"]),
        ]:
            t_vals = np.array([extract_fn(r) for r in truth_results])
            l_vals = np.array([extract_fn(r) for r in lie_results])
            metric_comparisons[metric_key] = compare_groups(t_vals, l_vals)

    # Step 7: Build output
    batch_result = {
        "dataset": {
            "directory": str(directory),
            "total_videos": len(results),
            "labeled": {
                "truth": len(truth_results),
                "lie": len(lie_results),
                "unlabeled": len(unlabeled),
            },
        },
        "group_aggregation": {
            "all": _aggregate_group(valid_results),
            "truth": truth_agg,
            "lie": lie_agg,
        },
        "truth_vs_lie": {
            "stress_score": comparison,
            "per_metric": metric_comparisons,
        } if comparison else None,
        "per_video": [
            {
                "filename": r.filename,
                "label": r.label,
                "analysis": r.analysis,
            }
            for r in results
        ],
    }

    return batch_result


def save_batch_result(result: dict, output_path: str | Path) -> None:
    """Save batch analysis result to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
