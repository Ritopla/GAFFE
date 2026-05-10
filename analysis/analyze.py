#!/usr/bin/env python3
"""
GAFFE Analyze — Single-video stress analysis from GAFFE JSON output.

Usage:
    python analyze.py <gaffe_json> [--charts] [--output-dir results/]

Reads a _gaffe.json file produced by demo.py and generates:
  - Structured analysis JSON (summary, temporal profile, peaks, metric breakdown)
  - Optional: publication-ready matplotlib charts

Examples:
    # Analyze a single video (JSON output only)
    python analyze.py trial_truth_013_gaffe.json

    # Analyze with charts
    python analyze.py trial_truth_013_gaffe.json --charts

    # Custom output directory and chart format
    python analyze.py trial_truth_013_gaffe.json --charts --output-dir results/ --chart-format pdf
"""

import argparse
import json
import sys
from pathlib import Path

from analysis.single_video import (
    analyze_single_video,
    load_gaffe_json,
    save_analysis,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GAFFE — Single-video stress analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "gaffe_json",
        type=str,
        help="Path to a _gaffe.json file produced by demo.py",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Output directory for analysis JSON and charts (default: same as input)",
    )
    parser.add_argument(
        "--charts",
        action="store_true",
        help="Generate matplotlib visualization charts",
    )
    parser.add_argument(
        "--chart-format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Chart image format (default: png)",
    )
    parser.add_argument(
        "--chart-dpi",
        type=int,
        default=150,
        help="Chart resolution in DPI (default: 150)",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=5.0,
        help="Temporal segment window in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--top-peaks",
        type=int,
        default=5,
        help="Number of peak/valley moments to detect (default: 5)",
    )
    return parser.parse_args()


def print_summary(analysis: dict) -> None:
    """Print a human-readable summary to stdout."""
    summary = analysis["summary"]
    breakdown = analysis["metric_breakdown"]
    temporal = analysis["temporal_profile"]
    peaks = analysis["peaks_and_valleys"]

    print(f"\n{'═' * 60}")
    print(f"  GAFFE Analysis — {analysis['source_file']}")
    print(f"{'═' * 60}")

    # Video info
    meta = analysis["video_metadata"]
    print(f"\n  Video: {meta.get('video_file', '?')}")
    print(f"  Resolution: {meta.get('video_resolution', '?')} @ {meta.get('video_fps', '?')} FPS")
    total = meta.get("total_frames_processed", 0)
    fps = meta.get("video_fps", 30)
    print(f"  Duration: {total / fps:.1f}s ({total} frames)")

    # Stress summary
    ss = summary.get("stress_score")
    if ss:
        print(f"\n  ── Stress Score ──────────────────────────────")
        print(f"  Mean:   {ss['mean']:.4f}  (95% CI: [{ss['ci_95_lower']:.4f}, {ss['ci_95_upper']:.4f}])")
        print(f"  Median: {ss['median']:.4f}")
        print(f"  Std:    {ss['std']:.4f}")
        print(f"  Range:  [{ss['min']:.4f}, {ss['max']:.4f}]")
        print(f"  P25/P75: {ss['p25']:.4f} / {ss['p75']:.4f}")

    # Level distribution
    levels = summary.get("level_distribution", {})
    if levels:
        print(f"\n  ── Level Distribution ────────────────────────")
        for level, info in levels.items():
            bar = "█" * int(info["percentage"] / 2.5)
            print(f"  {level:>10}: {info['percentage']:5.1f}%  {bar}")

    # Metric breakdown
    if breakdown:
        print(f"\n  ── Metric Contribution ───────────────────────")
        for key, label in [("bfi", "BFI"), ("ear", "EAR"), ("bad", "BAD")]:
            m = breakdown[key]
            print(
                f"  {label:>5}: raw={m['raw_mean']:.4f}  "
                f"weighted={m['weighted_contribution']:.4f}  "
                f"({m['contribution_pct']:.1f}%)"
            )

    # Temporal profile
    if temporal.get("segments"):
        print(f"\n  ── Temporal Profile ({temporal['n_segments']} segments) ─────────")
        print(f"  Overall trend: {temporal['overall_trend']}")
        for seg in temporal["segments"][:10]:  # Show first 10
            trend_icon = {
                "rising": "↗", "falling": "↘",
                "stable": "→", "volatile": "↕",
            }.get(seg["trend"], "?")
            print(
                f"  [{seg['start_s']:6.1f}s – {seg['end_s']:6.1f}s] "
                f"μ={seg['mean_stress']:.3f} σ={seg['std_stress']:.3f} "
                f"{trend_icon} {seg['trend']}"
            )
        if temporal["n_segments"] > 10:
            print(f"  ... and {temporal['n_segments'] - 10} more segments")

    # Peaks
    if peaks.get("peaks"):
        print(f"\n  ── Peak Stress Moments ───────────────────────")
        for p in peaks["peaks"][:3]:
            print(
                f"  t={p['timestamp_s']:.1f}s  "
                f"score={p['score_smoothed']:.4f}  "
                f"driver={p['dominant_metric'].upper()}"
            )

    print(f"\n{'═' * 60}\n")


def main() -> None:
    args = parse_args()

    json_path = Path(args.gaffe_json)
    if not json_path.exists():
        print(f"Error: File not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    # Determine output directory
    output_dir = Path(args.output_dir) if args.output_dir else json_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run analysis
    print(f"Analyzing: {json_path.name}")
    analysis = analyze_single_video(
        json_path,
        window_seconds=args.window,
        top_n_peaks=args.top_peaks,
    )

    # Save analysis JSON
    stem = json_path.stem.replace("_gaffe", "")
    analysis_path = output_dir / f"{stem}_analysis.json"
    save_analysis(analysis, analysis_path)
    print(f"Analysis saved to: {analysis_path}")

    # Print summary
    print_summary(analysis)

    # Generate charts (if requested)
    if args.charts:
        from analysis.charts import generate_all_charts

        data = load_gaffe_json(json_path)
        charts_dir = output_dir / "charts"
        saved = generate_all_charts(
            data, analysis, charts_dir,
            fmt=args.chart_format, dpi=args.chart_dpi,
        )
        print(f"Charts saved to: {charts_dir}/")
        for p in saved:
            print(f"  → {p.name}")


if __name__ == "__main__":
    main()
