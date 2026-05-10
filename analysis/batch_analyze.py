#!/usr/bin/env python3
"""
GAFFE Batch Analyze — Process a dataset of videos for stress analysis.

Usage:
    python batch_analyze.py <directory> [--process-videos] [--charts]

Discovers GAFFE JSON files (or raw videos) in a directory, extracts
truth/lie labels from filenames, computes per-video analysis and
aggregate statistics, and performs statistical comparisons.

Examples:
    # Analyze existing GAFFE JSON files
    python batch_analyze.py ./dataset1/

    # Process raw videos first, then analyze with charts
    python batch_analyze.py ./dataset1/ --process-videos --charts

    # Custom output file
    python batch_analyze.py ./dataset1/ -o dataset1_report.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from analysis.batch import (
    batch_analyze,
    save_batch_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GAFFE — Batch stress analysis for a dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Directory containing video files and/or _gaffe.json files",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: <directory>/batch_analysis.json)",
    )
    parser.add_argument(
        "--process-videos",
        action="store_true",
        help="Run demo.py on unprocessed video files before analysis",
    )
    parser.add_argument(
        "--charts",
        action="store_true",
        help="Generate per-video dashboard charts",
    )
    parser.add_argument(
        "--chart-format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Chart image format (default: png)",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=5.0,
        help="Temporal segment window in seconds (default: 5.0)",
    )
    return parser.parse_args()


def print_batch_summary(result: dict) -> None:
    """Print a human-readable batch analysis summary."""
    ds = result.get("dataset", {})
    labeled = ds.get("labeled", {})
    groups = result.get("group_aggregation", {})
    comparison = result.get("truth_vs_lie", {})

    print(f"\n{'═' * 65}")
    print(f"  GAFFE Batch Analysis — {ds.get('directory', '?')}")
    print(f"{'═' * 65}")

    print(f"\n  Videos: {ds.get('total_videos', 0)} total "
          f"({labeled.get('truth', 0)} truth, "
          f"{labeled.get('lie', 0)} lie, "
          f"{labeled.get('unlabeled', 0)} unlabeled)")

    # Group summaries
    for group_name, color_label in [("all", "ALL VIDEOS"), ("truth", "TRUTH"), ("lie", "LIE")]:
        grp = groups.get(group_name, {})
        if grp.get("n_videos", 0) == 0:
            continue

        ss = grp.get("stress_score", {})
        metrics = grp.get("metrics", {})

        print(f"\n  ── {color_label} GROUP ({grp['n_videos']} videos) ────────────")
        print(f"  Stress: μ={ss.get('mean', 0):.4f} ± {ss.get('std', 0):.4f} "
              f"(CI: [{ss.get('ci_95_lower', 0):.4f}, {ss.get('ci_95_upper', 0):.4f}])")
        print(f"  BFI:    μ={metrics.get('bfi', {}).get('mean', 0):.4f}")
        print(f"  EAR:    μ={metrics.get('ear_stress', {}).get('mean', 0):.4f}")
        print(f"  BAD:    μ={metrics.get('bad', {}).get('mean', 0):.4f}")
        
        vid_str = ", ".join(grp.get("videos", []))
        if len(vid_str) > 80:
            vid_str = vid_str[:77] + "..."
        print(f"  Videos: {vid_str}")

    # Statistical comparison
    if comparison and comparison.get("stress_score"):
        comp = comparison["stress_score"]
        effect = comp.get("effect_size", {})
        mwu = comp.get("mann_whitney_u", {})

        print(f"\n  ── TRUTH vs LIE COMPARISON ───────────────────")
        print(f"  Cohen's d: {effect.get('cohens_d', 0):.4f} "
              f"({effect.get('interpretation', '?')}, "
              f"{effect.get('direction', '?')})")

        if mwu.get("p_value") is not None:
            sig = "✓ YES" if comp.get("significant_p005") else "✗ NO"
            print(f"  Mann-Whitney U: U={mwu.get('u_statistic', 0):.1f}, "
                  f"p={mwu.get('p_value', 0):.6f} "
                  f"(significant at α=0.05: {sig})")
            print(f"  Rank-biserial r: {mwu.get('rank_biserial_r', 0):.4f}")

        # Per-metric comparisons
        per_metric = comparison.get("per_metric", {})
        if per_metric:
            print(f"\n  ── PER-METRIC EFFECT SIZES ───────────────────")
            for mk, label in [("bfi", "BFI"), ("ear_stress", "EAR"), ("bad", "BAD")]:
                mc = per_metric.get(mk, {})
                me = mc.get("effect_size", {})
                if me:
                    print(f"  {label:>5}: d={me.get('cohens_d', 0):+.4f} "
                          f"({me.get('interpretation', '?')})")

    print(f"\n{'═' * 65}\n")


def main() -> None:
    args = parse_args()

    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"Error: Not a directory: {directory}", file=sys.stderr)
        sys.exit(1)

    # Use a local results/ directory to avoid writing into read-only
    # sources (e.g. Google Drive symlinks).
    results_dir = Path("results") / directory.resolve().name
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        Path(args.output) if args.output
        else results_dir / "batch_analysis.json"
    )

    print(f"Batch analysis: {directory}")
    print(f"Output dir:     {results_dir}")
    print(f"Output:         {output_path}")
    print("-" * 50)

    result = batch_analyze(
        directory,
        output_dir=results_dir,
        process_videos=args.process_videos,
        window_seconds=args.window,
        generate_charts=args.charts,
        chart_format=args.chart_format,
    )

    # Save
    save_batch_result(result, output_path)
    print(f"\nBatch results saved to: {output_path}")

    # Print summary
    print_batch_summary(result)


if __name__ == "__main__":
    main()
