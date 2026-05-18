#!/usr/bin/env python3
"""
GAFFE Analysis CLI — Analyze stress detection results.

Three subcommands are available:

  single   Analyze a single _gaffe.json file produced by detection/detect.py.
  batch    Analyze a directory of _gaffe.json files (one per video).
  compare  Cross-dataset comparison plots from two or more batch_analysis.json files.

Output is always written to results/<name>/ and is overwritten on each run.
Pass -o / --output to override the default location.

Usage:
    python -m analysis.cli single  <gaffe_json>      [options]
    python -m analysis.cli batch   <directory>       [options]
    python -m analysis.cli compare <json1> <json2>   [options]

Examples:
    # Analyze a single video result and generate the summary chart
    python -m analysis.cli single results/my_video/my_video_gaffe.json --charts

    # Generate all individual charts as well
    python -m analysis.cli single results/my_video/my_video_gaffe.json --charts --all-charts

    # Batch analysis with summary charts
    python -m analysis.cli batch datasets/Deceptive/ --charts

    # Batch analysis, process raw videos first, generate all charts
    python -m analysis.cli batch datasets/Deceptive/ --process-videos --charts --all-charts

    # Cross-dataset comparison
    python -m analysis.cli compare results/Deceptive/batch_analysis.json \
                                    results/Truthful/batch_analysis.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.single_video import (
    analyze_single_video,
    load_gaffe_json,
    save_analysis,
)
from analysis.batch import (
    batch_analyze,
    save_batch_result,
)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────


def _default_single_output_dir(gaffe_json: Path) -> Path:
    """Default output directory for a single-video analysis."""
    stem = gaffe_json.stem.replace("_gaffe", "")
    out = Path("results") / stem
    out.mkdir(parents=True, exist_ok=True)
    return out


def _default_batch_output_dir(directory: Path) -> Path:
    """Default output directory for a batch analysis."""
    out = Path("results") / directory.resolve().name
    out.mkdir(parents=True, exist_ok=True)
    return out


def _default_compare_output_dir() -> Path:
    """Default output directory for a comparison plot."""
    out = Path("results") / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ──────────────────────────────────────────────────────────────
# Subcommand: single
# ──────────────────────────────────────────────────────────────


def _print_single_summary(analysis: dict) -> None:
    """Print a human-readable single-video analysis summary to stdout."""
    summary = analysis["summary"]
    temporal = analysis["temporal_profile"]
    peaks = analysis["peaks_and_valleys"]
    method = analysis.get("detection_method") or \
             analysis.get("video_metadata", {}).get("detection_method", "landmark")

    print(f"\n{'═' * 60}")
    print(f"  GAFFE Analysis — {analysis['source_file']}")
    print(f"  Detection method: {method}")
    print(f"{'═' * 60}")

    meta = analysis["video_metadata"]
    print(f"\n  Video: {meta.get('video_file', '?')}")
    print(f"  Resolution: {meta.get('video_resolution', '?')} @ "
          f"{meta.get('video_fps', '?')} FPS")
    total = meta.get("total_frames_processed", 0)
    fps = meta.get("video_fps", 30) or 30
    print(f"  Duration: {total / fps:.1f}s ({total} frames)")

    ss = summary.get("stress_score")
    if ss:
        print(f"\n  ── Stress Score ──────────────────────────────")
        print(f"  Mean:   {ss['mean']:.4f}  "
              f"(95% CI: [{ss['ci_95_lower']:.4f}, {ss['ci_95_upper']:.4f}])")
        print(f"  Median: {ss['median']:.4f}")
        print(f"  Std:    {ss['std']:.4f}")
        print(f"  Range:  [{ss['min']:.4f}, {ss['max']:.4f}]")
        print(f"  P25/P75: {ss['p25']:.4f} / {ss['p75']:.4f}")

    levels = summary.get("level_distribution", {})
    if levels:
        print(f"\n  ── Level Distribution ────────────────────────")
        for level, info in levels.items():
            bar = "█" * int(info["percentage"] / 2.5)
            print(f"  {level:>10}: {info['percentage']:5.1f}%  {bar}")

    if method == "fer":
        breakdown = analysis.get("emotion_breakdown", {})
        if breakdown:
            print(f"\n  ── Emotion Contribution (FER) ────────────────")
            order = ["angry", "fear", "disgust", "happy", "sad", "surprise", "neutral"]
            for name in order:
                m = breakdown.get(name)
                if not m:
                    continue
                w = m.get("weight_used", 0.0)
                star = " ★" if w > 0 else "  "
                print(
                    f"  {name:>9}{star}: raw={m['raw_mean']:.4f}  "
                    f"weighted={m['weighted_contribution']:.4f}  "
                    f"({m['contribution_pct']:.1f}%)"
                )
            print(f"  (★ = emotion contributing to stress score)")
    else:
        breakdown = analysis.get("metric_breakdown", {})
        if breakdown:
            print(f"\n  ── Metric Contribution ───────────────────────")
            for key, label in [("bfi", "BFI"), ("ear", "EAR"), ("bad", "BAD")]:
                m = breakdown.get(key)
                if not m:
                    continue
                print(
                    f"  {label:>5}: raw={m['raw_mean']:.4f}  "
                    f"weighted={m['weighted_contribution']:.4f}  "
                    f"({m['contribution_pct']:.1f}%)"
                )

    if temporal.get("segments"):
        print(f"\n  ── Temporal Profile ({temporal['n_segments']} segments) ─────────")
        print(f"  Overall trend: {temporal['overall_trend']}")
        for seg in temporal["segments"][:10]:
            icon = {"rising": "↗", "falling": "↘", "stable": "→", "volatile": "↕"}.get(
                seg["trend"], "?"
            )
            print(
                f"  [{seg['start_s']:6.1f}s – {seg['end_s']:6.1f}s] "
                f"μ={seg['mean_stress']:.3f} σ={seg['std_stress']:.3f} "
                f"{icon} {seg['trend']}"
            )
        if temporal["n_segments"] > 10:
            print(f"  ... and {temporal['n_segments'] - 10} more segments")

    if peaks.get("peaks"):
        print(f"\n  ── Peak Stress Moments ───────────────────────")
        for p in peaks["peaks"][:3]:
            print(
                f"  t={p['timestamp_s']:.1f}s  "
                f"score={p['score_smoothed']:.4f}  "
                f"driver={p['dominant_metric'].upper()}"
            )

    print(f"\n{'═' * 60}\n")


def _cmd_single(args: argparse.Namespace) -> None:
    json_path = Path(args.gaffe_json)
    if not json_path.exists():
        print(f"Error: File not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = (
        Path(args.output_dir) if args.output_dir
        else _default_single_output_dir(json_path)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Analyzing: {json_path.name}")
    analysis = analyze_single_video(
        json_path,
        window_seconds=args.window,
        top_n_peaks=args.top_peaks,
    )

    stem = json_path.stem.replace("_gaffe", "")
    analysis_path = output_dir / f"{stem}_analysis.json"
    save_analysis(analysis, analysis_path)
    print(f"Analysis saved to: {analysis_path}")

    _print_single_summary(analysis)

    if args.charts:
        from analysis.charts import generate_all_charts
        data = load_gaffe_json(json_path)
        charts_dir = output_dir / "charts"
        saved = generate_all_charts(
            data, analysis, charts_dir,
            fmt=args.chart_format,
            dpi=args.chart_dpi,
            summary_only=not args.all_charts,
        )
        print(f"Charts saved to: {charts_dir}/")
        for p in saved:
            print(f"  → {p.name}")


# ──────────────────────────────────────────────────────────────
# Subcommand: batch
# ──────────────────────────────────────────────────────────────


def _print_batch_summary(result: dict) -> None:
    """Print a human-readable batch analysis summary to stdout."""
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


def _cmd_batch(args: argparse.Namespace) -> None:
    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"Error: Not a directory: {directory}", file=sys.stderr)
        sys.exit(1)

    output_dir = (
        Path(args.output_dir) if args.output_dir
        else _default_batch_output_dir(directory)
    )
    output_path = (
        Path(args.output) if args.output
        else output_dir / "batch_analysis.json"
    )

    print(f"Batch analysis: {directory}")
    print(f"Output dir:     {output_dir}")
    print(f"Output:         {output_path}")
    print("-" * 50)

    result = batch_analyze(
        directory,
        output_dir=output_dir,
        process_videos=args.process_videos,
        window_seconds=args.window,
        generate_charts=args.charts,
        chart_format=args.chart_format,
        summary_only=not args.all_charts,
    )

    save_batch_result(result, output_path)
    print(f"\nBatch results saved to: {output_path}")
    _print_batch_summary(result)


# ──────────────────────────────────────────────────────────────
# Subcommand: compare
# ──────────────────────────────────────────────────────────────


def _cmd_compare(args: argparse.Namespace) -> None:
    """Generate a cross-dataset comparison plot from batch_analysis.json files."""
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else _default_compare_output_dir()
    )
    output_path = (
        Path(args.output) if args.output
        else output_dir / "comparison.png"
    )

    json_paths = args.inputs
    labels: list[str] = []
    stress_means: list[float] = []
    stress_errors: list[float] = []
    bfi_means: list[float] = []
    ear_means: list[float] = []
    bad_means: list[float] = []
    all_stress_distributions: list[tuple[str, list[float], str]] = []

    colors = ["#2ca02c", "#d62728", "#1f77b4", "#ff7f0e"]

    for i, path_str in enumerate(json_paths):
        path = Path(path_str)
        if not path.exists():
            print(f"Warning: File not found: {path}", file=sys.stderr)
            continue
        with open(path) as f:
            data = json.load(f)

        label = path.parent.name if path.parent.name != "results" else f"Dataset {i + 1}"
        labels.append(label)

        agg = data.get("group_aggregation", {}).get("all", {})
        ss = agg.get("stress_score", {})
        metrics = agg.get("metrics", {})

        mean = ss.get("mean", 0.0)
        stress_means.append(mean)
        err = ss.get("ci_95_upper", mean) - mean or ss.get("std", 0.0)
        stress_errors.append(err)
        bfi_means.append(metrics.get("bfi", {}).get("mean", 0.0))
        ear_means.append(metrics.get("ear_stress", {}).get("mean", 0.0))
        bad_means.append(metrics.get("bad", {}).get("mean", 0.0))

        dist = [
            vid["analysis"]["summary"]["stress_score"]["mean"]
            for vid in data.get("per_video", [])
            if vid.get("analysis", {}).get("summary", {}).get("stress_score")
        ]
        all_stress_distributions.append((label, dist, colors[i % len(colors)]))

    if not labels:
        print("Error: No valid batch_analysis.json files provided.", file=sys.stderr)
        sys.exit(1)

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    x = np.arange(len(labels))

    # 1. Mean stress score with 95% CI error bars
    ax = axes[0, 0]
    ax.bar(x, stress_means, yerr=stress_errors, capsize=10,
           color=colors[: len(labels)], alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Mean Stress Score (with 95% CI)", fontsize=14)
    ax.set_ylabel("Stress Score")
    ax.set_ylim(0, max(stress_means) * 1.5 if stress_means else 1)

    # 2. Sub-metric grouped bar chart
    ax = axes[0, 1]
    width = 0.25
    ax.bar(x - width, bfi_means, width, label="BFI", color="#9467bd", alpha=0.7)
    ax.bar(x,         ear_means, width, label="EAR", color="#8c564b", alpha=0.7)
    ax.bar(x + width, bad_means, width, label="BAD", color="#e377c2", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Sub-Metric Contributions", fontsize=14)
    ax.set_ylabel("Metric Score")
    ax.legend()

    # 3. Stress score distribution histogram
    ax = axes[1, 0]
    for label, dist, color in all_stress_distributions:
        ax.hist(dist, bins=15, alpha=0.5, label=label, color=color, density=True)
    ax.set_title("Stress Score Distribution", fontsize=14)
    ax.set_xlabel("Mean Stress Score")
    ax.set_ylabel("Density")
    ax.legend()

    # 4. Box plot
    ax = axes[1, 1]
    plot_data = [dist for _, dist, _ in all_stress_distributions]
    ax.boxplot(
        plot_data, labels=labels, patch_artist=True,
        boxprops=dict(facecolor="lightblue", color="blue"),
        medianprops=dict(color="red", linewidth=2),
    )
    ax.set_title("Stress Score Boxplot", fontsize=14)
    ax.set_ylabel("Stress Score")

    plt.suptitle("GAFFE — Batch Results Comparison", fontsize=18, y=1.02)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved to: {output_path}")


# ──────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GAFFE — Stress analysis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── single ──────────────────────────────────────────────
    p_single = sub.add_parser(
        "single",
        help="Analyze a single _gaffe.json file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_single.add_argument("gaffe_json", help="Path to a _gaffe.json file")
    p_single.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Output directory (default: results/<video_stem>/)",
    )
    p_single.add_argument("--charts", action="store_true",
                          help="Generate charts (summary dashboard by default)")
    p_single.add_argument(
        "--all-charts", action="store_true",
        help="Generate all individual charts in addition to the dashboard "
             "(requires --charts)",
    )
    p_single.add_argument("--chart-format", default="png",
                          choices=["png", "pdf", "svg"])
    p_single.add_argument("--chart-dpi", type=int, default=150)
    p_single.add_argument("--window", type=float, default=5.0,
                          help="Temporal segment window in seconds (default: 5.0)")
    p_single.add_argument("--top-peaks", type=int, default=5,
                          help="Number of peak moments to detect (default: 5)")

    # ── batch ────────────────────────────────────────────────
    p_batch = sub.add_parser(
        "batch",
        help="Analyze a directory of _gaffe.json files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_batch.add_argument("directory", help="Directory with _gaffe.json files or videos")
    p_batch.add_argument(
        "-o", "--output",
        default=None,
        help="Output JSON file path (default: results/<dir_name>/batch_analysis.json)",
    )
    p_batch.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: results/<dir_name>/)",
    )
    p_batch.add_argument("--process-videos", action="store_true",
                         help="Run detect.py on unprocessed video files first")
    p_batch.add_argument("--charts", action="store_true",
                         help="Generate per-video summary dashboards")
    p_batch.add_argument(
        "--all-charts", action="store_true",
        help="Generate all individual charts per video (requires --charts)",
    )
    p_batch.add_argument("--chart-format", default="png",
                         choices=["png", "pdf", "svg"])
    p_batch.add_argument("--window", type=float, default=5.0,
                         help="Temporal segment window in seconds (default: 5.0)")

    # ── compare ──────────────────────────────────────────────
    p_compare = sub.add_parser(
        "compare",
        help="Cross-dataset comparison plot from batch_analysis.json files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_compare.add_argument("inputs", nargs="+",
                           help="Paths to batch_analysis.json files (at least 2)")
    p_compare.add_argument(
        "-o", "--output",
        default=None,
        help="Output image path (default: results/comparison/comparison.png)",
    )
    p_compare.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: results/comparison/)",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "single":
        _cmd_single(args)
    elif args.command == "batch":
        _cmd_batch(args)
    elif args.command == "compare":
        _cmd_compare(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
