"""
Matplotlib visualizations for single-video stress analysis.

All functions accept GAFFE JSON data (loaded dict) and produce
publication-ready charts. Each function returns a matplotlib Figure
so callers can save or display as needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for file output
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.patches import FancyBboxPatch
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ──────────────────────────────────────────────────────────────
# Style configuration
# ──────────────────────────────────────────────────────────────

# Color palette
COLORS = {
    "stress_line": "#E63946",
    "stress_fill": "#E63946",
    "bfi": "#457B9D",
    "ear": "#E9C46A",
    "bad": "#2A9D8F",
    "LOW": "#06D6A0",
    "MODERATE": "#FFD166",
    "HIGH": "#F77F00",
    "VERY_HIGH": "#EF476F",
    "grid": "#E5E5E5",
    "bg": "#FAFAFA",
    "text": "#2B2D42",
    "text_light": "#8D99AE",
}

# Color palette per le 7 emozioni FER
EMOTION_COLORS = {
    "angry":    "#E63946",  # rosso
    "fear":     "#9D4EDD",  # viola
    "disgust":  "#2A9D8F",  # verde acqua
    "happy":    "#FFD166",  # giallo
    "sad":      "#457B9D",  # blu
    "surprise": "#F77F00",  # arancio
    "neutral":  "#8D99AE",  # grigio
}

# Ordine canonico delle emozioni per visualizzazioni e summary
FER_EMOTION_NAMES = (
    "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral",
)

# Emozioni considerate "di stress" da FER (contribuiscono allo stress score)
FER_STRESS_EMOTIONS = ("angry", "fear", "disgust")


def _apply_style(fig, ax):
    """Apply consistent styling to a chart."""
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["text_light"])
    ax.spines["bottom"].set_color(COLORS["text_light"])
    ax.tick_params(colors=COLORS["text_light"], labelsize=9)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)


def _check_matplotlib():
    """Raise if matplotlib is not installed."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for chart generation. "
            "Install it with: pip install matplotlib"
        )


# ──────────────────────────────────────────────────────────────
# 1. Stress timeline
# ──────────────────────────────────────────────────────────────


def plot_stress_timeline(
    data: dict,
    smoothing_window: int = 15,
    figsize: tuple[float, float] = (14, 5),
) -> "plt.Figure":
    """
    Plot the stress score over time with level-colored background bands.

    Shows both raw and smoothed stress signal, with colored horizontal
    bands indicating the classification thresholds.
    """
    _check_matplotlib()

    fps = data["metadata"].get("video_fps", 30.0)
    frames = [f for f in data["frames"] if f.get("face_detected", False)]

    if not frames:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No face detected", ha="center", va="center")
        return fig

    times = np.array([f["frame_id"] / fps for f in frames])
    scores = np.array([f["stress"]["score"] for f in frames])

    # Smooth
    kernel = np.ones(smoothing_window) / smoothing_window
    smoothed = np.convolve(scores, kernel, mode="same")

    fig, ax = plt.subplots(figsize=figsize)
    _apply_style(fig, ax)

    # Level bands
    band_levels = [
        (0.00, 0.25, COLORS["LOW"], "LOW", 0.08),
        (0.25, 0.50, COLORS["MODERATE"], "MODERATE", 0.08),
        (0.50, 0.75, COLORS["HIGH"], "HIGH", 0.08),
        (0.75, 1.00, COLORS["VERY_HIGH"], "VERY HIGH", 0.08),
    ]
    for y_low, y_high, color, label, alpha in band_levels:
        ax.axhspan(y_low, y_high, color=color, alpha=alpha)
        ax.text(
            times[-1] * 1.01, (y_low + y_high) / 2, label,
            fontsize=7, color=color, va="center", alpha=0.8,
            fontweight="bold",
        )

    # Raw signal (transparent)
    ax.plot(
        times, scores,
        color=COLORS["stress_line"], alpha=0.2, linewidth=0.5,
        label="Raw",
    )

    # Smoothed signal
    ax.plot(
        times, smoothed,
        color=COLORS["stress_line"], linewidth=2.0,
        label=f"Smoothed (w={smoothing_window})",
    )

    # Fill under smoothed curve
    ax.fill_between(
        times, 0, smoothed,
        color=COLORS["stress_fill"], alpha=0.1,
    )

    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time (s)", fontsize=10, color=COLORS["text"])
    ax.set_ylabel("Stress Score", fontsize=10, color=COLORS["text"])
    ax.set_title(
        f"Stress Timeline — {data['metadata'].get('video_file', '?')}",
        fontsize=13, fontweight="bold", color=COLORS["text"], pad=15,
    )
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────
# 2. Metric contribution stacked area
# ──────────────────────────────────────────────────────────────


def plot_metric_contributions(
    data: dict,
    smoothing_window: int = 15,
    figsize: tuple[float, float] = (14, 5),
) -> "plt.Figure":
    """
    Stacked area chart showing each metric's weighted contribution
    to the composite stress score over time.
    """
    _check_matplotlib()

    fps = data["metadata"].get("video_fps", 30.0)
    frames = [f for f in data["frames"] if f.get("face_detected", False)]

    if not frames:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No face detected", ha="center", va="center")
        return fig

    times = np.array([f["frame_id"] / fps for f in frames])

    # Extract weighted contributions
    weights = frames[0]["stress"].get("weights_used", {})
    w_bfi = weights.get("bfi", 0.45)
    w_ear = weights.get("ear", 0.35)
    w_bad = weights.get("bad", 0.20)

    bfi_contrib = np.array([f["metrics"]["bfi"]["value"] * w_bfi for f in frames])
    ear_contrib = np.array([f["metrics"]["ear"]["ear_stress_score"] * w_ear for f in frames])
    bad_contrib = np.array([f["metrics"]["bad"]["value"] * w_bad for f in frames])

    # Smooth all
    kernel = np.ones(smoothing_window) / smoothing_window
    bfi_s = np.convolve(bfi_contrib, kernel, mode="same")
    ear_s = np.convolve(ear_contrib, kernel, mode="same")
    bad_s = np.convolve(bad_contrib, kernel, mode="same")

    fig, ax = plt.subplots(figsize=figsize)
    _apply_style(fig, ax)

    ax.stackplot(
        times, bfi_s, ear_s, bad_s,
        labels=[
            f"BFI (w={w_bfi})",
            f"EAR Stress (w={w_ear})",
            f"BAD (w={w_bad})",
        ],
        colors=[COLORS["bfi"], COLORS["ear"], COLORS["bad"]],
        alpha=0.8,
    )

    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time (s)", fontsize=10, color=COLORS["text"])
    ax.set_ylabel("Weighted Contribution", fontsize=10, color=COLORS["text"])
    ax.set_title(
        f"Metric Contributions — {data['metadata'].get('video_file', '?')}",
        fontsize=13, fontweight="bold", color=COLORS["text"], pad=15,
    )
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────
# 3. Level distribution pie chart
# ──────────────────────────────────────────────────────────────


def plot_level_distribution(
    data: dict,
    figsize: tuple[float, float] = (7, 7),
) -> "plt.Figure":
    """Donut chart of stress level distribution over all frames."""
    _check_matplotlib()

    frames = [f for f in data["frames"] if f.get("face_detected", False)]
    if not frames:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No face detected", ha="center", va="center")
        return fig

    levels = [f["stress"]["level"] for f in frames]
    level_order = ["LOW", "MODERATE", "HIGH", "VERY_HIGH"]
    counts = [levels.count(l) for l in level_order]

    # Filter out zero counts for cleaner chart
    labels = []
    sizes = []
    colors = []
    for l, c in zip(level_order, counts):
        if c > 0:
            labels.append(l)
            sizes.append(c)
            colors.append(COLORS[l])

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(COLORS["bg"])

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.82,
        wedgeprops={"width": 0.4, "edgecolor": "white", "linewidth": 2},
    )

    for t in texts:
        t.set_fontsize(10)
        t.set_color(COLORS["text"])
        t.set_fontweight("bold")
    for t in autotexts:
        t.set_fontsize(9)
        t.set_color(COLORS["text"])

    # Center text
    mean_score = float(np.mean([f["stress"]["score"] for f in frames]))
    ax.text(
        0, 0, f"{mean_score:.2f}",
        ha="center", va="center",
        fontsize=28, fontweight="bold", color=COLORS["text"],
    )
    ax.text(
        0, -0.12, "mean stress",
        ha="center", va="center",
        fontsize=10, color=COLORS["text_light"],
    )

    ax.set_title(
        f"Stress Level Distribution — {data['metadata'].get('video_file', '?')}",
        fontsize=13, fontweight="bold", color=COLORS["text"], pad=20,
    )

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────
# 4. Metric breakdown bar chart
# ──────────────────────────────────────────────────────────────


def plot_metric_bars(
    analysis: dict,
    figsize: tuple[float, float] = (10, 6),
) -> "plt.Figure":
    """
    Horizontal bar chart comparing each metric's contribution.

    Takes the analysis dict (output of analyze_single_video).
    """
    _check_matplotlib()

    breakdown = analysis.get("metric_breakdown", {})
    if not breakdown:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    metrics = ["bfi", "ear", "bad"]
    labels = ["BFI\n(Brow Furrow)", "EAR\n(Eye/Blink)", "BAD\n(Brow Asymmetry)"]
    raw_means = [breakdown[m]["raw_mean"] for m in metrics]
    weighted = [breakdown[m]["weighted_contribution"] for m in metrics]
    pcts = [breakdown[m]["contribution_pct"] for m in metrics]
    bar_colors = [COLORS[m] for m in metrics]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    _apply_style(fig, ax1)
    _apply_style(fig, ax2)

    # Left: raw mean values
    bars1 = ax1.barh(labels, raw_means, color=bar_colors, height=0.5, alpha=0.85)
    ax1.set_xlim(0, 1)
    ax1.set_xlabel("Raw Mean Value", fontsize=10, color=COLORS["text"])
    ax1.set_title("Raw Metric Values", fontsize=12, fontweight="bold", color=COLORS["text"])
    for bar, val in zip(bars1, raw_means):
        ax1.text(
            val + 0.02, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center", fontsize=10, color=COLORS["text"],
        )

    # Right: weighted contributions with percentage labels
    bars2 = ax2.barh(labels, weighted, color=bar_colors, height=0.5, alpha=0.85)
    ax2.set_xlim(0, max(weighted) * 1.5 if weighted else 1)
    ax2.set_xlabel("Weighted Contribution", fontsize=10, color=COLORS["text"])
    ax2.set_title("Contribution to Stress Score", fontsize=12, fontweight="bold", color=COLORS["text"])
    for bar, val, pct in zip(bars2, weighted, pcts):
        ax2.text(
            val + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f} ({pct:.1f}%)",
            va="center", fontsize=10, color=COLORS["text"],
        )

    fig.suptitle(
        f"Metric Breakdown — {analysis.get('source_file', '?')}",
        fontsize=14, fontweight="bold", color=COLORS["text"], y=1.02,
    )

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────
# 5. Blink rate overlay
# ──────────────────────────────────────────────────────────────


def plot_blink_rate(
    data: dict,
    figsize: tuple[float, float] = (14, 5),
) -> "plt.Figure":
    """
    Dual-axis plot: stress score + blink rate over time.

    Useful for investigating the relationship between blink dynamics
    and overall stress levels.
    """
    _check_matplotlib()

    fps = data["metadata"].get("video_fps", 30.0)
    frames = [f for f in data["frames"] if f.get("face_detected", False)]

    if not frames:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No face detected", ha="center", va="center")
        return fig

    times = np.array([f["frame_id"] / fps for f in frames])
    scores = np.array([f["stress"]["score"] for f in frames])
    blink_rates = np.array([f["metrics"]["ear"]["blink_rate_per_min"] for f in frames])

    fig, ax1 = plt.subplots(figsize=figsize)
    _apply_style(fig, ax1)

    # Stress score
    ax1.plot(times, scores, color=COLORS["stress_line"], alpha=0.5, linewidth=1)
    ax1.set_ylabel("Stress Score", fontsize=10, color=COLORS["stress_line"])
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis="y", labelcolor=COLORS["stress_line"])

    # Blink rate on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(times, blink_rates, color=COLORS["bfi"], alpha=0.8, linewidth=1.5)
    ax2.set_ylabel("Blink Rate (blinks/min)", fontsize=10, color=COLORS["bfi"])
    ax2.tick_params(axis="y", labelcolor=COLORS["bfi"])
    ax2.spines["right"].set_color(COLORS["bfi"])
    ax2.spines["top"].set_visible(False)

    # Reference lines for blink baselines
    ax2.axhline(y=17.5, color=COLORS["bfi"], linestyle="--", alpha=0.3, linewidth=1)
    ax2.axhline(y=25.0, color=COLORS["VERY_HIGH"], linestyle="--", alpha=0.3, linewidth=1)
    ax2.text(times[0], 17.5, " baseline (17.5)", fontsize=7, color=COLORS["bfi"], alpha=0.6, va="bottom")
    ax2.text(times[0], 25.0, " stress (25.0)", fontsize=7, color=COLORS["VERY_HIGH"], alpha=0.6, va="bottom")

    ax1.set_xlabel("Time (s)", fontsize=10, color=COLORS["text"])
    ax1.set_title(
        f"Stress Score vs Blink Rate — {data['metadata'].get('video_file', '?')}",
        fontsize=13, fontweight="bold", color=COLORS["text"], pad=15,
    )

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────
# 6. Summary dashboard — all key info in one figure
# ──────────────────────────────────────────────────────────────


def plot_summary_dashboard(
    data: dict,
    analysis: dict,
    smoothing_window: int = 15,
    figsize: tuple[float, float] = (20, 14),
) -> "plt.Figure":
    """
    Comprehensive single-figure dashboard combining all key information.

    Layout (3 rows):
      Row 1: Stress timeline with level bands (full width)
      Row 2: Metric contribution stacked area (full width)
      Row 3: Level donut | Metric bars | Key stats panel
    """
    _check_matplotlib()
    from matplotlib.gridspec import GridSpec

    fps = data["metadata"].get("video_fps", 30.0)
    frames = [f for f in data["frames"] if f.get("face_detected", False)]
    video_name = data["metadata"].get("video_file", "Unknown")
    total_frames = data["metadata"].get("total_frames_processed", len(data["frames"]))
    duration_s = total_frames / fps

    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor(COLORS["bg"])

    gs = GridSpec(
        3, 3, figure=fig,
        height_ratios=[1.0, 0.8, 1.0],
        hspace=0.35, wspace=0.35,
    )

    # ── Title ─────────────────────────────────────────────────
    fig.suptitle(
        f"GAFFE Stress Analysis — {video_name}",
        fontsize=18, fontweight="bold", color=COLORS["text"],
        y=0.98,
    )

    if not frames:
        ax = fig.add_subplot(gs[:, :])
        ax.text(0.5, 0.5, "No face detected in video",
                ha="center", va="center", fontsize=16, color=COLORS["text_light"])
        ax.set_facecolor(COLORS["bg"])
        ax.axis("off")
        return fig

    times = np.array([f["frame_id"] / fps for f in frames])
    scores = np.array([f["stress"]["score"] for f in frames])
    kernel = np.ones(smoothing_window) / smoothing_window
    smoothed = np.convolve(scores, kernel, mode="same")

    # ── Row 1: Stress Timeline ────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    _apply_style(fig, ax1)

    band_levels = [
        (0.00, 0.25, COLORS["LOW"], "LOW", 0.08),
        (0.25, 0.50, COLORS["MODERATE"], "MOD", 0.08),
        (0.50, 0.75, COLORS["HIGH"], "HIGH", 0.08),
        (0.75, 1.00, COLORS["VERY_HIGH"], "V.HIGH", 0.08),
    ]
    for y_low, y_high, color, label, alpha in band_levels:
        ax1.axhspan(y_low, y_high, color=color, alpha=alpha)
        ax1.text(
            times[-1] * 1.005, (y_low + y_high) / 2, label,
            fontsize=7, color=color, va="center", alpha=0.9, fontweight="bold",
        )

    ax1.plot(times, scores, color=COLORS["stress_line"], alpha=0.15, linewidth=0.5)
    ax1.plot(times, smoothed, color=COLORS["stress_line"], linewidth=2.0, label="Smoothed")
    ax1.fill_between(times, 0, smoothed, color=COLORS["stress_fill"], alpha=0.1)

    ax1.set_xlim(times[0], times[-1])
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Time (s)", fontsize=10, color=COLORS["text"])
    ax1.set_ylabel("Stress Score", fontsize=10, color=COLORS["text"])
    ax1.set_title("Stress Timeline", fontsize=12, fontweight="bold",
                   color=COLORS["text"], loc="left", pad=8)
    ax1.legend(fontsize=8, loc="upper right", framealpha=0.9)

    # ── Row 2: Metric Contributions Stacked Area ─────────────
    ax2 = fig.add_subplot(gs[1, :])
    _apply_style(fig, ax2)

    weights = frames[0]["stress"].get("weights_used", {})
    w_bfi = weights.get("bfi", 0.45)
    w_ear = weights.get("ear", 0.35)
    w_bad = weights.get("bad", 0.20)

    bfi_c = np.convolve(
        np.array([f["metrics"]["bfi"]["value"] * w_bfi for f in frames]),
        kernel, mode="same")
    ear_c = np.convolve(
        np.array([f["metrics"]["ear"]["ear_stress_score"] * w_ear for f in frames]),
        kernel, mode="same")
    bad_c = np.convolve(
        np.array([f["metrics"]["bad"]["value"] * w_bad for f in frames]),
        kernel, mode="same")

    ax2.stackplot(
        times, bfi_c, ear_c, bad_c,
        labels=[f"BFI ({w_bfi})", f"EAR ({w_ear})", f"BAD ({w_bad})"],
        colors=[COLORS["bfi"], COLORS["ear"], COLORS["bad"]],
        alpha=0.8,
    )
    ax2.set_xlim(times[0], times[-1])
    ax2.set_ylim(0, max(0.6, float(np.max(bfi_c + ear_c + bad_c)) * 1.1))
    ax2.set_xlabel("Time (s)", fontsize=10, color=COLORS["text"])
    ax2.set_ylabel("Weighted Contribution", fontsize=10, color=COLORS["text"])
    ax2.set_title("Metric Contributions Over Time", fontsize=12, fontweight="bold",
                   color=COLORS["text"], loc="left", pad=8)
    ax2.legend(fontsize=8, loc="upper right", framealpha=0.9)

    # ── Row 3, Left: Level Distribution Donut ─────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.set_facecolor(COLORS["bg"])

    levels_list = [f["stress"]["level"] for f in frames]
    level_order = ["LOW", "MODERATE", "HIGH", "VERY_HIGH"]
    counts = [levels_list.count(l) for l in level_order]

    pie_labels, pie_sizes, pie_colors = [], [], []
    for l, c in zip(level_order, counts):
        if c > 0:
            pie_labels.append(l)
            pie_sizes.append(c)
            pie_colors.append(COLORS[l])

    if pie_sizes:
        wedges, texts, autotexts = ax3.pie(
            pie_sizes, labels=pie_labels, colors=pie_colors,
            autopct="%1.1f%%", startangle=90, pctdistance=0.80,
            wedgeprops={"width": 0.4, "edgecolor": "white", "linewidth": 2},
        )
        for t in texts:
            t.set_fontsize(9)
            t.set_color(COLORS["text"])
            t.set_fontweight("bold")
        for t in autotexts:
            t.set_fontsize(8)
            t.set_color(COLORS["text"])

    mean_score = float(np.mean(scores))
    ax3.text(0, 0.04, f"{mean_score:.2f}", ha="center", va="center",
             fontsize=22, fontweight="bold", color=COLORS["text"])
    ax3.text(0, -0.12, "mean", ha="center", va="center",
             fontsize=9, color=COLORS["text_light"])
    ax3.set_title("Level Distribution", fontsize=12, fontweight="bold",
                   color=COLORS["text"], pad=10)

    # ── Row 3, Center: Metric Contribution Bars ───────────────
    ax4 = fig.add_subplot(gs[2, 1])
    _apply_style(fig, ax4)

    breakdown = analysis.get("metric_breakdown", {})
    if breakdown:
        metric_keys = ["bfi", "ear", "bad"]
        bar_labels = ["BFI\n(Brow Furrow)", "EAR\n(Eye/Blink)", "BAD\n(Asymmetry)"]
        weighted_vals = [breakdown[m]["weighted_contribution"] for m in metric_keys]
        pcts = [breakdown[m]["contribution_pct"] for m in metric_keys]
        bar_colors = [COLORS[m] for m in metric_keys]

        bars = ax4.barh(bar_labels, weighted_vals, color=bar_colors,
                        height=0.5, alpha=0.85)
        ax4.set_xlim(0, max(weighted_vals) * 1.6 if weighted_vals else 1)
        for bar, val, pct in zip(bars, weighted_vals, pcts):
            ax4.text(
                val + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f} ({pct:.1f}%)",
                va="center", fontsize=9, color=COLORS["text"],
            )
        ax4.set_xlabel("Weighted Contribution", fontsize=9, color=COLORS["text"])
        ax4.set_title("Metric Breakdown", fontsize=12, fontweight="bold",
                       color=COLORS["text"], pad=10)

    # ── Row 3, Right: Key Stats Panel ─────────────────────────
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.set_facecolor(COLORS["bg"])
    ax5.axis("off")

    summary = analysis.get("summary", {})
    ss = summary.get("stress_score", {})
    temporal = analysis.get("temporal_profile", {})
    peaks = analysis.get("peaks_and_valleys", {})

    # Build stats text
    stats_lines = [
        ("VIDEO INFO", None),
        (f"Duration: {duration_s:.1f}s  •  {total_frames} frames", COLORS["text"]),
        (f"Resolution: {data['metadata'].get('video_resolution', '?')}", COLORS["text"]),
        (f"Face Detection: {summary.get('detection_rate', 0):.1f}%", COLORS["text"]),
        ("", None),
        ("STRESS SCORE", None),
        (f"Mean: {ss.get('mean', 0):.4f}  ±  {ss.get('std', 0):.4f}", COLORS["text"]),
        (f"95% CI: [{ss.get('ci_95_lower', 0):.4f}, {ss.get('ci_95_upper', 0):.4f}]", COLORS["text"]),
        (f"Median: {ss.get('median', 0):.4f}", COLORS["text"]),
        (f"Range: [{ss.get('min', 0):.4f}, {ss.get('max', 0):.4f}]", COLORS["text"]),
        (f"IQR: [{ss.get('p25', 0):.4f}, {ss.get('p75', 0):.4f}]", COLORS["text"]),
        ("", None),
        ("DYNAMICS", None),
        (f"Overall trend: {temporal.get('overall_trend', '?')}", COLORS["text"]),
    ]

    # Blink info from metric breakdown
    ear_sub = breakdown.get("ear", {}).get("sub_metrics", {})
    if ear_sub:
        stats_lines.append(
            (f"Blink rate: {ear_sub.get('blink_rate_mean', 0):.1f}/min "
             f"(max: {ear_sub.get('blink_rate_max', 0):.1f})",
             COLORS["text"])
        )
        dur = ear_sub.get("avg_blink_duration_ms", 0)
        if dur > 0:
            stats_lines.append(
                (f"Avg blink duration: {dur:.0f} ms", COLORS["text"])
            )

    # Top peak
    peak_list = peaks.get("peaks", [])
    if peak_list:
        p = peak_list[0]
        stats_lines.append(("", None))
        stats_lines.append(("PEAK STRESS", None))
        stats_lines.append(
            (f"t={p['timestamp_s']:.1f}s  score={p['score_smoothed']:.3f}  "
             f"[{p['dominant_metric'].upper()}]",
             COLORS["stress_line"])
        )

    y_pos = 0.95
    for text, color in stats_lines:
        if color is None and text:
            # Section header
            ax5.text(0.05, y_pos, text, transform=ax5.transAxes,
                     fontsize=10, fontweight="bold", color=COLORS["text"],
                     fontfamily="monospace")
            y_pos -= 0.005
            ax5.plot([0.05, 0.95], [y_pos, y_pos],
                     color=COLORS["grid"], linewidth=0.8,
                     transform=ax5.transAxes, clip_on=False)
            y_pos -= 0.045
        elif text == "":
            y_pos -= 0.03
        else:
            ax5.text(0.08, y_pos, text, transform=ax5.transAxes,
                     fontsize=9, color=color, fontfamily="monospace")
            y_pos -= 0.055

    ax5.set_title("Key Statistics", fontsize=12, fontweight="bold",
                   color=COLORS["text"], pad=10)

    return fig


# ──────────────────────────────────────────────────────────────
# FER-specific charts
# ──────────────────────────────────────────────────────────────


def _detected_fer_frames(data: dict) -> list[dict]:
    """Frame con volto rilevato per uno schema FER (emotions non null)."""
    return [
        f for f in data["frames"]
        if f.get("face_detected") and f.get("emotions") is not None
    ]


def plot_emotion_timeline_fer(
    data: dict,
    smoothing_window: int = 15,
    figsize: tuple[float, float] = (14, 5),
) -> "plt.Figure":
    """
    Andamento temporale delle 7 emozioni FER (sovrapposte come linee).

    Le emozioni "di stress" (angry, fear, disgust) sono evidenziate, le altre
    sono disegnate più sottili / trasparenti.
    """
    _check_matplotlib()

    fps = data["metadata"].get("video_fps", 30.0)
    frames = _detected_fer_frames(data)

    if not frames:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No face detected", ha="center", va="center")
        return fig

    times = np.array([f["frame_id"] / fps for f in frames])
    kernel = np.ones(smoothing_window) / smoothing_window

    fig, ax = plt.subplots(figsize=figsize)
    _apply_style(fig, ax)

    for name in FER_EMOTION_NAMES:
        vals = np.array([f["emotions"].get(name, 0.0) for f in frames])
        if len(vals) >= smoothing_window:
            vals = np.convolve(vals, kernel, mode="same")
        is_stress = name in FER_STRESS_EMOTIONS
        ax.plot(
            times, vals,
            color=EMOTION_COLORS[name],
            linewidth=1.8 if is_stress else 1.0,
            alpha=1.0 if is_stress else 0.55,
            label=name + (" ★" if is_stress else ""),
        )

    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time (s)", fontsize=10, color=COLORS["text"])
    ax.set_ylabel("Emotion intensity", fontsize=10, color=COLORS["text"])
    ax.set_title(
        f"FER Emotion Timeline — {data['metadata'].get('video_file', '?')}",
        fontsize=13, fontweight="bold", color=COLORS["text"], pad=15,
    )
    ax.legend(
        fontsize=8, loc="upper left", framealpha=0.9,
        ncol=4, title="★ = stress emotion",
    )

    fig.tight_layout()
    return fig


def plot_emotion_contributions_fer(
    data: dict,
    smoothing_window: int = 15,
    figsize: tuple[float, float] = (14, 5),
) -> "plt.Figure":
    """
    Stacked area: contributo pesato di angry/fear/disgust allo stress score.
    """
    _check_matplotlib()

    fps = data["metadata"].get("video_fps", 30.0)
    frames = _detected_fer_frames(data)

    if not frames:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No face detected", ha="center", va="center")
        return fig

    times = np.array([f["frame_id"] / fps for f in frames])
    weights = (frames[0].get("stress") or {}).get("weights_used", {})

    kernel = np.ones(smoothing_window) / smoothing_window
    series: dict[str, np.ndarray] = {}
    for name in FER_STRESS_EMOTIONS:
        w = float(weights.get(name, 1.0))
        arr = np.array([f["emotions"].get(name, 0.0) * w for f in frames])
        if len(arr) >= smoothing_window:
            arr = np.convolve(arr, kernel, mode="same")
        series[name] = arr

    fig, ax = plt.subplots(figsize=figsize)
    _apply_style(fig, ax)

    ax.stackplot(
        times,
        *[series[n] for n in FER_STRESS_EMOTIONS],
        labels=[
            f"{n} (w={weights.get(n, 1.0):g})" for n in FER_STRESS_EMOTIONS
        ],
        colors=[EMOTION_COLORS[n] for n in FER_STRESS_EMOTIONS],
        alpha=0.85,
    )

    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time (s)", fontsize=10, color=COLORS["text"])
    ax.set_ylabel("Weighted contribution", fontsize=10, color=COLORS["text"])
    ax.set_title(
        f"FER Emotion Contributions — {data['metadata'].get('video_file', '?')}",
        fontsize=13, fontweight="bold", color=COLORS["text"], pad=15,
    )
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)

    fig.tight_layout()
    return fig


def plot_emotion_bars_fer(
    analysis: dict,
    figsize: tuple[float, float] = (11, 6),
) -> "plt.Figure":
    """
    Doppio bar chart orizzontale: media grezza di ciascuna delle 7 emozioni
    + contributo pesato allo stress (solo angry/fear/disgust).
    """
    _check_matplotlib()

    breakdown = analysis.get("emotion_breakdown", {})
    if not breakdown:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    names      = list(FER_EMOTION_NAMES)
    raw_means  = [breakdown[n]["raw_mean"] for n in names]
    weighted   = [breakdown[n]["weighted_contribution"] for n in names]
    pcts       = [breakdown[n]["contribution_pct"] for n in names]
    bar_colors = [EMOTION_COLORS[n] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    _apply_style(fig, ax1)
    _apply_style(fig, ax2)

    # Sinistra: media grezza di ogni emozione
    bars1 = ax1.barh(names, raw_means, color=bar_colors, height=0.6, alpha=0.85)
    ax1.set_xlim(0, max(1.0, max(raw_means) * 1.2))
    ax1.set_xlabel("Raw mean value", fontsize=10, color=COLORS["text"])
    ax1.set_title("Mean emotion intensity",
                  fontsize=12, fontweight="bold", color=COLORS["text"])
    for bar, val in zip(bars1, raw_means):
        ax1.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=9, color=COLORS["text"])

    # Destra: contributo pesato allo stress
    bars2 = ax2.barh(names, weighted, color=bar_colors, height=0.6, alpha=0.85)
    ax2.set_xlim(0, max(0.01, max(weighted) * 1.5))
    ax2.set_xlabel("Weighted contribution", fontsize=10, color=COLORS["text"])
    ax2.set_title("Contribution to stress score",
                  fontsize=12, fontweight="bold", color=COLORS["text"])
    for bar, val, pct in zip(bars2, weighted, pcts):
        label = f"{val:.3f} ({pct:.1f}%)" if val > 0 else "—"
        ax2.text(val + max(weighted) * 0.02, bar.get_y() + bar.get_height() / 2,
                 label, va="center", fontsize=9, color=COLORS["text"])

    fig.suptitle(
        f"FER Emotion Breakdown — {analysis.get('source_file', '?')}",
        fontsize=14, fontweight="bold", color=COLORS["text"], y=1.02,
    )
    fig.tight_layout()
    return fig


def plot_summary_dashboard_fer(
    data: dict,
    analysis: dict,
    smoothing_window: int = 15,
    figsize: tuple[float, float] = (20, 14),
) -> "plt.Figure":
    """
    Dashboard riepilogativa per il pipeline FER.

    Layout (3 righe):
      Row 1: Stress timeline                                          [full]
      Row 2: Emotion contributions stacked area                       [full]
      Row 3: Level donut | Emotion bars | Key stats panel
    """
    _check_matplotlib()
    from matplotlib.gridspec import GridSpec

    fps = data["metadata"].get("video_fps", 30.0)
    frames = _detected_fer_frames(data)
    video_name = data["metadata"].get("video_file", "Unknown")
    total_frames = data["metadata"].get(
        "total_frames_processed", len(data["frames"])
    )
    duration_s = total_frames / fps

    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor(COLORS["bg"])

    gs = GridSpec(
        3, 3, figure=fig,
        height_ratios=[1.0, 0.8, 1.0],
        hspace=0.35, wspace=0.35,
    )

    fig.suptitle(
        f"FER Stress Analysis — {video_name}",
        fontsize=18, fontweight="bold", color=COLORS["text"], y=0.98,
    )

    if not frames:
        ax = fig.add_subplot(gs[:, :])
        ax.text(0.5, 0.5, "No face detected in video",
                ha="center", va="center",
                fontsize=16, color=COLORS["text_light"])
        ax.set_facecolor(COLORS["bg"])
        ax.axis("off")
        return fig

    times = np.array([f["frame_id"] / fps for f in frames])
    scores = np.array([f["stress"]["score"] for f in frames])
    kernel = np.ones(smoothing_window) / smoothing_window
    smoothed = (
        np.convolve(scores, kernel, mode="same")
        if len(scores) >= smoothing_window else scores
    )

    # ── Row 1: Stress Timeline ────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    _apply_style(fig, ax1)

    band_levels = [
        (0.00, 0.25, COLORS["LOW"], "LOW", 0.08),
        (0.25, 0.50, COLORS["MODERATE"], "MOD", 0.08),
        (0.50, 0.75, COLORS["HIGH"], "HIGH", 0.08),
        (0.75, 1.00, COLORS["VERY_HIGH"], "V.HIGH", 0.08),
    ]
    for y_low, y_high, color, label, alpha in band_levels:
        ax1.axhspan(y_low, y_high, color=color, alpha=alpha)
        ax1.text(
            times[-1] * 1.005, (y_low + y_high) / 2, label,
            fontsize=7, color=color, va="center",
            alpha=0.9, fontweight="bold",
        )

    ax1.plot(times, scores, color=COLORS["stress_line"], alpha=0.15, linewidth=0.5)
    ax1.plot(times, smoothed, color=COLORS["stress_line"], linewidth=2.0, label="Smoothed")
    ax1.fill_between(times, 0, smoothed, color=COLORS["stress_fill"], alpha=0.1)

    ax1.set_xlim(times[0], times[-1])
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Time (s)", fontsize=10, color=COLORS["text"])
    ax1.set_ylabel("Stress Score", fontsize=10, color=COLORS["text"])
    ax1.set_title("Stress Timeline", fontsize=12, fontweight="bold",
                  color=COLORS["text"], loc="left", pad=8)
    ax1.legend(fontsize=8, loc="upper right", framealpha=0.9)

    # ── Row 2: Emotion Contributions Stacked Area ─────────────
    ax2 = fig.add_subplot(gs[1, :])
    _apply_style(fig, ax2)

    weights = (frames[0].get("stress") or {}).get("weights_used", {})
    series: dict[str, np.ndarray] = {}
    for name in FER_STRESS_EMOTIONS:
        w = float(weights.get(name, 1.0))
        arr = np.array([f["emotions"].get(name, 0.0) * w for f in frames])
        if len(arr) >= smoothing_window:
            arr = np.convolve(arr, kernel, mode="same")
        series[name] = arr

    ax2.stackplot(
        times,
        *[series[n] for n in FER_STRESS_EMOTIONS],
        labels=[
            f"{n} (w={weights.get(n, 1.0):g})" for n in FER_STRESS_EMOTIONS
        ],
        colors=[EMOTION_COLORS[n] for n in FER_STRESS_EMOTIONS],
        alpha=0.85,
    )
    stack_total = sum(series.values())
    ax2.set_xlim(times[0], times[-1])
    ax2.set_ylim(0, max(0.6, float(np.max(stack_total)) * 1.1))
    ax2.set_xlabel("Time (s)", fontsize=10, color=COLORS["text"])
    ax2.set_ylabel("Weighted Contribution", fontsize=10, color=COLORS["text"])
    ax2.set_title("Emotion Contributions Over Time",
                  fontsize=12, fontweight="bold",
                  color=COLORS["text"], loc="left", pad=8)
    ax2.legend(fontsize=8, loc="upper right", framealpha=0.9)

    # ── Row 3, Left: Level Distribution Donut ─────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.set_facecolor(COLORS["bg"])

    levels_list = [f["stress"]["level"] for f in frames]
    level_order = ["LOW", "MODERATE", "HIGH", "VERY_HIGH"]
    counts = [levels_list.count(l) for l in level_order]

    pie_labels, pie_sizes, pie_colors = [], [], []
    for l, c in zip(level_order, counts):
        if c > 0:
            pie_labels.append(l)
            pie_sizes.append(c)
            pie_colors.append(COLORS[l])

    if pie_sizes:
        wedges, texts, autotexts = ax3.pie(
            pie_sizes, labels=pie_labels, colors=pie_colors,
            autopct="%1.1f%%", startangle=90, pctdistance=0.80,
            wedgeprops={"width": 0.4, "edgecolor": "white", "linewidth": 2},
        )
        for t in texts:
            t.set_fontsize(9)
            t.set_color(COLORS["text"])
            t.set_fontweight("bold")
        for t in autotexts:
            t.set_fontsize(8)
            t.set_color(COLORS["text"])

    mean_score = float(np.mean(scores))
    ax3.text(0, 0.04, f"{mean_score:.2f}", ha="center", va="center",
             fontsize=22, fontweight="bold", color=COLORS["text"])
    ax3.text(0, -0.12, "mean", ha="center", va="center",
             fontsize=9, color=COLORS["text_light"])
    ax3.set_title("Level Distribution", fontsize=12, fontweight="bold",
                  color=COLORS["text"], pad=10)

    # ── Row 3, Center: Emotion Bars ───────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    _apply_style(fig, ax4)

    breakdown = analysis.get("emotion_breakdown", {})
    if breakdown:
        names    = list(FER_EMOTION_NAMES)
        weighted = [breakdown[n]["weighted_contribution"] for n in names]
        raw_means = [breakdown[n]["raw_mean"] for n in names]
        colors_  = [EMOTION_COLORS[n] for n in names]

        bars = ax4.barh(names, weighted, color=colors_, height=0.6, alpha=0.85)
        ax4.set_xlim(0, max(0.01, max(weighted) * 1.6))
        for bar, w, raw in zip(bars, weighted, raw_means):
            label = f"{w:.3f}" if w > 0 else f"(raw={raw:.2f})"
            ax4.text(
                w + max(weighted) * 0.02,
                bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=8, color=COLORS["text"],
            )
        ax4.set_xlabel("Weighted contribution", fontsize=9, color=COLORS["text"])
        ax4.set_title("Emotion breakdown",
                      fontsize=12, fontweight="bold",
                      color=COLORS["text"], pad=10)

    # ── Row 3, Right: Key Stats Panel ─────────────────────────
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.set_facecolor(COLORS["bg"])
    ax5.axis("off")

    summary = analysis.get("summary", {})
    ss = summary.get("stress_score", {})
    temporal = analysis.get("temporal_profile", {})
    peaks = analysis.get("peaks_and_valleys", {})

    stats_lines = [
        ("VIDEO INFO", None),
        (f"Duration: {duration_s:.1f}s  •  {total_frames} frames", COLORS["text"]),
        (f"Resolution: {data['metadata'].get('video_resolution', '?')}", COLORS["text"]),
        (f"Method: FER (emotion-based)", COLORS["text"]),
        (f"Face Detection: {summary.get('detection_rate', 0):.1f}%", COLORS["text"]),
        ("", None),
        ("STRESS SCORE", None),
        (f"Mean: {ss.get('mean', 0):.4f}  ±  {ss.get('std', 0):.4f}", COLORS["text"]),
        (f"95% CI: [{ss.get('ci_95_lower', 0):.4f}, {ss.get('ci_95_upper', 0):.4f}]", COLORS["text"]),
        (f"Median: {ss.get('median', 0):.4f}", COLORS["text"]),
        (f"Range: [{ss.get('min', 0):.4f}, {ss.get('max', 0):.4f}]", COLORS["text"]),
        (f"IQR: [{ss.get('p25', 0):.4f}, {ss.get('p75', 0):.4f}]", COLORS["text"]),
        ("", None),
        ("DYNAMICS", None),
        (f"Overall trend: {temporal.get('overall_trend', '?')}", COLORS["text"]),
    ]

    if breakdown:
        top_emo = max(
            FER_STRESS_EMOTIONS,
            key=lambda n: breakdown[n]["weighted_contribution"],
        )
        stats_lines.append(
            (f"Top stress emotion: {top_emo} "
             f"({breakdown[top_emo]['contribution_pct']:.1f}%)",
             COLORS["text"])
        )

    peak_list = peaks.get("peaks", [])
    if peak_list:
        p = peak_list[0]
        stats_lines.append(("", None))
        stats_lines.append(("PEAK STRESS", None))
        stats_lines.append(
            (f"t={p['timestamp_s']:.1f}s  score={p['score_smoothed']:.3f}  "
             f"[{str(p.get('dominant_metric', '?')).upper()}]",
             COLORS["stress_line"])
        )

    y_pos = 0.95
    for text, color in stats_lines:
        if color is None and text:
            ax5.text(0.05, y_pos, text, transform=ax5.transAxes,
                     fontsize=10, fontweight="bold", color=COLORS["text"],
                     fontfamily="monospace")
            y_pos -= 0.005
            ax5.plot([0.05, 0.95], [y_pos, y_pos],
                     color=COLORS["grid"], linewidth=0.8,
                     transform=ax5.transAxes, clip_on=False)
            y_pos -= 0.045
        elif text == "":
            y_pos -= 0.03
        else:
            ax5.text(0.08, y_pos, text, transform=ax5.transAxes,
                     fontsize=9, color=color, fontfamily="monospace")
            y_pos -= 0.055

    ax5.set_title("Key Statistics", fontsize=12, fontweight="bold",
                  color=COLORS["text"], pad=10)

    return fig


# ──────────────────────────────────────────────────────────────
# Generate all charts at once (dispatcher fer / landmark)
# ──────────────────────────────────────────────────────────────


def generate_all_charts(
    data: dict,
    analysis: dict,
    output_dir: str | Path,
    fmt: str = "png",
    dpi: int = 150,
    summary_only: bool = True,
) -> list[Path]:
    """
    Generate charts for a single GAFFE JSON analysis.

    Automatically selects the correct chart set based on the detection method
    (landmark or fer) recorded in the analysis.

    By default only the summary dashboard is generated (one file). Pass
    ``summary_only=False`` (via the ``--all-charts`` CLI flag) to also produce
    the individual per-metric / per-emotion charts.

    Args:
        data:         Loaded GAFFE JSON (raw frames).
        analysis:     Output of analyze_single_video().
        output_dir:   Directory where chart files are saved (created if needed).
        fmt:          Image format — 'png', 'pdf', or 'svg'.
        dpi:          Resolution in dots per inch.
        summary_only: If True (default), generate only the dashboard chart.
                      If False, also generate individual metric/emotion charts.

    Returns:
        List of Path objects for every file that was saved.
    """
    _check_matplotlib()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(analysis.get("source_file", "video")).stem.replace("_gaffe", "")

    # Detect pipeline method: check analysis dict, then metadata, then frame sniffing.
    method = analysis.get("detection_method")
    if not method:
        method = data.get("metadata", {}).get("detection_method")
    if not method:
        for f in data.get("frames", []):
            if f.get("face_detected"):
                if "emotions" in f and f.get("emotions") is not None:
                    method = "fer"
                elif "metrics" in f and f.get("metrics") is not None:
                    method = "landmark"
                break

    saved: list[Path] = []

    def _save(fig, name: str) -> None:
        path = output_dir / f"{stem}_{name}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        saved.append(path)

    if method == "fer":
        # Summary dashboard (always generated)
        _save(plot_summary_dashboard_fer(data, analysis), "dashboard")
        # Individual charts (only when summary_only=False)
        if not summary_only:
            _save(plot_stress_timeline(data),           "stress_timeline")
            _save(plot_level_distribution(data),        "level_distribution")
            _save(plot_emotion_timeline_fer(data),      "emotion_timeline")
            _save(plot_emotion_contributions_fer(data), "emotion_contributions")
            _save(plot_emotion_bars_fer(analysis),      "emotion_bars")
    else:
        # Summary dashboard (always generated)
        _save(plot_summary_dashboard(data, analysis), "dashboard")
        # Individual charts (only when summary_only=False)
        if not summary_only:
            _save(plot_stress_timeline(data),      "stress_timeline")
            _save(plot_level_distribution(data),   "level_distribution")
            _save(plot_metric_contributions(data), "metric_contributions")
            _save(plot_blink_rate(data),           "blink_rate")
            _save(plot_metric_bars(analysis),      "metric_bars")

    return saved
