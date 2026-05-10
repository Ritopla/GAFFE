import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_comparison(json_paths: list[str], output_path: str):
    """Plot comparisons of aggregated batch analysis results."""
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    labels = []
    stress_means = []
    stress_errors = []
    
    bfi_means = []
    ear_means = []
    bad_means = []
    
    all_stress_distributions = []
    
    colors = ['#2ca02c', '#d62728', '#1f77b4', '#ff7f0e']
    
    for i, path_str in enumerate(json_paths):
        path = Path(path_str)
        if not path.exists():
            print(f"File not found: {path}")
            continue
            
        with open(path) as f:
            data = json.load(f)
            
        # Get label from directory name if possible, else filename
        label = path.parent.name if path.parent.name != "results" else f"Dataset {i+1}"
        labels.append(label)
        
        # Get the 'all' group aggregation
        agg = data.get("group_aggregation", {}).get("all", {})
        if not agg:
            continue
            
        ss = agg.get("stress_score", {})
        metrics = agg.get("metrics", {})
        
        stress_means.append(ss.get("mean", 0))
        # Use 95% CI for error bars if available, otherwise std
        err = ss.get("ci_95_upper", ss.get("mean", 0)) - ss.get("mean", 0)
        if err == 0:
            err = ss.get("std", 0)
        stress_errors.append(err)
        
        bfi_means.append(metrics.get("bfi", {}).get("mean", 0))
        ear_means.append(metrics.get("ear_stress", {}).get("mean", 0))
        bad_means.append(metrics.get("bad", {}).get("mean", 0))
        
        # Collect distributions for histogram
        dist = []
        for vid in data.get("per_video", []):
            if vid.get("analysis", {}).get("summary", {}).get("stress_score"):
                dist.append(vid["analysis"]["summary"]["stress_score"]["mean"])
        all_stress_distributions.append((label, dist, colors[i % len(colors)]))

    # 1. Overall Stress Score Comparison (Bar chart)
    ax = axes[0, 0]
    x = np.arange(len(labels))
    ax.bar(x, stress_means, yerr=stress_errors, capsize=10, color=colors[:len(labels)], alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title('Mean Stress Score (with 95% CI)', fontsize=14)
    ax.set_ylabel('Stress Score')
    ax.set_ylim(0, max(stress_means) * 1.5 if stress_means else 1)
    
    # 2. Metric Breakdown (Grouped Bar Chart)
    ax = axes[0, 1]
    width = 0.25
    x_bfi = x - width
    x_ear = x
    x_bad = x + width
    
    ax.bar(x_bfi, bfi_means, width, label='BFI (Blink Flow)', color='#9467bd', alpha=0.7)
    ax.bar(x_ear, ear_means, width, label='EAR (Eye Aspect)', color='#8c564b', alpha=0.7)
    ax.bar(x_bad, bad_means, width, label='BAD (Asymmetry)', color='#e377c2', alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title('Sub-Metric Contributions', fontsize=14)
    ax.set_ylabel('Metric Score')
    ax.legend()
    
    # 3. Stress Score Distribution (Histogram)
    ax = axes[1, 0]
    for label, dist, color in all_stress_distributions:
        ax.hist(dist, bins=15, alpha=0.5, label=label, color=color, density=True)
    ax.set_title('Stress Score Distribution', fontsize=14)
    ax.set_xlabel('Mean Stress Score')
    ax.set_ylabel('Density')
    ax.legend()
    
    # 4. Boxplot for better distribution view
    ax = axes[1, 1]
    plot_data = [dist for _, dist, _ in all_stress_distributions]
    ax.boxplot(plot_data, labels=labels, patch_artist=True, 
               boxprops=dict(facecolor='lightblue', color='blue'),
               medianprops=dict(color='red', linewidth=2))
    ax.set_title('Stress Score Boxplot', fontsize=14)
    ax.set_ylabel('Stress Score')
    
    plt.suptitle('GAFFE Batch Analysis Results Comparison', fontsize=18, y=1.02)
    plt.tight_layout()
    
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot comparison of GAFFE batch results")
    parser.add_argument("inputs", nargs="+", help="Paths to batch_analysis.json files")
    parser.add_argument("-o", "--output", default="results/comparison_plot.png", 
                        help="Output image path")
    args = parser.parse_args()
    
    plot_comparison(args.inputs, args.output)
