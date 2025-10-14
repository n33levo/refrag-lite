"""Plotting utilities for evaluation reports."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")  # Non-interactive backend
sns.set_theme(style="darkgrid")


def plot_quality_vs_cost(
    results: Dict[str, List[dict]],
    output_path: str,
    metrics: List[str] = ["em", "f1"],
    figsize: Tuple[int, int] = (12, 5),
    dpi: int = 300,
) -> None:
    """Plot quality metrics vs token cost.

    Args:
        results: Dict mapping method name to list of result dicts
        output_path: Output path for plot
        metrics: Metrics to plot
        figsize: Figure size
        dpi: DPI for saved figure

    Example:
        >>> results = {
        ...     "baseline": [{"em": 0.4, "f1": 0.5, "tokens": 1000}],
        ...     "ours": [{"em": 0.39, "f1": 0.48, "tokens": 500}],
        ... }
        >>> plot_quality_vs_cost(results, "plot.png")
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        for method_name, method_results in results.items():
            tokens = [r["tokens"] for r in method_results if "tokens" in r]
            values = [r[metric] for r in method_results if metric in r]

            if tokens and values:
                ax.scatter(tokens, values, label=method_name, s=100, alpha=0.7)

        ax.set_xlabel("Average Tokens", fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(f"{metric.upper()} vs Token Cost", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_ttft_vs_quality(
    results: Dict[str, List[dict]],
    output_path: str,
    metrics: List[str] = ["em", "f1"],
    figsize: Tuple[int, int] = (12, 5),
    dpi: int = 300,
) -> None:
    """Plot TTFT vs quality metrics.

    Args:
        results: Dict mapping method name to list of result dicts
        output_path: Output path for plot
        metrics: Metrics to plot
        figsize: Figure size
        dpi: DPI for saved figure
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        for method_name, method_results in results.items():
            ttft = [r["ttft"] for r in method_results if "ttft" in r]
            values = [r[metric] for r in method_results if metric in r]

            if ttft and values:
                ax.scatter(ttft, values, label=method_name, s=100, alpha=0.7)

        ax.set_xlabel("TTFT (seconds)", fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(f"{metric.upper()} vs TTFT", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_pareto_frontier(
    results: Dict[str, List[dict]],
    output_path: str,
    x_metric: str = "tokens",
    y_metric: str = "em",
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 300,
) -> None:
    """Plot Pareto frontier of cost vs quality.

    Args:
        results: Dict mapping method name to list of result dicts
        output_path: Output path for plot
        x_metric: Metric for x-axis (cost)
        y_metric: Metric for y-axis (quality)
        figsize: Figure size
        dpi: DPI for saved figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    for method_name, method_results in results.items():
        x_vals = [r[x_metric] for r in method_results if x_metric in r and y_metric in r]
        y_vals = [r[y_metric] for r in method_results if x_metric in r and y_metric in r]

        if x_vals and y_vals:
            ax.scatter(x_vals, y_vals, label=method_name, s=100, alpha=0.7)

            # Draw line connecting points if multiple
            if len(x_vals) > 1:
                sorted_pairs = sorted(zip(x_vals, y_vals))
                ax.plot([p[0] for p in sorted_pairs], [p[1] for p in sorted_pairs], alpha=0.4)

    ax.set_xlabel(x_metric.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel(y_metric.replace("_", " ").upper(), fontsize=12)
    ax.set_title("Pareto Frontier: Quality vs Cost", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_ablation_heatmap(
    results: pd.DataFrame,
    output_path: str,
    x_var: str,
    y_var: str,
    z_var: str,
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 300,
) -> None:
    """Plot ablation study as heatmap.

    Args:
        results: DataFrame with ablation results
        output_path: Output path for plot
        x_var: Variable for x-axis
        y_var: Variable for y-axis
        z_var: Variable for color (value)
        figsize: Figure size
        dpi: DPI for saved figure
    """
    # Pivot data for heatmap
    pivot = results.pivot(index=y_var, columns=x_var, values=z_var)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax, cbar_kws={"label": z_var})

    ax.set_xlabel(x_var.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel(y_var.replace("_", " ").title(), fontsize=12)
    ax.set_title(f"Ablation: {z_var.upper()} vs {x_var} and {y_var}", fontsize=14)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_training_curves(
    log_file: str,
    output_path: str,
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 300,
) -> None:
    """Plot training curves from CSV log.

    Args:
        log_file: Path to CSV log file
        output_path: Output path for plot
        metrics: Metrics to plot (None = all)
        figsize: Figure size
        dpi: DPI for saved figure
    """
    df = pd.read_csv(log_file)

    # Filter to only metric rows
    df = df[df["type"] == "metric"]

    if metrics is None:
        # Auto-detect metrics (exclude 'type' and 'step')
        metrics = [col for col in df.columns if col not in ["type", "step"]]

    # Plot each metric
    n_metrics = len(metrics)
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_metrics > 1 else [axes]

    for i, metric in enumerate(metrics):
        if metric in df.columns:
            ax = axes[i]
            ax.plot(df["step"], df[metric])
            ax.set_xlabel("Step", fontsize=10)
            ax.set_ylabel(metric.replace("_", " ").title(), fontsize=10)
            ax.set_title(metric.replace("_", " ").title(), fontsize=12)
            ax.grid(True, alpha=0.3)

    # Hide extra subplots
    for i in range(n_metrics, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_comparison_bar(
    results: Dict[str, Dict[str, float]],
    output_path: str,
    metrics: List[str] = ["em", "f1", "tokens"],
    figsize: Tuple[int, int] = (12, 6),
    dpi: int = 300,
) -> None:
    """Plot bar chart comparing methods across metrics.

    Args:
        results: Dict mapping method name to metric dict
        output_path: Output path for plot
        metrics: Metrics to compare
        figsize: Figure size
        dpi: DPI for saved figure

    Example:
        >>> results = {
        ...     "baseline": {"em": 0.4, "f1": 0.5, "tokens": 1000},
        ...     "ours": {"em": 0.39, "f1": 0.48, "tokens": 500},
        ... }
        >>> plot_comparison_bar(results, "bar.png")
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]

    methods = list(results.keys())
    x = np.arange(len(methods))

    for ax, metric in zip(axes, metrics):
        values = [results[m].get(metric, 0) for m in methods]
        ax.bar(x, values, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha="right")
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
        ax.set_title(metric.replace("_", " ").title(), fontsize=14)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
