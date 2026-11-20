"""Plot the deployment results summarized in the README."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROMPT_TOKEN_REDUCTION = 0.67
THROUGHPUT_MULTIPLIER = 3.0


def main():
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    panels = [
        ("Prompt tokens", [100, 100 * (1 - PROMPT_TOKEN_REDUCTION)],
         "% of baseline", "%", 115),
        ("Throughput", [1, THROUGHPUT_MULTIPLIER],
         "× baseline", "×", 3.5),
    ]

    for ax, (title, values, unit, suffix, limit) in zip(axes, panels):
        bars = ax.bar(["Baseline RAG", "RefRag-lite"], values)
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.set_ylim(0, limit)
        ax.bar_label(bars, labels=[f"{value:g}{suffix}" for value in values],
                     padding=3)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    output_dir = Path(__file__).resolve().parent
    fig.savefig(output_dir / "deployment_results.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
