"""Draw the result charts from the CSVs in results/.

    python src/plot.py

Needs only pandas and matplotlib (requirements-plot.txt), so it also runs
locally on CSVs copied off the Colab runtime.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # file output only; also overrides Colab's inline backend in subprocesses

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from common import RESULTS_DIR

# Light chart palette. vLLM is the story, so it gets the one accent hue;
# the HF baselines are context and stay gray. Every bar is also labeled by
# name and value, so color never carries identity alone.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
ACCENT = "#2a78d6"
CONTEXT = "#898781"

SUBTITLE = "Qwen2.5-1.5B-Instruct · fp16 · free Colab T4 · 128 greedy output tokens per prompt"


def compact(value: float, _pos: object = None) -> str:
    """Axis tick label: 0, 500, 1K, 1.5K, ..."""
    return f"{value / 1000:g}K" if value >= 1000 else f"{value:g}"


def new_figure(width: float, height: float, ncols: int = 1) -> tuple[Figure, list[Axes]]:
    """A figure on the chart surface with recessive axes: hairline y-grid, baseline only."""
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), dpi=200, facecolor=SURFACE, squeeze=False)
    for ax in axes[0]:
        ax.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE)
        ax.grid(axis="y", color=GRIDLINE, linewidth=1)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY, length=0, labelsize=9)
        ax.yaxis.set_major_formatter(FuncFormatter(compact))
    return fig, list(axes[0])


def add_titles(fig: Figure, title: str, subtitle: str) -> None:
    """Left-aligned title and subtitle above the plot area."""
    fig.text(0.02, 0.965, title, ha="left", va="top", fontsize=14, fontweight="semibold", color=INK)
    fig.text(0.02, 0.905, subtitle, ha="left", va="top", fontsize=9, color=INK_SECONDARY)


def method_label(method: str, batch_size: object) -> str:
    """Two-line x-axis label for an offline method."""
    if method == "hf_naive":
        return "HF transformers\none prompt at a time"
    if method == "hf_batched":
        return f"HF transformers\nstatic batches of {int(batch_size)}"
    if method == "vllm":
        return "vLLM\ncontinuous batching"
    return method


def plot_offline(df: pd.DataFrame) -> Figure:
    """Bar chart of mean output tokens/s per method, with 95% CI error bars."""
    fig, (ax,) = new_figure(8, 5)
    x = list(range(len(df)))
    colors = [ACCENT if m == "vllm" else CONTEXT for m in df["method"]]
    ax.bar(x, df["mean_tok_s"], width=0.4, color=colors, zorder=2)
    ax.errorbar(
        x, df["mean_tok_s"],
        yerr=[df["mean_tok_s"] - df["ci95_low"], df["ci95_high"] - df["mean_tok_s"]],
        fmt="none", ecolor=INK, elinewidth=1, capsize=4, zorder=3,
    )

    naive = df.loc[df["method"] == "hf_naive", "mean_tok_s"]
    for xi, row in zip(x, df.itertuples()):
        show_speedup = not naive.empty and row.method != "hf_naive"
        ax.annotate(
            f"{row.mean_tok_s:,.0f} tok/s", (xi, row.ci95_high), xytext=(0, 18 if show_speedup else 5),
            textcoords="offset points", ha="center", va="bottom", fontsize=11, fontweight="semibold", color=INK,
        )
        if show_speedup:
            speedup = row.mean_tok_s / naive.iloc[0]
            ax.annotate(
                f"{speedup:.{0 if speedup >= 10 else 1}f}× vs one-at-a-time", (xi, row.ci95_high),
                xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color=INK_SECONDARY,
            )

    ax.set_xticks(x, [method_label(m, b) for m, b in zip(df["method"], df["batch_size"])])
    ax.tick_params(axis="x", labelsize=10, labelcolor=INK)
    ax.set_ylabel("output tokens / second", color=INK_SECONDARY, fontsize=9)
    ax.set_ylim(0, df["ci95_high"].max() * 1.2)
    add_titles(fig, "Offline throughput: vLLM vs Hugging Face transformers", SUBTITLE + " · mean of 5 runs, 95% CI")
    fig.subplots_adjust(left=0.1, right=0.98, top=0.8, bottom=0.14)
    return fig


def plot_load_test(df: pd.DataFrame) -> Figure:
    """Two panels: server throughput and p95 latency against concurrency (never a dual axis)."""
    fig, axes = new_figure(10, 4.4, ncols=2)
    x = list(range(len(df)))
    panels = [
        (axes[0], "output_tok_s", "Throughput (output tokens / s)", lambda v: f"{v:,.0f}"),
        (axes[1], "p95_latency_s", "p95 request latency (s)", lambda v: f"{v:.1f}s"),
    ]
    for ax, column, title, fmt in panels:
        ax.plot(
            x, df[column], color=ACCENT, linewidth=2, solid_capstyle="round", solid_joinstyle="round",
            marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3,
        )
        # Label only the two ends; the axis carries the rest.
        for i in (0, len(df) - 1):
            ax.annotate(
                fmt(df[column].iloc[i]), (x[i], df[column].iloc[i]), xytext=(0, 9), textcoords="offset points",
                ha="center", va="bottom", fontsize=9, fontweight="semibold", color=INK,
            )
        ax.set_xticks(x, [str(c) for c in df["concurrency"]])
        ax.set_xlabel("concurrent requests", color=INK_SECONDARY, fontsize=9)
        ax.set_xlim(-0.3, len(df) - 0.7)
        ax.set_ylim(0, df[column].max() * 1.18)
        ax.set_title(title, loc="left", fontsize=10, color=INK, pad=8)
    add_titles(fig, "vLLM server under load: throughput and latency vs concurrency", SUBTITLE + " · streaming /v1/completions")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.76, bottom=0.14, wspace=0.18)
    return fig


def main() -> None:
    charts = [
        ("offline_throughput", plot_offline),
        ("load_test", plot_load_test),
    ]
    for name, draw in charts:
        csv_path = RESULTS_DIR / f"{name}.csv"
        if not csv_path.exists():
            print(f"skipping {name}: {csv_path} not found")
            continue
        fig = draw(pd.read_csv(csv_path))
        png_path = RESULTS_DIR / f"{name}.png"
        fig.savefig(png_path, facecolor=SURFACE)
        plt.close(fig)
        print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
