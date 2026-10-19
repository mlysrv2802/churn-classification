from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from churn.config import TARGET

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"


def _binned_rate(x: pd.Series, y: pd.Series, bins: int) -> pd.DataFrame:
    """Mean x, churn rate and size per equal-count (quantile) bin of x."""
    groups = pd.qcut(x, q=bins, duplicates="drop")
    g = pd.DataFrame({"x": x, "y": y}).groupby(groups, observed=True)
    return pd.DataFrame({"x": g["x"].mean(), "rate": g["y"].mean(), "n": g.size()})


def plot_churn_vs_numeric(
    df: pd.DataFrame, columns: list[str], out_path: Path, bins: int = 10
) -> Path:
    """Small multiples: churn rate per quantile bin of each column.

    Dots are the data; the dashed grey line is the best straight line through
    them. Dots that bend away from the line are the non-linearity.
    `df` is the cleaned frame (Churn is 0/1).
    """
    fig, axes = plt.subplots(
        1, len(columns), figsize=(4.6 * len(columns), 4.2), sharey=True, facecolor=SURFACE
    )
    axes = np.atleast_1d(axes)
    for ax, col in zip(axes, columns):
        b = _binned_rate(df[col], df[TARGET], bins)
        slope, intercept = np.polyfit(b["x"], b["rate"], deg=1, w=np.sqrt(b["n"]))
        xs = np.linspace(b["x"].min(), b["x"].max(), 50)

        ax.set_facecolor(SURFACE)
        (fit_line,) = ax.plot(xs, slope * xs + intercept, color=INK_SECONDARY, lw=1.5, ls="--",
                              label="best straight line")
        (data_line,) = ax.plot(b["x"], b["rate"], color=SERIES, lw=2, marker="o", ms=7,
                               markeredgecolor=SURFACE, markeredgewidth=2,
                               label="actual churn rate")
        ax.set_title(col, loc="left", color=INK, fontsize=12, fontweight="bold")
        ax.set_xlabel(f"{col} (mean of each of {len(b)} equal-size bins)",
                      color=INK_SECONDARY, fontsize=9)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK_SECONDARY, length=0)

    axes[0].set_ylabel("churn rate", color=INK_SECONDARY)
    axes[0].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    fig.suptitle("Churn rate vs numeric columns", x=0.01, ha="left", color=INK, fontsize=14)
    fig.legend(handles=[data_line, fit_line], loc="upper right", ncol=2, frameon=False,
               labelcolor=INK_SECONDARY, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out_path
