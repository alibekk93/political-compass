"""Figures for the common-space results (static matplotlib, light mode).

Colors follow the dataviz reference palette: categorical slots in fixed
order (identity = bloc lineage; scatter forms cap at 4 colored series, the
rest fold to a muted "other"), one blue sequential ramp for magnitude, no
dual axes (stacked panels instead), recessive grid/axis chrome.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
CATEGORICAL = ["#2a78d6", "#008300", "#e87ba4", "#eda100"]  # scatter cap: 4 slots
LINE_SLOTS = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834"]
SEQ_RAMP = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
     "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"],
)


def first_year(congress: int) -> int:
    return 1787 + 2 * congress


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PAGE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": PAGE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
            "text.color": INK,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlelocation": "left",
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
        }
    )


def _bloc_colors(positions: pd.DataFrame) -> dict[int, str]:
    """Top-4 blocs by total membership get the categorical slots (fixed
    assignment, never recycled); every other bloc folds to muted."""
    sizes = positions.groupby("bloc").size().sort_values(ascending=False)
    colors = {int(b): MUTED for b in sizes.index}
    for slot, b in enumerate(sizes.index[:4]):
        colors[int(b)] = CATEGORICAL[slot]
    return colors


def fig_small_multiples(positions: pd.DataFrame, path: Path, chamber: str, n_panels: int = 12):
    congresses = sorted(positions["congress"].unique())
    picks = sorted({congresses[i] for i in np.linspace(0, len(congresses) - 1, n_panels).round().astype(int)})
    colors = _bloc_colors(positions)
    ncol = 4
    nrow = int(np.ceil(len(picks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 2.9 * nrow), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for ax, t in zip(axes.ravel(), picks):
        ax.set_visible(True)
        sub = positions[positions["congress"] == t]
        for b, g in sub.groupby("bloc"):
            ax.scatter(g["dim1"], g["dim2"], s=14, c=colors[int(b)], alpha=0.85,
                       linewidths=0.4, edgecolors=SURFACE, label=f"bloc {b}")
        ax.set_title(f"Congress {t} · {first_year(t)}–{first_year(t) + 2}")
        ax.set_xlim(-2.6, 2.6)
        ax.set_ylim(-2.6, 2.6)
        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([-2, 0, 2])
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=f"bloc {b}")
               for b, c in sorted(colors.items()) if c != MUTED]
    handles.append(plt.Line2D([], [], marker="o", ls="", color=MUTED, label="other blocs"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"{chamber}: members in the common voting space", fontweight="bold",
                 x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_polarization(diag: pd.DataFrame, path: Path, chamber: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True)
    ax1.plot(diag.index, diag["separation_k2"], color=LINE_SLOTS[0], lw=2)
    ax1.set_title(f"{chamber}: two-cluster separation (centroid distance / within-cluster spread)")
    ax2.plot(diag.index, diag["silhouette_k2"], color=LINE_SLOTS[0], lw=2)
    ax2.set_title("Silhouette score at k = 2")
    ax2.set_xlabel("congress (first year of term below)")
    years = np.arange(10, diag.index.max() + 1, 10)
    ax2.set_xticks(years)
    ax2.set_xticklabels([f"{t}\n{first_year(t)}" for t in years])
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_anchor_residuals(diag: pd.DataFrame, path: Path, chamber: str):
    d = diag.dropna(subset=["anchor_resid"])
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(d.index, d["anchor_resid"], color=LINE_SLOTS[0], lw=2)
    ax.set_title(f"{chamber}: anchor residual after alignment (spikes = realignment eras)")
    ax.set_xlabel("congress")
    for t in d["anchor_resid"].nlargest(3).index:
        ax.annotate(f"{t} ({first_year(t)})", (t, d.loc[t, 'anchor_resid']),
                    textcoords="offset points", xytext=(4, 4), fontsize=8, color=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_eigen_spectrum(diag: pd.DataFrame, path: Path, chamber: str, n_eigs: int = 10):
    cols = [f"eig{i}" for i in range(1, n_eigs + 1)]
    E = diag[cols].to_numpy(dtype=float)
    E = np.clip(E, 0.0, None)
    shares = E / E.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    im = ax.imshow(shares.T, aspect="auto", cmap=SEQ_RAMP, origin="upper",
                   extent=(diag.index.min() - 0.5, diag.index.max() + 0.5, n_eigs + 0.5, 0.5),
                   vmin=0, vmax=1)
    ax.set_title(f"{chamber}: MDS eigenvalue shares — voting is low-dimensional when row 1 is dark")
    ax.set_xlabel("congress")
    ax.set_ylabel("dimension")
    ax.set_yticks(range(1, n_eigs + 1, 3))
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("share of top-10 eigenvalue mass", color=INK_2, fontsize=8)
    cbar.ax.tick_params(labelsize=8, color=MUTED, labelcolor=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_trajectories(positions: pd.DataFrame, path: Path, chamber: str, n_members: int = 6):
    spans = positions.groupby("icpsr")["congress"].agg(["count", "min", "max"])
    picks = spans.sort_values(["count", "min"], ascending=[False, True]).head(n_members).index
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.scatter(positions["dim1"], positions["dim2"], s=4, c=MUTED, alpha=0.08, linewidths=0)
    for slot, m in enumerate(picks):
        g = positions[positions["icpsr"] == m].sort_values("congress")
        ax.plot(g["dim1"], g["dim2"], color=LINE_SLOTS[slot], lw=2, alpha=0.9,
                marker="o", ms=3.5, mec=SURFACE, mew=0.4)
        end = g.iloc[-1]
        ax.annotate(f"icpsr {m}\n{first_year(g['congress'].min())}–{first_year(int(end['congress'])) + 2}",
                    (end["dim1"], end["dim2"]), textcoords="offset points", xytext=(6, 4),
                    fontsize=7.5, color=INK_2)
    ax.set_title(f"{chamber}: longest-serving members' paths through the common space")
    ax.set_xlabel("dimension 1")
    ax.set_ylabel("dimension 2")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_all(positions: pd.DataFrame, diag: pd.DataFrame, figdir: Path, chamber: str) -> list[Path]:
    apply_style()
    figdir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (fig_small_multiples, f"{chamber}_compass_small_multiples.png"),
        (fig_polarization, f"{chamber}_polarization_timeline.png"),
        (fig_anchor_residuals, f"{chamber}_anchor_residuals.png"),
        (fig_eigen_spectrum, f"{chamber}_eigen_spectrum.png"),
        (fig_trajectories, f"{chamber}_trajectories.png"),
    ]
    paths = []
    for fn, name in jobs:
        p = figdir / name
        data = positions if fn in (fig_small_multiples, fig_trajectories) else diag
        fn(data, p, chamber)
        paths.append(p)
    return paths
