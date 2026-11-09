"""Figures (earth-tone palette, 200 dpi)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .analysis import TARGET_INFIDELITY, pareto_mask  # noqa: E402

BROWN, TEAL, OCHRE, SAND, SIENNA, OLIVE = "#6b4423", "#2a7f7f", "#cc9a2b", "#d9c3a0", "#a0522d", "#5b6b3a"
COLORS = {
    "Periodic Ramsey": BROWN,
    "Per-qubit Kalman": OCHRE,
    "Low-rank twin": TEAL,
    "Oracle twin": OLIVE,
}
HATCH = {"Periodic Ramsey": "//", "Per-qubit Kalman": "..", "Low-rank twin": "xx", "Oracle twin": "--"}
MARK = {"Periodic Ramsey": "s", "Per-qubit Kalman": "^", "Low-rank twin": "o", "Oracle twin": "D"}
DPI = 200


def _style(ax: plt.Axes) -> None:
    ax.grid(alpha=0.25, color=BROWN)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_pareto(df: pd.DataFrame, path: Path, n: int) -> None:
    """Cost-quality scatter; per-seed points faint, seed means bold, Pareto front outlined."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    means = df.groupby(["label", "period"])[["shots_per_qubit_hour", "mean_infidelity"]].mean().reset_index()
    for lab, g in df.groupby("label"):
        ax.scatter(g.shots_per_qubit_hour, g.mean_infidelity, s=10, color=COLORS[lab], alpha=0.25, marker=MARK[lab])
    for lab, g in means.groupby("label"):
        g = g.sort_values("period")
        ax.plot(g.shots_per_qubit_hour, g.mean_infidelity, color=COLORS[lab], lw=1, alpha=0.7)
        ax.scatter(g.shots_per_qubit_hour, g.mean_infidelity, s=42, color=COLORS[lab], marker=MARK[lab],
                   edgecolor="black", linewidth=0.5, label=lab, zorder=3)
    m = pareto_mask(means.shots_per_qubit_hour.to_numpy(), means.mean_infidelity.to_numpy())
    front = means[m].sort_values("shots_per_qubit_hour")
    ax.step(front.shots_per_qubit_hour, front.mean_infidelity, where="post", color="black", lw=1.6, ls="--",
            label="Pareto front (seed means)", zorder=2)
    ax.axhline(TARGET_INFIDELITY, color=SIENNA, lw=1, ls=":")
    ax.text(ax.get_xlim()[0] * 1.05, TARGET_INFIDELITY * 1.04, "target infidelity", color=SIENNA, fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Calibration cost (Ramsey shots per qubit per hour)")
    ax.set_ylabel("Mean pi-pulse infidelity (average gate)")
    ax.set_title(f"Calibration cost versus gate quality, N = {n} qubits (points: seeds; markers: seed means)", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def fig_scaling(sc: pd.DataFrame, path: Path) -> None:
    """Grouped hatched bars: calibration cost needed to reach the target infidelity versus array size."""
    labels = ["Periodic Ramsey", "Per-qubit Kalman", "Low-rank twin"]
    ns = sorted(sc.n.unique())
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    w = 0.26
    for j, lab in enumerate(labels):
        d = sc[sc.label == lab].set_index("n").reindex(ns)
        x = np.arange(len(ns)) + (j - 1) * w
        ax.bar(x, d["mean"], w, yerr=d["std"].fillna(0), color=COLORS[lab], hatch=HATCH[lab], edgecolor="black",
               linewidth=0.6, capsize=3, label=lab, alpha=0.9)
        for xi, v in zip(x, d["mean"]):
            if np.isnan(v):
                ax.text(xi, 12, "n/r", ha="center", fontsize=7)
    ax.set_xticks(np.arange(len(ns)))
    ax.set_xticklabels([str(int(n)) for n in ns])
    ax.set_yscale("log")
    ax.set_xlabel("Array size N (qubits)")
    ax.set_ylabel(f"Shots per qubit per hour to reach mean infidelity {TARGET_INFIDELITY:.0e}")
    ax.set_title("Cost to hold the infidelity target versus array size (mean ± std over seeds)", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def fig_ablation(ct: pd.DataFrame, path: Path, n: int) -> None:
    """Box plots of the per-seed cost reduction factor for each twin variant, seeds overlaid."""
    d = ct[ct.label != "Periodic Ramsey"].copy()
    order = [l for l in ["K = 4", "K = 8", "K = 16", "K = 32", "K = 64", "Oracle loadings", "Uniform global mode",
                         "Random sentinels", "Sentinel every 10 min"] if l in set(d.label)]
    data = [d[d.label == l]["reduction_vs_periodic"].dropna().to_numpy() for l in order]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    bp = ax.boxplot(data, positions=range(len(order)), widths=0.55, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set(facecolor=SAND, edgecolor=BROWN)
    for med in bp["medians"]:
        med.set(color=SIENNA, lw=2)
    rng = np.random.default_rng(0)
    for i, v in enumerate(data):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=22, color=TEAL, edgecolor="black", linewidth=0.4, zorder=3)
    ax.axhline(1.0, color="black", lw=1, ls="--")
    for i, l in enumerate(order):
        k = int(d[d.label == l]["cost_at_target"].notna().sum())
        tot = int((d.label == l).sum())
        if k < tot:
            ax.text(i, ax.get_ylim()[1] * 0.97, f"target reached\nin {k}/{tot} seeds", ha="center", va="top", fontsize=7, color=SIENNA)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Cost reduction versus periodic Ramsey (factor)")
    ax.set_title(f"Ablation of the low-rank twin, N = {n} (dots: individual seeds; dashed: no saving)", fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def fig_robustness(rb: pd.DataFrame, path: Path, n: int) -> None:
    """Four-panel grouped bars: infidelity at a fixed shot budget under four stressors."""
    titles = {
        "latent_share": "Correlated share of drift variance",
        "tls_rate_per_hour": "Telegraph switching rate (per qubit per hour)",
        "true_rank": "True rank of the correlated drift",
        "shots": "Shots per Ramsey measurement",
    }
    labels = ["Periodic Ramsey", "Per-qubit Kalman", "Low-rank twin"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2), sharey=True)
    for ax, (name, title) in zip(axes, titles.items()):
        d = rb[rb.sweep == name]
        vals = sorted(d.value.unique())
        w = 0.26
        for j, lab in enumerate(labels):
            g = d[d.label == lab].set_index("value").reindex(vals)
            ax.bar(np.arange(len(vals)) + (j - 1) * w, g["mean"], w, yerr=g["std"].fillna(0), color=COLORS[lab],
                   hatch=HATCH[lab], edgecolor="black", linewidth=0.5, capsize=2, label=lab, alpha=0.9)
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels([f"{v:g}" for v in vals], fontsize=8)
        ax.set_xlabel(title, fontsize=8)
        ax.set_yscale("log")
        _style(ax)
    axes[0].set_ylabel("Mean infidelity at 6 measurements/qubit/h", fontsize=9)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"Robustness at a fixed calibration budget, N = {n} (mean ± std over seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
