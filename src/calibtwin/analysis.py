"""Aggregation helpers: cost-at-target interpolation, Pareto front, seed statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_INFIDELITY = 3e-4  # mean pi-pulse infidelity every policy is asked to reach


def cost_at_target(cost: np.ndarray, infid: np.ndarray, target: float = TARGET_INFIDELITY) -> float:
    """Smallest shots/qubit/hour reaching ``infid <= target`` (log-log interpolation along the sweep).

    Returns NaN if even the most expensive setting misses the target, and the cheapest cost if the
    cheapest setting already meets it.
    """
    order = np.argsort(cost)
    c, f = np.asarray(cost, float)[order], np.asarray(infid, float)[order]
    ok = f <= target
    if not ok.any():
        return float("nan")
    if ok[0]:
        return float(c[0])
    j = int(np.argmax(ok))  # first (cheapest) feasible point; j >= 1
    lc0, lc1 = np.log(c[j - 1]), np.log(c[j])
    lf0, lf1 = np.log(f[j - 1]), np.log(f[j])
    if lf0 == lf1:
        return float(c[j])
    w = (np.log(target) - lf0) / (lf1 - lf0)
    return float(np.exp(lc0 + w * (lc1 - lc0)))


def pareto_mask(cost: np.ndarray, infid: np.ndarray) -> np.ndarray:
    """True for points not dominated in (cost, infidelity), both minimised."""
    cost, infid = np.asarray(cost), np.asarray(infid)
    keep = np.ones(len(cost), dtype=bool)
    for i in range(len(cost)):
        dom = (cost <= cost[i]) & (infid <= infid[i]) & ((cost < cost[i]) | (infid < infid[i]))
        keep[i] = not dom.any()
    return keep


def cost_table(df: pd.DataFrame, group: list[str], target: float = TARGET_INFIDELITY) -> pd.DataFrame:
    """Per (group, label, seed) cost at target from a long table of sweep points."""
    rows = []
    for key, g in df.groupby(group + ["label", "seed"]):
        rec = dict(zip(group + ["label", "seed"], key if isinstance(key, tuple) else (key,)))
        rec["cost_at_target"] = cost_at_target(g["shots_per_qubit_hour"].to_numpy(), g["mean_infidelity"].to_numpy(), target)
        rows.append(rec)
    return pd.DataFrame(rows)


def mean_std(df: pd.DataFrame, by: list[str], col: str) -> pd.DataFrame:
    """Mean, sample std (ddof=1) and count of ``col`` over seeds."""
    g = df.groupby(by)[col]
    return pd.DataFrame({"mean": g.mean(), "std": g.std(ddof=1), "n_seeds": g.count()}).reset_index()
