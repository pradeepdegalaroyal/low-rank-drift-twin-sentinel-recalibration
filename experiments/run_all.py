"""Reproduce every result and figure.  ``python experiments/run_all.py`` (full) or ``--quick`` (smoke run)."""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calibtwin import plotting  # noqa: E402
from calibtwin.analysis import TARGET_INFIDELITY, cost_table, mean_std  # noqa: E402
from calibtwin.drift import DriftParams, measurement_sigma_khz  # noqa: E402
from calibtwin.policies import PolicySpec  # noqa: E402
from calibtwin.simulate import TRAIN_SAMPLES, run_trial  # noqa: E402

BUDGET = 600.0  # shots / qubit / hour used for the fixed-budget comparison
PERIODS_FULL = (5, 10, 20, 40, 80, 160, 320)
PERIODS_QUICK = (5, 20, 80)


def _task(args: tuple) -> list[dict]:
    tag, params, specs, seed, horizon = args
    rows = []
    for r in run_trial(params, specs, seed, horizon_min=horizon):
        d = asdict(r)
        d.update(tag)
        d["seed"] = seed
        rows.append(d)
    return rows


def _run(tasks: list[tuple], workers: int, cache: Path | None = None) -> pd.DataFrame:
    if cache is not None and cache.exists():
        return pd.read_csv(cache)
    if workers <= 1:
        out = [_task(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            out = list(ex.map(_task, tasks))
    return pd.DataFrame([r for rows in out for r in rows])


def base_specs(periods: tuple[int, ...], shots: int = 100) -> list[PolicySpec]:
    specs: list[PolicySpec] = []
    for p in periods:
        specs += [
            PolicySpec("hold", p, shots, label="Periodic Ramsey"),
            PolicySpec("ar1", p, shots, label="Per-qubit Kalman"),
            PolicySpec("twin", p, shots, label="Low-rank twin"),
        ]
    return specs


def infidelity_at_budget(cost: np.ndarray, infid: np.ndarray, budget: float = BUDGET) -> float:
    """Log-log interpolated mean infidelity at a fixed shots/qubit/hour budget (NaN if out of range)."""
    o = np.argsort(cost)
    c, f = np.asarray(cost, float)[o], np.asarray(infid, float)[o]
    if budget < c[0] or budget > c[-1]:
        return float("nan")
    return float(np.exp(np.interp(np.log(budget), np.log(c), np.log(f))))


def budget_table(df: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    """Infidelity at a budget of 6 Ramsey measurements per qubit per hour (600 shots at 100 shots each)."""
    rows = []
    for key, g in df.groupby(group + ["label", "seed"]):
        rec = dict(zip(group + ["label", "seed"], key))
        shots = rec["value"] if rec.get("sweep") == "shots" else 100
        rec["budget"] = 6.0 * shots
        rec["infid_at_budget"] = infidelity_at_budget(
            g["shots_per_qubit_hour"].to_numpy(), g["mean_infidelity"].to_numpy(), rec["budget"]
        )
        rows.append(rec)
    return pd.DataFrame(rows)


def validation(out: Path, quick: bool) -> dict:
    from calibtwin.validate import gate_fidelity_check, ramsey_estimator_check

    g = pd.DataFrame(gate_fidelity_check(np.array([0.0, 25, 50, 100, 200, 300])))
    g["abs_diff"] = (g["closed_form"] - g["qiskit"]).abs()
    g.to_csv(out / "validation_gate.csv", index=False)
    r = pd.DataFrame(ramsey_estimator_check([25, 100, 400], repeats=60 if quick else 300))
    r["std_ratio"] = r["empirical_std_khz"] / r["model_std_khz"]
    r.to_csv(out / "validation_ramsey.csv", index=False)
    return {
        "gate_max_abs_diff": float(g["abs_diff"].max()),
        "ramsey_std_ratio_min": float(r["std_ratio"].min()),
        "ramsey_std_ratio_max": float(r["std_ratio"].max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke run (< 1 minute)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--analyse-only", action="store_true", help="re-use the raw sweep CSVs in results/")
    a = ap.parse_args()
    quick = a.quick
    res = ROOT / ("results_quick" if quick else "results")
    figs = ROOT / ("figures_quick" if quick else "figures")
    res.mkdir(exist_ok=True)
    figs.mkdir(exist_ok=True)
    t0 = time.time()

    seeds = [0, 1] if quick else [0, 1, 2, 3, 4]
    periods = PERIODS_QUICK if quick else PERIODS_FULL
    horizon = 240 if quick else 720
    summary: dict = {"quick": quick, "seeds": seeds, "target_infidelity": TARGET_INFIDELITY, "budget_shots_per_qubit_hour": BUDGET}

    def cache(name: str) -> Path | None:
        return res / name if a.analyse_only else None

    print("[1/5] Qiskit / Aer validation", flush=True)
    summary["validation"] = validation(res, quick)

    print("[2/5] Pareto sweep (N = 256)", flush=True)
    n_main = 64 if quick else 256
    specs = base_specs(periods) + [PolicySpec("twin", p, 100, loadings="oracle", label="Oracle twin") for p in periods]
    pareto = _run([({"n_qubits": n_main}, DriftParams(n_qubits=n_main), specs, s, horizon) for s in seeds], a.workers, cache("pareto.csv"))
    pareto.to_csv(res / "pareto.csv", index=False)
    summary["rank_selected_main"] = int(pareto.loc[pareto.label == "Low-rank twin", "twin_rank"].mode().iloc[0])

    print("[3/5] Scaling in qubit number", flush=True)
    ns = [16, 64] if quick else [16, 64, 256, 1024, 4096]
    tasks = [({"n_qubits": n}, DriftParams(n_qubits=n), base_specs(periods), s, horizon) for n in ns for s in seeds]
    if not quick:
        big_periods = (5, 10, 20, 40)
        tasks += [({"n_qubits": 16384}, DriftParams(n_qubits=16384), base_specs(big_periods), s, horizon) for s in seeds[:3]]
    scaling = _run(tasks, a.workers, cache("scaling.csv"))
    scaling.to_csv(res / "scaling.csv", index=False)

    print("[4/5] Ablation (N = 1024)", flush=True)
    n_ab = 64 if quick else 1024
    abl_specs = [PolicySpec("hold", p, 100, label="Periodic Ramsey") for p in periods]
    for p in periods:
        for k in ([4, 16] if quick else [4, 8, 16, 32, 64]):
            abl_specs.append(PolicySpec("twin", p, 100, n_sentinels=k, label=f"K = {k}"))
        abl_specs += [
            PolicySpec("twin", p, 100, loadings="oracle", label="Oracle loadings"),
            PolicySpec("twin", p, 100, loadings="uniform", label="Uniform global mode"),
            PolicySpec("twin", p, 100, selection="random", label="Random sentinels"),
            PolicySpec("twin", p, 100, sentinel_every=10, label="Sentinel every 10 min"),
        ]
    abl = _run([({"n_qubits": n_ab}, DriftParams(n_qubits=n_ab), abl_specs, s, horizon) for s in seeds], a.workers, cache("ablation.csv"))
    abl.to_csv(res / "ablation.csv", index=False)

    print("[5/5] Robustness sweeps (N = 256)", flush=True)
    n_rb = 64 if quick else 256
    sweeps: dict[str, list] = {
        "latent_share": [0.3, 0.8] if quick else [0.3, 0.5, 0.8, 0.95],
        "tls_rate_per_hour": [0.0, 1.6] if quick else [0.0, 0.1, 0.4, 1.6],
        "true_rank": [3, 12] if quick else [2, 3, 6, 12],
        "shots": [25, 400] if quick else [25, 100, 400],
    }
    tasks = []
    for name, vals in sweeps.items():
        for v in vals:
            if name == "shots":
                prm, sp = DriftParams(n_qubits=n_rb), base_specs(periods, shots=int(v))
            else:
                prm, sp = DriftParams(n_qubits=n_rb).with_(**{name: v}), base_specs(periods)
            tasks += [({"sweep": name, "value": float(v)}, prm, sp, s, horizon) for s in seeds]
    rob = _run(tasks, a.workers, cache("robustness.csv"))
    rob.to_csv(res / "robustness.csv", index=False)

    # ---------------------------------------------------------------- summaries
    ct_sc = cost_table(scaling.assign(n=scaling["n_qubits"]), ["n"])
    sc = mean_std(ct_sc, ["n", "label"], "cost_at_target")
    sc.to_csv(res / "scaling_cost_at_target.csv", index=False)
    pv = ct_sc.pivot(index=["n", "seed"], columns="label", values="cost_at_target").reset_index()
    pv["reduction_vs_periodic"] = pv["Periodic Ramsey"] / pv["Low-rank twin"]
    pv["reduction_vs_kalman"] = pv["Per-qubit Kalman"] / pv["Low-rank twin"]
    red = (
        pv.groupby("n")[["reduction_vs_periodic", "reduction_vs_kalman"]].agg(["mean", "std", "count"]).reset_index()
    )
    red.columns = ["_".join(c).strip("_") for c in red.columns]
    red.to_csv(res / "scaling_reduction.csv", index=False)

    ct_main = cost_table(pareto.assign(n=n_main), ["n"])
    main_stats = mean_std(ct_main, ["label"], "cost_at_target")
    main_stats.to_csv(res / "pareto_cost_at_target.csv", index=False)

    ct_ab = cost_table(abl.assign(n=n_ab), ["n"])
    base = ct_ab[ct_ab.label == "Periodic Ramsey"].set_index("seed")["cost_at_target"]
    ct_ab["reduction_vs_periodic"] = ct_ab.apply(lambda r: base.get(r["seed"], np.nan) / r["cost_at_target"], axis=1)
    ct_ab.to_csv(res / "ablation_cost_at_target.csv", index=False)
    ab_stats = mean_std(ct_ab[ct_ab.label != "Periodic Ramsey"], ["label"], "reduction_vs_periodic")
    reached = ct_ab.groupby("label")["cost_at_target"].apply(lambda x: int(x.notna().sum())).rename("seeds_reaching_target")
    ab_stats = ab_stats.drop(columns="n_seeds").merge(reached, on="label", how="right")
    ab_stats["seeds_total"] = len(seeds)
    ab_stats.to_csv(res / "ablation_reduction.csv", index=False)
    # best infidelity each variant attains anywhere on the sweep (shows how far unreachable variants fall short)
    best = abl.loc[abl.groupby(["label", "seed"])["mean_infidelity"].idxmin()]
    best.groupby("label")[["mean_infidelity", "shots_per_qubit_hour"]].agg(["mean", "std"]).to_csv(res / "ablation_best_point.csv")

    bt = budget_table(rob, ["sweep", "value"])
    bt.to_csv(res / "robustness_infid_at_budget.csv", index=False)
    rb_stats = mean_std(bt, ["sweep", "value", "label"], "infid_at_budget")
    rb_stats.to_csv(res / "robustness_summary.csv", index=False)
    ctr = cost_table(rob, ["sweep", "value"])
    ctr.to_csv(res / "robustness_cost_at_target.csv", index=False)

    # ---------------------------------------------------------------- cost model extrapolation
    n_top = int(max(ns) if quick else 16384)
    tw = sc[(sc.n == n_top) & (sc.label == "Low-rank twin")]["mean"].iloc[0]
    hd = sc[(sc.n == n_top) & (sc.label == "Periodic Ramsey")]["mean"].iloc[0]
    k_sent, every, shots = 16, 2, 100
    sent_cost = lambda n: 60.0 * k_sent * shots / (every * n)  # shots / qubit / hour
    floor = float(tw - sent_cost(n_top))
    ext = []
    for n in (1024, 16384, 262144, 1048576):
        c_tw = floor + sent_cost(n)
        ext.append(
            {
                "n_qubits": n,
                "twin_shots_per_qubit_hour": c_tw,
                "periodic_shots_per_qubit_hour": float(hd),
                "twin_array_shots_per_second": c_tw * n / 3600.0,
                "periodic_array_shots_per_second": float(hd) * n / 3600.0,
                "modelled_not_simulated": n > n_top,
            }
        )
    pd.DataFrame(ext).to_csv(res / "extrapolation_cost_model.csv", index=False)
    summary["commissioning_shots_per_qubit"] = TRAIN_SAMPLES * 100
    summary["ramsey_sigma_khz_100_shots"] = measurement_sigma_khz(100)
    summary["runtime_seconds"] = round(time.time() - t0, 1)
    (res / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---------------------------------------------------------------- figures
    plotting.fig_pareto(pareto, figs / "fig1_pareto_cost_vs_infidelity.png", n_main)
    plotting.fig_scaling(sc, figs / "fig2_scaling_cost_at_target.png")
    plotting.fig_ablation(ct_ab, figs / "fig3_ablation_reduction.png", n_ab)
    plotting.fig_robustness(rb_stats, figs / "fig4_robustness_fixed_budget.png", n_rb)
    summary["runtime_seconds"] = round(time.time() - t0, 1)
    (res / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"done in {summary['runtime_seconds']} s -> {res}, {figs}", flush=True)


if __name__ == "__main__":
    main()
