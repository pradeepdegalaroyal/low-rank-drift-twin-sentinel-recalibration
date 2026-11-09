"""Run recalibration policies against a shared drift trace and score them."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .drift import DriftParams, DriftProcess, measurement_sigma_khz
from .fidelity import pi_pulse_infidelity
from .policies import Controller, PeriodicController, PolicySpec, TwinController
from .twin import fit_ar1, fit_twin, oracle_twin, uniform_global_twin

TRAIN_SAMPLES = 144
TRAIN_SPACING = 10  # minutes between commissioning snapshots (24 h in total)
INFID_THRESHOLD = 1e-3


@dataclass
class TrialResult:
    """Outcome of one policy on one seed."""

    label: str
    kind: str
    period: int
    shots_per_qubit_hour: float
    mean_infidelity: float
    frac_above_threshold: float
    worst_qubit_infidelity: float
    twin_rank: int


def run_trial(
    params: DriftParams,
    specs: list[PolicySpec],
    seed: int,
    horizon_min: int = 720,
    train_shots: int = 100,
) -> list[TrialResult]:
    """Simulate all ``specs`` on the same drift trace (common random numbers) and score them."""
    rng = np.random.default_rng(seed)
    proc = DriftProcess(params, rng)
    train = proc.sample_training(TRAIN_SAMPLES, TRAIN_SPACING, train_shots)
    sigma_tr = measurement_sigma_khz(train_shots)
    n = params.n_qubits
    learned = fit_twin(train, sigma_tr, TRAIN_SPACING)
    ar1 = fit_ar1(train, sigma_tr, TRAIN_SPACING)
    models = {"learned": learned, "oracle": oracle_twin(proc), "uniform": uniform_global_twin(learned, n)}

    ctrls: list[Controller] = []
    for sp in specs:
        if sp.kind == "hold":
            ctrls.append(PeriodicController(n, sp, None))
        elif sp.kind == "ar1":
            ctrls.append(PeriodicController(n, sp, ar1))
        elif sp.kind == "twin":
            ctrls.append(TwinController(n, sp, models[sp.loadings], np.random.default_rng(seed + 7)))
        else:
            raise ValueError(sp.kind)

    meas_rng = np.random.default_rng(seed + 13)
    acc_inf = np.zeros((len(specs), n))
    acc_above = np.zeros(len(specs))
    shots = np.zeros(len(specs))
    for t in range(horizon_min):
        delta = proc.step()
        for j, ctrl in enumerate(ctrls):
            c, s = ctrl.act(t, delta, meas_rng)
            inf = pi_pulse_infidelity(delta - c)
            acc_inf[j] += inf
            acc_above[j] += np.mean(inf > INFID_THRESHOLD)
            shots[j] += s

    out = []
    for j, sp in enumerate(specs):
        per_q = acc_inf[j] / horizon_min
        c_j = ctrls[j]
        rank = c_j.model.rank if isinstance(c_j, TwinController) else 0
        out.append(
            TrialResult(
                label=sp.label or f"{sp.kind}-P{sp.period}",
                kind=sp.kind,
                period=sp.period,
                shots_per_qubit_hour=float(shots[j] / (n * horizon_min / 60.0)),
                mean_infidelity=float(per_q.mean()),
                frac_above_threshold=float(acc_above[j] / horizon_min),
                worst_qubit_infidelity=float(per_q.max()),
                twin_rank=int(rank),
            )
        )
    return out
