"""Recalibration policies.  Every policy sees only noisy Ramsey measurements, never the true drift."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .drift import measurement_sigma_khz
from .twin import AR1Model, TwinModel, select_sentinels


@dataclass(frozen=True)
class PolicySpec:
    """Declarative description of a policy; ``label`` names it in result tables."""

    kind: str  # "hold", "ar1", "twin"
    period: int  # minutes between calibrations of one qubit (staggered across qubits)
    shots: int = 100  # shots per Ramsey measurement
    n_sentinels: int = 16
    sentinel_every: int = 2  # minutes between sentinel measurements
    selection: str = "doptimal"  # or "random"
    loadings: str = "learned"  # "learned", "oracle" or "uniform"
    label: str = ""


class Controller:
    """Base class: ``act`` returns the applied frequency correction c(t) and the shots spent."""

    def act(self, t: int, delta: NDArray[np.float64], rng: np.random.Generator) -> tuple[NDArray[np.float64], int]:
        raise NotImplementedError


class PeriodicController(Controller):
    """Staggered periodic recalibration; optional per-qubit OU decay and shrinkage (classical twin)."""

    def __init__(self, n: int, spec: PolicySpec, ar1: AR1Model | None) -> None:
        self.n, self.spec = n, spec
        self.sigma = measurement_sigma_khz(spec.shots)
        self.phase = np.arange(n) % spec.period
        self.m = np.zeros(n)
        self.t_cal = np.zeros(n, dtype=np.int64)
        self.decay = ar1.rho if ar1 is not None else 1.0
        self.shrink = (ar1.var / (ar1.var + self.sigma**2)) if ar1 is not None else 1.0

    def act(self, t: int, delta: NDArray[np.float64], rng: np.random.Generator) -> tuple[NDArray[np.float64], int]:
        if t == 0:
            due = np.ones(self.n, dtype=bool)
            shots = 0  # the initial full calibration is common to all policies and not charged
        else:
            due = (t + self.phase) % self.spec.period == 0
            shots = int(due.sum()) * self.spec.shots
        meas = delta + self.sigma * rng.standard_normal(self.n)
        self.m = np.where(due, meas, self.m)
        self.t_cal = np.where(due, t, self.t_cal)
        c = self.shrink * self.decay ** (t - self.t_cal) * self.m
        return c, shots


class TwinController(Controller):
    """Sentinel-driven low-rank twin plus staggered private-residual calibration.

    The latent state z (rank r) is tracked by a Kalman filter observing K sentinel qubits every
    ``sentinel_every`` minutes; the applied correction is V z plus a decayed, shrunk private residual
    measured when the qubit itself is recalibrated.
    """

    def __init__(self, n: int, spec: PolicySpec, model: TwinModel, rng: np.random.Generator) -> None:
        self.n, self.spec, self.model = n, spec, model
        self.sigma = measurement_sigma_khz(spec.shots)
        self.v = model.basis
        self.r = model.rank
        self.sent = select_sentinels(model.basis, spec.n_sentinels, rng, spec.selection)
        self.phase = np.arange(n) % spec.period
        self.z = np.zeros(self.r)
        self.p = np.diag(model.var_z) if self.r else np.zeros((0, 0))
        self.resid = np.zeros(n)
        self.t_cal = np.zeros(n, dtype=np.int64)
        self.r_noise = model.var_priv + self.sigma**2

    def _predict(self) -> None:
        if self.r:
            rho = self.model.rho_z
            self.z = rho * self.z
            self.p = self.p * np.outer(rho, rho) + np.diag((1 - rho**2) * self.model.var_z)

    def _update(self, meas: NDArray[np.float64]) -> None:
        if not self.r:
            return
        h = self.v[self.sent]
        info = np.linalg.inv(self.p) + h.T @ h / self.r_noise
        p_new = np.linalg.inv(info)
        self.z = p_new @ (np.linalg.solve(self.p, self.z) + h.T @ meas[self.sent] / self.r_noise)
        self.p = p_new

    def act(self, t: int, delta: NDArray[np.float64], rng: np.random.Generator) -> tuple[NDArray[np.float64], int]:
        spec = self.spec
        meas = delta + self.sigma * rng.standard_normal(self.n)  # one potential reading per qubit
        shots = 0
        if t > 0:
            self._predict()
        if t % spec.sentinel_every == 0:
            self._update(meas)
            if t > 0:
                shots += len(self.sent) * spec.shots
        if t == 0:
            due = np.ones(self.n, dtype=bool)
        else:
            due = (t + self.phase) % spec.period == 0
            shots += int(due.sum()) * spec.shots
        lat = self.v @ self.z if self.r else np.zeros(self.n)
        if due.any():
            var_lat = np.einsum("ij,jk,ik->i", self.v, self.p, self.v) if self.r else np.zeros(self.n)
            kappa = self.model.var_priv / (self.model.var_priv + self.sigma**2 + var_lat)
            self.resid = np.where(due, kappa * (meas - lat), self.resid)
            self.t_cal = np.where(due, t, self.t_cal)
        c = lat + self.model.rho_priv ** (t - self.t_cal) * self.resid
        return c, shots
