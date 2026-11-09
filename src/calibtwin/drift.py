"""Synthetic drift process of a qubit array (the 'physical device' the twin must track).

Each qubit i has a frequency error delta_i(t) in kHz made of
  (1) a low-rank correlated part  sum_k L[i,k] g_k(t)  with Ornstein-Uhlenbeck modes g_k
      (global temperature / flux / field-gradient drift, spatially smooth plus per-qubit loading noise),
  (2) a private Ornstein-Uhlenbeck part (local charge or two-level-system background),
  (3) a random-telegraph part (two-level-system switching) that no model in this study predicts.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

#: Ramsey free-evolution time in microseconds; single-shot frequency std is 1/(2 pi tau).
RAMSEY_TAU_US = 1.0


def measurement_sigma_khz(shots: float, tau_us: float = RAMSEY_TAU_US) -> float:
    """Std (kHz) of a Ramsey frequency estimate from ``shots`` shots at the quadrature point."""
    return 1e3 / (2.0 * np.pi * tau_us) / np.sqrt(shots)


@dataclass(frozen=True)
class DriftParams:
    """Parameters of the drift process; all frequencies in kHz, times in minutes."""

    n_qubits: int = 256
    sigma_total_khz: float = 120.0  # rms of the correlated + private parts
    latent_share: float = 0.8  # fraction of that variance carried by the correlated part
    true_rank: int = 3
    tau_latent_min: float = 60.0
    tau_private_min: float = 30.0
    loading_noise: float = 0.15  # per-qubit loading perturbation (relative to smooth profile)
    tls_rate_per_hour: float = 0.1  # per-qubit switching rate of the telegraph part
    tls_amp_khz: float = 40.0  # peak-to-peak size of a switch

    def with_(self, **kw: float) -> "DriftParams":
        return replace(self, **kw)


def grid_side(n: int) -> int:
    side = int(round(np.sqrt(n)))
    if side * side != n:
        raise ValueError("n_qubits must be a perfect square (square qubit grid)")
    return side


def _mode_indices(rank: int) -> list[tuple[int, int]]:
    modes: list[tuple[int, int]] = []
    order = 0
    while len(modes) < rank:
        for a in range(order + 1):
            modes.append((a, order - a))
        order += 1
    return modes[:rank]


def make_loadings(p: DriftParams, rng: np.random.Generator) -> NDArray[np.float64]:
    """Loading matrix L (n x rank) with unit mean-square row norm."""
    side = grid_side(p.n_qubits)
    xs = (np.arange(side) + 0.5) / side
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    gx, gy = gx.ravel(), gy.ravel()
    cols = [np.cos(np.pi * a * gx) * np.cos(np.pi * b * gy) for a, b in _mode_indices(p.true_rank)]
    smooth = np.stack(cols, axis=1)
    loadings = smooth + p.loading_noise * rng.standard_normal(smooth.shape)
    loadings /= np.sqrt(np.mean(np.sum(loadings**2, axis=1)))
    return loadings


class DriftProcess:
    """Stateful generator of delta(t), one minute per step."""

    def __init__(self, params: DriftParams, rng: np.random.Generator) -> None:
        self.p = params
        self.rng = rng
        n = params.n_qubits
        self.loadings = make_loadings(params, rng)
        self.sig_lat = params.sigma_total_khz * np.sqrt(params.latent_share)
        self.sig_priv = params.sigma_total_khz * np.sqrt(1.0 - params.latent_share)
        self.rho_lat = float(np.exp(-1.0 / params.tau_latent_min))
        self.rho_priv = float(np.exp(-1.0 / params.tau_private_min))
        self.p_flip = 1.0 - float(np.exp(-params.tls_rate_per_hour / 60.0))
        self.g = rng.standard_normal(params.true_rank)
        self.priv = self.sig_priv * rng.standard_normal(n)
        self.tls = rng.choice([-0.5, 0.5], size=n)

    def step(self) -> NDArray[np.float64]:
        """Advance one minute and return the true frequency error of every qubit (kHz)."""
        rng, p = self.rng, self.p
        self.g = self.rho_lat * self.g + np.sqrt(1 - self.rho_lat**2) * rng.standard_normal(self.g.shape)
        self.priv = self.rho_priv * self.priv + np.sqrt(1 - self.rho_priv**2) * self.sig_priv * rng.standard_normal(
            p.n_qubits
        )
        flips = rng.random(p.n_qubits) < self.p_flip
        self.tls = np.where(flips, -self.tls, self.tls)
        return self.sig_lat * (self.loadings @ self.g) + self.priv + p.tls_amp_khz * self.tls

    def sample_training(self, n_samples: int, spacing: int, shots: float) -> NDArray[np.float64]:
        """Commissioning data: noisy full-chip Ramsey measurements every ``spacing`` minutes."""
        sigma = measurement_sigma_khz(shots)
        rows = []
        for _ in range(n_samples):
            for _ in range(spacing):
                d = self.step()
            rows.append(d + sigma * self.rng.standard_normal(d.shape))
        return np.asarray(rows)
