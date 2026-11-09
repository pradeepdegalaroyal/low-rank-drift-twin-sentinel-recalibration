"""Digital-twin models fitted from commissioning data, and sentinel selection."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .drift import DriftProcess


@dataclass(frozen=True)
class AR1Model:
    """Per-qubit Ornstein-Uhlenbeck model used by the classical single-qubit baseline."""

    rho: float  # per-minute autocorrelation
    var: float  # signal variance (kHz^2)


@dataclass(frozen=True)
class TwinModel:
    """Low-rank-plus-private model: delta = V z + p, z and p independent AR(1) (per minute)."""

    basis: NDArray[np.float64]  # V, n x r, orthonormal columns
    var_z: NDArray[np.float64]  # variance of each latent coordinate (kHz^2)
    rho_z: NDArray[np.float64]  # per-minute autocorrelation of each latent coordinate
    var_priv: float  # private variance per qubit (kHz^2)
    rho_priv: float  # per-minute autocorrelation of the private part

    @property
    def rank(self) -> int:
        return int(self.basis.shape[1])


def _per_minute(rho_spaced: float, spacing: int) -> float:
    return float(np.clip(rho_spaced, 1e-3, 0.999) ** (1.0 / spacing))


def _lag_corr(a: NDArray[np.float64], var_signal: NDArray[np.float64] | float) -> NDArray[np.float64]:
    """Noise-corrected lag-1 autocorrelation along axis 0 (white measurement noise removed)."""
    cov = np.mean(a[:-1] * a[1:], axis=0)
    return cov / np.maximum(var_signal, 1e-9)


def fit_ar1(train: NDArray[np.float64], sigma_m: float, spacing: int) -> AR1Model:
    """Pooled single-qubit OU fit (the classical, correlation-blind twin)."""
    var_signal = max(float(np.mean(train**2)) - sigma_m**2, 1e-6)
    cov = float(np.mean(train[:-1] * train[1:]))
    rho_spaced = cov / var_signal
    return AR1Model(rho=_per_minute(rho_spaced, spacing), var=var_signal)


def fit_twin(
    train: NDArray[np.float64],
    sigma_m: float,
    spacing: int,
    rank: int | None = None,
    max_rank: int = 12,
) -> TwinModel:
    """Fit the low-rank-plus-private twin by singular value decomposition.

    ``train`` is (samples x qubits) noisy frequency estimates.  If ``rank`` is None the rank is
    chosen at the largest gap of the leading squared singular values (no knowledge of the true rank).
    """
    t, n = train.shape
    _, s, vt = np.linalg.svd(train, full_matrices=False)
    if rank is None:
        s2 = s**2
        top = int(min(max_rank, n - 1, t - 1))
        # largest spectral gap s_j^2 / s_{j+1}^2 among the leading values: latent modes sit above the
        # private-plus-measurement-noise continuum, which has no knowledge of the true rank.
        gaps = s2[:top] / np.maximum(s2[1 : top + 1], 1e-12)
        rank = int(np.argmax(gaps)) + 1
        rank = int(min(rank, top))
    basis = vt[:rank].T
    if rank == 0:
        basis = np.zeros((n, 0))
    scores = train @ basis
    resid = train - scores @ basis.T
    var_resid = float(np.sum(resid**2) / (t * max(n - rank, 1)))
    var_priv = max(var_resid - sigma_m**2, 1e-6)
    var_u = np.mean(scores**2, axis=0)
    var_z = np.maximum(var_u - var_resid, 1e-6)
    if rank > 0:
        rho_z = np.array([_per_minute(c, spacing) for c in _lag_corr(scores, var_z)])
    else:
        rho_z = np.zeros(0)
    rho_priv = _per_minute(float(np.mean(resid[:-1] * resid[1:]) / var_priv), spacing)
    return TwinModel(basis, var_z, rho_z, var_priv, rho_priv)


def uniform_global_twin(base: TwinModel, n: int) -> TwinModel:
    """Rank-1 twin that assumes every qubit follows one global mode with equal loading (no learning)."""
    basis = np.full((n, 1), 1.0 / np.sqrt(n))
    var_z = np.array([float(np.sum(base.var_z))])
    rho = np.array([float(np.average(base.rho_z, weights=base.var_z))]) if base.rank else np.array([0.9])
    return TwinModel(basis, var_z * 1.0, rho, base.var_priv, base.rho_priv)


def oracle_twin(proc: DriftProcess) -> TwinModel:
    """Twin built from the true loadings and time constants (upper bound for any learned twin)."""
    q, r = np.linalg.qr(proc.loadings)
    cov = proc.sig_lat**2 * (r @ r.T)
    lam, w = np.linalg.eigh(cov)
    order = np.argsort(lam)[::-1]
    lam, w = lam[order], w[:, order]
    basis = q @ w
    rho_z = np.full(basis.shape[1], proc.rho_lat)
    return TwinModel(basis, lam, rho_z, proc.sig_priv**2, proc.rho_priv)


def select_sentinels(
    basis: NDArray[np.float64], k: int, rng: np.random.Generator, method: str = "doptimal"
) -> NDArray[np.int64]:
    """Choose ``k`` sentinel qubits: greedy D-optimal on the learned basis, or uniformly at random."""
    n, r = basis.shape
    k = int(min(k, n))
    if method == "random" or r == 0:
        return rng.choice(n, size=k, replace=False)
    m_inv = np.eye(r) / 1e-2  # inverse of (ridge * I)
    chosen: list[int] = []
    available = np.ones(n, dtype=bool)
    for _ in range(k):
        mv = basis @ m_inv
        gain = np.einsum("ij,ij->i", mv, basis)  # v^T M^-1 v
        gain = np.where(available, gain, -np.inf)
        i = int(np.argmax(gain))
        chosen.append(i)
        available[i] = False
        v = basis[i]
        mi_v = m_inv @ v
        m_inv = m_inv - np.outer(mi_v, mi_v) / (1.0 + v @ mi_v)
    return np.asarray(chosen, dtype=np.int64)
