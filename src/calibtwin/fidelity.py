"""Gate infidelity of a detuned resonant pi pulse (closed form and exact check)."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

#: Rabi frequency Omega / 2 pi of the drive, in kHz (a 250 ns pi pulse).
RABI_KHZ = 2000.0


def pi_pulse_infidelity(detuning_khz: ArrayLike, rabi_khz: float = RABI_KHZ) -> NDArray[np.float64]:
    """Average gate infidelity of a pi pulse driven with frequency error ``detuning_khz``.

    The pulse is exp(-i (Omega sx + eps sz) T / 2) with Omega T = pi.  With
    r = eps / Omega the process fidelity is sin^2(theta/2) / (1 + r^2),
    theta/2 = (pi/2) sqrt(1 + r^2), and the average gate fidelity is
    (2 F_pro + 1) / 3 for one qubit.
    """
    r = np.asarray(detuning_khz, dtype=float) / rabi_khz
    w2 = 1.0 + r * r
    f_pro = np.sin(0.5 * np.pi * np.sqrt(w2)) ** 2 / w2
    return (2.0 / 3.0) * (1.0 - f_pro)


def pi_pulse_unitary(detuning_khz: float, rabi_khz: float = RABI_KHZ) -> NDArray[np.complex128]:
    """Exact 2x2 unitary of the detuned pi pulse via a matrix exponential."""
    from scipy.linalg import expm

    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    omega = 2 * np.pi * rabi_khz * 1e3
    eps = 2 * np.pi * detuning_khz * 1e3
    t_pi = np.pi / omega
    return expm(-0.5j * (omega * sx + eps * sz) * t_pi)


def exact_infidelity(detuning_khz: float, rabi_khz: float = RABI_KHZ) -> float:
    """Average gate infidelity from the matrix exponential (no closed form)."""
    ideal = pi_pulse_unitary(0.0, rabi_khz)
    v = pi_pulse_unitary(detuning_khz, rabi_khz)
    f_pro = abs(np.trace(ideal.conj().T @ v)) ** 2 / 4.0
    return float((2.0 / 3.0) * (1.0 - f_pro))
