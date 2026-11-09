"""Cross-checks of the two analytic building blocks against Qiskit and Aer.

1. The closed-form pi-pulse infidelity against ``qiskit.quantum_info.average_gate_fidelity``.
2. The Gaussian Ramsey-estimator model (std = 1 / (2 pi tau sqrt(shots))) against shot-sampled
   Ramsey circuits run on the Aer simulator.
"""
from __future__ import annotations

import numpy as np

from .drift import RAMSEY_TAU_US, measurement_sigma_khz
from .fidelity import RABI_KHZ, pi_pulse_infidelity, pi_pulse_unitary


def gate_fidelity_check(detunings_khz: np.ndarray) -> list[dict[str, float]]:
    """Closed form vs Qiskit average gate fidelity for a list of detunings."""
    from qiskit.quantum_info import Operator, average_gate_fidelity

    ideal = Operator(pi_pulse_unitary(0.0))
    rows = []
    for d in detunings_khz:
        qk = 1.0 - average_gate_fidelity(Operator(pi_pulse_unitary(float(d))), ideal)
        rows.append({"detuning_khz": float(d), "closed_form": float(pi_pulse_infidelity(d)), "qiskit": float(qk)})
    return rows


def ramsey_estimator_check(
    shots_list: list[int], true_khz: float = 20.0, repeats: int = 300, seed: int = 3
) -> list[dict[str, float]]:
    """Empirical std of the quadrature-point Ramsey estimate on Aer vs the model sigma."""
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    tau = RAMSEY_TAU_US * 1e-6
    phase = 2 * np.pi * true_khz * 1e3 * tau
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.rz(phase + np.pi / 2, 0)  # accumulated detuning phase plus the quadrature offset
    qc.h(0)
    qc.measure(0, 0)
    sim = AerSimulator()
    k = 0
    rows = []
    for shots in shots_list:
        ests = []
        for _ in range(repeats):
            k += 1
            counts = sim.run(qc, shots=shots, seed_simulator=seed * 100003 + k).result().get_counts()
            p1 = counts.get("1", 0) / shots
            # P(1) = (1 + sin(phase)) / 2, so arcsin(2 P(1) - 1) recovers the detuning phase
            phi = np.arcsin(np.clip(2 * p1 - 1, -1, 1))
            ests.append(phi / (2 * np.pi * tau) / 1e3)
        ests_arr = np.asarray(ests)
        rows.append(
            {
                "shots": float(shots),
                "empirical_mean_khz": float(ests_arr.mean()),
                "empirical_std_khz": float(ests_arr.std(ddof=1)),
                "model_std_khz": float(measurement_sigma_khz(shots)),
                "true_khz": true_khz,
            }
        )
    return rows


__all__ = ["gate_fidelity_check", "ramsey_estimator_check", "RABI_KHZ"]
