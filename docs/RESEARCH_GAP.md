# Research gap

## State of the art (from the provided literature)

(In citations, doi = digital object identifier.)

Programmable processors have grown from tens to hundreds of individually controlled qubits: 53 superconducting qubits (Arute et al., 2019, doi:10.1038/s41586-019-1666-5), 51 and then 256 Rydberg atoms (Bernien et al., 2017, doi:10.1038/nature24622; Ebadi et al., 2021, doi:10.1038/s41586-021-03582-4). The reviews of superconducting, circuit-quantum-electrodynamics (circuit QED) and trapped-ion hardware name control errors, decoherence and scalable control as the obstacles to the next orders of magnitude (Huang et al., 2020, doi:10.1007/s11432-020-2881-9; Blais et al., 2020, doi:10.1038/s41567-020-0806-z; Bruzewicz et al., 2019, doi:10.1063/1.5088164), and the noisy intermediate-scale quantum (NISQ) review stresses that algorithms must be matched to limited, imperfect resources (Bharti et al., 2022, doi:10.1103/RevModPhys.94.015004). On the software side, recent work either pushes classical simulation of the quantum state to GPU clusters and containers, where cost still grows exponentially with qubit number (Verma and Rahwani, 2027, doi:10.1007/978-3-032-30963-1_32), or audits whether an implemented quantum program matches its intended mathematical model (Zhang and Chen, 2027, doi:10.1007/978-3-032-30693-7_17). A trainability result transfers a useful lesson: exploiting locality and structure turns an exponentially hard scaling into a polynomial one (Cerezo et al., 2021, doi:10.1038/s41467-021-21728-w).

## Related work

| Paper | Year | Role | What it does | What it does NOT cover |
|---|---|---|---|---|
| Arute et al., doi:10.1038/s41586-019-1666-5 | 2019 | landmark | 53-qubit processor, verification by classical simulation | cost of keeping all qubits calibrated as N grows |
| Bernien et al., doi:10.1038/nature24622 | 2017 | landmark | 51-atom programmable Ising simulator | drift tracking or recalibration budgets |
| Ebadi et al., doi:10.1038/s41586-021-03582-4 | 2021 | landmark | 256-atom simulator, phase diagram | per-qubit control overhead at larger arrays |
| Bharti et al., doi:10.1103/RevModPhys.94.015004 | 2022 | landmark | NISQ algorithm and benchmarking review | control-layer calibration economics |
| Cerezo et al., doi:10.1038/s41467-021-21728-w | 2021 | landmark | locality gives polynomial rather than exponential gradient decay | device drift; it is a statement about cost functions |
| Havlicek et al., doi:10.1038/s41586-019-0980-2 | 2019 | landmark | quantum feature-space classifier on a superconducting chip | calibration of the processor it runs on |
| Huang et al., doi:10.1007/s11432-020-2881-9 | 2020 | review | superconducting hardware: design, control, readout | a quantitative model of calibration cost with scale |
| Blais et al., doi:10.1038/s41567-020-0806-z | 2020 | review | circuit QED architecture and scalability challenges | a learned control model |
| Bruzewicz et al., doi:10.1063/1.5088164 | 2019 | review | trapped-ion progress, control errors at scale | calibration scheduling |
| Gaita-Arino et al., doi:10.1038/s41557-019-0232-y | 2019 | review | molecular spins; scalability via replication | drift of replicated units |
| Verma and Rahwani, doi:10.1007/978-3-032-30963-1_32 | 2027 | recent | distributed state-vector simulation up to about 25 qubits | a simulation whose cost does not grow exponentially in N |
| Zhang and Chen, doi:10.1007/978-3-032-30693-7_17 | 2027 | recent | semantic verification of an implemented quantum oracle | verification of a model of the *hardware* |
| Kashvi and Verma, doi:10.1007/978-3-032-30957-0_10 | 2027 | recent | single-qubit geometric MaxCut, motivated by circuit depth and decoherence limits | control of many physical qubits |

## Gap

The cited hardware papers calibrate and verify every qubit individually, and the reviews list scalable control as open. The recent software papers either simulate the quantum state (exponential in N) or verify program semantics. None of the papers in the brief quantifies how the *measurement cost of keeping the control layer stable* scales with qubit number, or tests whether a learned, cheap digital twin of the device's drift can replace per-qubit measurements. A twin that does not simulate the quantum state, but models the low-rank structure of correlated drift (shared temperature, flux or field fluctuations) and is validated against exact quantum simulation at the building-block level, is feasible on a laptop at N up to 16 384.

**Novelty (one sentence).** This work measures, under equal infidelity targets and a shared drift trace, how much Ramsey recalibration cost a low-rank sentinel-driven digital twin saves over periodic and per-qubit-Kalman recalibration as the array grows from 16 to 16 384 qubits, and where that saving disappears.

## Hypotheses

* **H1 (scaling).** At a mean pi-pulse infidelity target of 3e-4 the low-rank twin needs fewer Ramsey shots per qubit per hour than periodic Ramsey recalibration once N is large enough that the fixed sentinel cost is amortised; the reduction factor grows with N and saturates at a floor set by the private (non-correlated) drift. *Falsified if* the paired reduction factor is not above 1 for any N >= 1024 across 5 seeds.
* **H2 (structure matters).** The saving comes from the learned low-rank structure: replacing it by a uniform global mode, or by random sentinels, removes most of the saving.
* **H3 (limits).** The saving vanishes when the correlated share of the variance is small, and degrades with unmodelled telegraph noise and with a true rank above the sentinel budget.

**Success metric.** Paired ratio (periodic Ramsey cost at target) / (twin cost at target), mean +/- standard deviation over seeds; secondary: mean infidelity at a fixed 600 shots per qubit per hour.

**Scope statement.** Simulation only: a linear-Gaussian drift generator plus telegraph noise; Qiskit and Aer validate the gate-error and Ramsey-estimator building blocks, not the drift model itself.
