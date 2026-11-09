# Method

All quantities are simulated; nothing here was run on a physical device. Frequencies are in kilohertz (kHz) and times in minutes.

## 1. Device model (the "physical" system)

A square array of N qubits; qubit i has a frequency error

delta_i(t) = sum_k L[i,k] g_k(t) + p_i(t) + s_i(t)

* **Correlated part**: r = 3 unit-variance Ornstein-Uhlenbeck (OU) modes g_k with a 60 min correlation time. The loadings L are smooth cosine profiles over the chip plus 15 % per-qubit perturbation, so the modes are *not* recoverable by spatial interpolation alone.
* **Private part** p_i: independent OU, 30 min correlation time.
* **Telegraph part** s_i: two-level-system (TLS) switching, peak-to-peak 40 kHz, 0.1 switches per qubit per hour. No model in the study predicts it.
* Total root-mean-square (rms) drift of the first two parts: 120 kHz, of which 80 % of the variance is correlated (`latent_share`).

**Gate error.** A resonant pi pulse with Rabi frequency Omega / 2 pi = 2 MHz (250 ns) driven with frequency error eps has average gate infidelity (2/3)(1 - sin^2(theta/2) / (1 + r^2)), with r = eps / Omega and theta/2 = (pi/2) sqrt(1 + r^2). This closed form is checked against `qiskit.quantum_info.average_gate_fidelity` (`validation_gate.csv`).

**Measurement.** A Ramsey experiment (free evolution tau = 1 us, operated at the quadrature point) returns a frequency estimate with standard deviation 1 / (2 pi tau sqrt(S)) for S shots. The Gaussian model is checked against shot-sampled circuits on the Aer simulator (`validation_ramsey.csv`). Cost is counted in Ramsey shots per qubit per hour.

The device parameters are illustrative for a slow-gate platform (spin or atom qubits); they are not calibrated to a specific machine.

## 2. Policies compared

| Policy | Role | What it does |
|---|---|---|
| Periodic Ramsey | standard-practice baseline (the quantum-device baseline: Ramsey recalibration on a timer) | Each qubit is re-measured every P minutes (staggered), the new estimate is applied and held. |
| Per-qubit Kalman | strong classical baseline | Same schedule, but the held value is shrunk and decayed with a per-qubit OU model fitted on commissioning data (a classical single-qubit digital twin, blind to correlations). |
| Low-rank twin (proposed) | method | A rank-r plus private model fitted by singular value decomposition (SVD) on commissioning data; K sentinel qubits are measured every 2 min and a Kalman filter tracks the r latent coordinates; every qubit's correction is V z(t) plus a shrunk, decayed private residual measured on the same staggered schedule as the baselines. |
| Oracle twin | upper bound | The twin built from the true loadings and time constants. |

*Commissioning*: 144 noisy full-chip snapshots (10 min apart, 100 shots each) per seed, charged once and reported separately from the running cost. *Rank selection*: the largest gap of the leading squared singular values (no knowledge of the true rank). *Sentinel selection*: greedy D-optimal on the learned basis (random selection is an ablation).

## 3. Metrics

* **Mean infidelity**: average gate infidelity over all qubits and a 12 h horizon (1 min steps), measured against the *true* drift.
* **Cost**: Ramsey shots per qubit per hour, including sentinel shots. The initial full calibration is common to all policies and not charged.
* **Cost at target** (primary): the shots per qubit per hour needed to reach mean infidelity 3e-4, by log-log interpolation along the period sweep P in {5, 10, 20, 40, 80, 160, 320}.
* **Infidelity at fixed budget**: mean infidelity interpolated at 600 shots per qubit per hour (used in robustness sweeps where the target may be unreachable).
* Also recorded: fraction of qubit-time above infidelity 1e-3 and the worst single-qubit mean infidelity.

## 4. Experiments

1. **Cost-quality frontier** at N = 256, all policies, 5 seeds.
2. **Scaling** with N in {16, 64, 256, 1024, 4096} (5 seeds) and N = 16384 (3 seeds, shorter period sweep).
3. **Ablation** at N = 1024: number of sentinels K in {4, 8, 16, 32, 64}; learned vs oracle vs uniform-global-mode loadings; D-optimal vs random sentinels; sentinel interval 2 vs 10 min.
4. **Robustness** at N = 256: correlated share of variance, TLS rate, true rank of the correlated drift (the twin always selects its own rank), shots per measurement.
5. **Cost model** (not simulation): the measured private-calibration cost at the largest simulated N plus the sentinel cost 60 K S / (n_s N) is evaluated at larger N and labelled as modelled.

Seeds are 0-4; all policies within a seed share the same drift trace (common random numbers), so ratios are paired. Mean and sample standard deviation (ddof = 1) over seeds are reported.

## 5. Validity limits

The twin is linear-Gaussian; the drift generator is also linear-Gaussian plus telegraph noise, which favours the twin. Every claim is conditional on that model class; see the Limitations section of the README.
