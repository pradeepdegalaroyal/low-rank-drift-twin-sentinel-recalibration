"""Fast tests of the building blocks (run in a few seconds)."""
import numpy as np
import pytest

from calibtwin.analysis import cost_at_target, pareto_mask
from calibtwin.drift import DriftParams, DriftProcess, measurement_sigma_khz
from calibtwin.fidelity import exact_infidelity, pi_pulse_infidelity
from calibtwin.policies import PolicySpec
from calibtwin.simulate import run_trial
from calibtwin.twin import fit_twin, select_sentinels


def test_closed_form_matches_matrix_exponential():
    for d in (0.0, 30.0, 120.0, 400.0):
        assert pi_pulse_infidelity(d) == pytest.approx(exact_infidelity(d), abs=1e-12)


def test_infidelity_zero_at_zero_detuning_and_monotone():
    d = np.linspace(0, 500, 50)
    f = pi_pulse_infidelity(d)
    assert f[0] == pytest.approx(0.0, abs=1e-15)
    assert np.all(np.diff(f) > 0)


def test_measurement_sigma_scales_inverse_sqrt_shots():
    assert measurement_sigma_khz(400) == pytest.approx(measurement_sigma_khz(100) / 2)


def test_drift_is_stationary_with_requested_variance():
    p = DriftParams(n_qubits=256, tls_rate_per_hour=0.0)
    proc = DriftProcess(p, np.random.default_rng(1))
    x = np.array([proc.step() for _ in range(600)])
    assert np.sqrt(np.mean(x**2)) == pytest.approx(p.sigma_total_khz, rel=0.2)


def test_twin_recovers_true_rank_and_subspace():
    p = DriftParams(n_qubits=256)
    proc = DriftProcess(p, np.random.default_rng(2))
    train = proc.sample_training(144, 10, 100)
    model = fit_twin(train, measurement_sigma_khz(100), 10)
    assert model.rank == p.true_rank
    q, _ = np.linalg.qr(proc.loadings)
    overlap = np.linalg.norm(q.T @ model.basis, ord=2)  # largest principal cosine
    assert overlap > 0.95


def test_sentinels_distinct_and_dopt_better_conditioned_than_random():
    rng = np.random.default_rng(0)
    basis = np.linalg.qr(rng.standard_normal((200, 3)))[0]
    s = select_sentinels(basis, 6, rng)
    assert len(set(s.tolist())) == 6
    r = select_sentinels(basis, 6, np.random.default_rng(5), "random")
    d_opt = np.linalg.slogdet(basis[s].T @ basis[s])[1]
    d_rnd = np.linalg.slogdet(basis[r].T @ basis[r])[1]
    assert d_opt >= d_rnd


def test_cost_at_target_interpolation():
    cost = np.array([100.0, 200.0, 400.0])
    infid = np.array([4e-4, 2e-4, 1e-4])
    c = cost_at_target(cost, infid, 3e-4)
    assert 100 < c < 200
    assert np.isnan(cost_at_target(cost, infid, 1e-5))


def test_pareto_mask():
    m = pareto_mask(np.array([1.0, 2.0, 3.0]), np.array([3.0, 1.0, 2.0]))
    assert m.tolist() == [True, True, False]


def test_run_trial_twin_beats_uncorrected_error_and_charges_shots():
    p = DriftParams(n_qubits=64)
    specs = [PolicySpec("hold", 20, label="hold"), PolicySpec("twin", 20, label="twin")]
    res = run_trial(p, specs, seed=0, horizon_min=120)
    hold, twin = res
    uncorrected = float(np.mean(pi_pulse_infidelity(np.random.default_rng(0).normal(0, 120, 20000))))
    assert hold.mean_infidelity < uncorrected
    assert twin.mean_infidelity < uncorrected
    assert hold.shots_per_qubit_hour == pytest.approx(3 * 100, rel=0.1)
    assert twin.shots_per_qubit_hour > hold.shots_per_qubit_hour  # sentinels cost extra at N = 64
