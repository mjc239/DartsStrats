"""
Tests for the measurement-design machinery.

The load-bearing claim in :mod:`darts.design` is that the analytic derivatives
of the score distribution are right. Everything else -- the information
matrices, the optimal designs, the equivalence certificates -- is built on top
of them, and an error there would produce plausible-looking designs that were
simply wrong. So the first test differentiates the score probabilities a
completely different way (finite differences of the direct, non-FFT likelihood)
and demands agreement.
"""

import numpy as np
import pytest

from darts.dartboards import generate_dartboard
from darts.design import (best_pair, best_single_target, c_criterion,
                          candidate_targets, design_information,
                          equivalence_certificate, greedy_design,
                          information_at_points, information_maps,
                          kernel_derivatives, optimal_design, robust_design,
                          sigma_gradient)
from darts.fitting import (ScoreLikelihood, fit_from_scores, fit_multi_target,
                           simulate_scores, simulate_session)
from darts.transitions import _correlate_fft
from darts.utils import mm_per_pixel

PX = 128


@pytest.fixture(scope="module")
def board():
    return generate_dartboard(PX)[0]


def test_kernel_derivatives_match_finite_differences(board):
    """
    The analytic derivatives of the score probabilities must agree with
    numerically differentiating an independent implementation.

    ``ScoreLikelihood`` evaluates the Gaussian directly on the pixel grid; the
    design code convolves score masks with an analytically differentiated
    kernel. They share no code beyond the board itself.
    """
    like = ScoreLikelihood(board=board)
    Sigma = np.array([[16.0 ** 2, 40.0], [40.0, 13.0 ** 2]])
    mm = mm_per_pixel(PX)
    allowed = np.unique(board).astype(int)
    target = (PX // 2 - 20, PX // 2 + 12)

    def p_direct(b, S):
        mu = np.array([(target[1] - PX // 2) * mm + b[0],
                       (target[0] - PX // 2) * mm + b[1]])
        pr = like.score_probabilities(mu, S)
        return np.array([pr[int(s)] for s in allowed])

    numeric = []
    h = 1e-4
    for k in range(2):
        e = np.zeros(2)
        e[k] = h
        numeric.append((p_direct(e, Sigma) - p_direct(-e, Sigma)) / (2 * h))
    for a, b in [(0, 0), (0, 1), (1, 1)]:
        E = np.zeros((2, 2))
        E[a, b] = E[b, a] = 1.0
        hh = 1e-4 * Sigma[0, 0]
        numeric.append((p_direct(np.zeros(2), Sigma + hh * E)
                        - p_direct(np.zeros(2), Sigma - hh * E)) / (2 * hh))

    _, dK = kernel_derivatives(PX, Sigma, mm)
    masks = np.stack([(board == s).astype(float) for s in allowed])
    analytic = np.stack([_correlate_fft(masks, d)[:, target[0], target[1]]
                         for d in dK])

    for i in range(5):
        assert analytic[i] == pytest.approx(numeric[i], abs=2e-9)


def test_probability_derivatives_sum_to_zero(board):
    """The score probabilities sum to 1, so their derivatives must sum to 0."""
    maps = information_maps(PX, 14.0, board=board)
    Sigma = 14.0 ** 2 * np.eye(2)
    _, dK = kernel_derivatives(PX, Sigma, mm_per_pixel(PX))
    allowed = np.unique(board).astype(int)
    masks = np.stack([(board == s).astype(float) for s in allowed])
    for d in dK:
        g = _correlate_fft(masks, d)
        assert np.abs(g.sum(axis=0)).max() < 1e-12


def test_information_matrices_are_symmetric_psd(board):
    maps = information_maps(PX, 15.0, board=board)
    pts = candidate_targets(PX, point_stride=8)
    I = information_at_points(maps, pts)
    assert np.abs(I - I.transpose(0, 2, 1)).max() < 1e-18
    eigs = np.linalg.eigvalsh(I)
    assert eigs.min() > -1e-15


def test_c_criterion_matches_a_direct_solve(board):
    maps = information_maps(PX, 15.0, board=board)
    I = information_at_points(maps, candidate_targets(PX, point_stride=16))
    c = sigma_gradient(15.0)
    got = c_criterion(I, c)
    want = np.array([c @ np.linalg.solve(M, c) for M in I])
    assert got == pytest.approx(want, rel=1e-6)


def test_splitting_never_hurts_and_the_optimum_is_certified(board):
    """
    Information is additive, so the achievable set is the convex hull of the
    per-target matrices and the optimum can be no worse than the best single
    target. The equivalence theorem certificate proves optimality: it is >= 1
    for any design and equals 1 only at the optimum.
    """
    maps = information_maps(PX, 16.0, board=board)
    pts = candidate_targets(PX, point_stride=3)
    I = information_at_points(maps, pts)
    c = sigma_gradient(16.0)

    _, v1, _ = best_single_target(I, c)
    opt = optimal_design(I, c)
    assert opt["value"] <= v1 + 1e-12
    assert opt["certificate"] == pytest.approx(1.0, abs=5e-3)

    # any design at all has certificate >= 1
    for k in (1, 2, 4):
        idx, _ = greedy_design(I, c, k)
        M = design_information(I[idx], np.ones(k))
        assert equivalence_certificate(I, c, M) >= 1.0 - 1e-9


def test_more_targets_never_hurt_when_weights_are_free(board):
    """
    Adding a target can only help if the weights are free, because the old
    design remains available. Greedy selection with exchange refinement should
    not go backwards for these k.
    """
    maps = information_maps(PX, 16.0, board=board)
    I = information_at_points(maps, candidate_targets(PX, point_stride=4))
    c = sigma_gradient(16.0)
    vals = [greedy_design(I, c, k)[1] for k in (1, 2, 3, 4)]
    assert all(vals[i + 1] <= vals[i] + 1e-9 for i in range(len(vals) - 1))


def test_exhaustive_pair_beats_greedy_pair(board):
    maps = information_maps(PX, 16.0, board=board)
    I = information_at_points(maps, candidate_targets(PX, point_stride=6))
    c = sigma_gradient(16.0)
    _, v_ex = best_pair(I, c)
    _, v_greedy = greedy_design(I, c, 2)
    assert v_ex <= v_greedy + 1e-12


def test_a_tight_player_is_measured_best_at_the_bull(board):
    """
    The bull is the finest concentric structure on the board, so for a player
    whose scatter is comparable to its radii it is the most informative target
    by a wide margin -- and for a loose player it is nearly useless, because
    they never hit it.
    """
    c_tight = sigma_gradient(6.5)
    tight = information_maps(PX, 6.5, board=board)
    centre = PX // 2
    mm = mm_per_pixel(PX)
    t20 = (int(round(centre + 103 / mm)), centre)

    se_bull = c_criterion(tight["info"][centre, centre], c_tight)
    se_t20 = c_criterion(tight["info"][t20[0], t20[1]], c_tight)
    assert se_bull < se_t20

    loose = information_maps(PX, 28.0, board=board)
    c_loose = sigma_gradient(28.0)
    assert (c_criterion(loose["info"][centre, centre], c_loose)
            > c_criterion(loose["info"][t20[0], t20[1]], c_loose))


def test_isotropic_information_is_the_constrained_full_information(board):
    """
    The isotropic parameterisation is the full one restricted to
    ``Sigma = sigma^2 I``, so its information must be exactly ``J^T I J`` for
    the Jacobian ``J`` of that embedding. This is a much stronger check than
    comparing variances, and it pins down the units and chain rule in
    :func:`kernel_derivatives`.
    """
    sigma = 15.0
    pts = candidate_targets(PX, point_stride=8)
    full = information_at_points(information_maps(PX, sigma, board=board), pts)
    iso = information_at_points(
        information_maps(PX, sigma, params="isotropic", board=board), pts)

    J = np.zeros((5, 3))
    J[0, 0] = J[1, 1] = 1.0
    J[2, 2] = J[4, 2] = 2 * sigma          # d Sigma_xx / d sigma
    assert np.abs(iso - np.einsum("ai,pab,bj->pij", J, full, J)).max() < 1e-15


def test_isotropic_model_is_no_less_precise_than_the_full_one(board):
    """
    Estimating three parameters instead of five cannot make sigma less
    precise, since the constrained model's information is a restriction of the
    unconstrained one. Compared relatively: at the bull the two coincide, and
    inverting the 5x5 there costs a few significant figures.
    """
    pts = candidate_targets(PX, point_stride=8)
    full = information_at_points(information_maps(PX, 15.0, board=board), pts)
    iso = information_at_points(
        information_maps(PX, 15.0, params="isotropic", board=board), pts)
    v_full = c_criterion(full, sigma_gradient(15.0))
    v_iso = c_criterion(iso, sigma_gradient(15.0, params="isotropic"))
    assert (v_iso <= v_full * (1 + 1e-6)).all()


def test_robust_design_efficiency_is_bounded_by_one(board):
    scenarios = []
    pts = candidate_targets(PX, point_stride=8)
    for sigma in (8.0, 16.0, 28.0):
        I = information_at_points(information_maps(PX, sigma, board=board), pts)
        c = sigma_gradient(sigma)
        scenarios.append((I, c, optimal_design(I, c)["value"]))
    idx, worst, effs = robust_design(scenarios, 3, n_restarts=3)
    assert len(idx) == 3
    assert 0.0 < worst <= 1.0 + 1e-9
    assert all(e <= 1.0 + 1e-9 for e in effs)
    assert min(effs) == pytest.approx(worst)


# --------------------------------------------------------------------------
# The multi-target fit these designs are for
# --------------------------------------------------------------------------

def test_multi_target_reduces_to_the_single_target_fit():
    """With one target and a matched start, the two fits must be identical."""
    board = generate_dartboard(256)[0]
    t = np.array([0.0, 103.0])
    scores = simulate_scores(600, t + np.array([3.0, -4.0]),
                             15.0 ** 2 * np.eye(2), board=board, seed=1)
    a = fit_from_scores(scores, board=board, mu_init=t, tol=1e-12, max_iter=2000)
    b = fit_multi_target([(t, scores)], board=board, tol=1e-12, max_iter=2000)
    assert a["log_likelihood"] == pytest.approx(b["log_likelihood"], abs=1e-8)
    assert a["mu"] == pytest.approx(t + b["b"], abs=1e-4)


def test_multi_target_em_is_monotone_and_recovers_the_truth():
    board = generate_dartboard(256)[0]
    b_true = np.array([3.0, -4.0])
    Sigma = 15.0 ** 2 * np.eye(2)
    design = [np.array([0.0, 103.0]), np.array([-60.0, -40.0]), np.zeros(2)]
    sessions = simulate_session(design, 600, b_true, Sigma, board=board, seed=4)
    f = fit_multi_target(sessions, board=board)
    assert (np.diff(np.array(f["history"])) > -1e-7).all()
    assert f["sigma_mm"] == pytest.approx(15.0, rel=0.15)
    assert f["b"] == pytest.approx(b_true, abs=4.0)


def test_free_means_fit_at_least_as_well_as_a_shared_bias():
    """A strictly larger model cannot have a lower maximised likelihood."""
    board = generate_dartboard(256)[0]
    design = [np.array([0.0, 103.0]), np.array([-60.0, -40.0])]
    sessions = simulate_session(design, 300, np.array([2.0, -2.0]),
                                14.0 ** 2 * np.eye(2), board=board, seed=9)
    shared = fit_multi_target(sessions, board=board)
    free = fit_multi_target(sessions, board=board, shared_bias=False)
    assert free["log_likelihood"] >= shared["log_likelihood"] - 1e-6


def test_acceleration_does_not_change_the_answer():
    """SQUAREM is a convergence accelerator, not a different estimator."""
    board = generate_dartboard(256)[0]
    scores = simulate_scores(400, [0.0, 103.0], 15.0 ** 2 * np.eye(2),
                             board=board, seed=2)
    fast = fit_from_scores(scores, board=board, tol=1e-12, max_iter=3000)
    slow = fit_from_scores(scores, board=board, tol=1e-12, max_iter=6000,
                           accelerate=False)
    assert fast["sigma_mm"] == pytest.approx(slow["sigma_mm"], rel=1e-3)
    assert fast["n_em_steps"] < slow["n_em_steps"]


@pytest.mark.slow
def test_fisher_prediction_matches_the_spread_of_actual_fits():
    """
    The whole design calculation is only useful if its predicted standard
    error is the one you actually get. Fit many simulated sessions at a
    sample size large enough for the asymptotics to bite, and compare.
    """
    board = generate_dartboard(256)[0]
    sigma, n, reps = 16.0, 1200, 40
    maps = information_maps(256, sigma, board=board)
    c = sigma_gradient(sigma)
    centre, mm = 128, mm_per_pixel(256)
    target_px = (centre + int(round(134 / mm)), centre)
    target_mm = np.array([0.0, 134.0])
    predicted = np.sqrt(c_criterion(maps["info"][target_px], c) / n)

    fits = []
    for seed in range(reps):
        sessions = simulate_session([target_mm], n, np.zeros(2),
                                    sigma ** 2 * np.eye(2), board=board,
                                    seed=seed)
        fits.append(fit_multi_target(sessions, board=board)["sigma_mm"])
    observed = np.std(fits, ddof=1)
    assert observed == pytest.approx(predicted, rel=0.35)
