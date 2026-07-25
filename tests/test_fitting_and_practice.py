"""Tests for throw-distribution fitting and the practice calculator."""

import numpy as np
import pytest

from darts.dartboards import generate_dartboard
from darts.fitting import (ScoreLikelihood, bootstrap_uncertainty,
                           effective_sample_size, fit_from_positions,
                           fit_from_scores, simulate_scores)
from darts.practice import (classify_points, leg_win_probability,
                            sigma_sensitivity, transition_arrays_by_class)
from darts.transitions import transition_arrays


@pytest.fixture(scope="module")
def board():
    return generate_dartboard(256)[0]


def test_fit_from_positions_recovers_the_truth():
    rng = np.random.default_rng(0)
    Sigma = np.array([[144.0, 30.0], [30.0, 64.0]])
    mu = np.array([5.0, 100.0])
    z = rng.multivariate_normal(mu, Sigma, 20000)
    f = fit_from_positions(z)
    assert f["mu"] == pytest.approx(mu, abs=0.5)
    assert f["Sigma"] == pytest.approx(Sigma, abs=4.0)


def test_score_probabilities_sum_to_one(board):
    like = ScoreLikelihood(board=board)
    p = like.score_probabilities(np.zeros(2), 15.0 ** 2 * np.eye(2))
    assert sum(p.values()) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("true_sigma", [10.0, 20.0])
def test_em_recovers_sigma_from_scores_alone(board, true_sigma):
    scores = simulate_scores(3000, [0.0, 103.0], true_sigma ** 2 * np.eye(2),
                             board=board, seed=3)
    f = fit_from_scores(scores, board=board)
    assert f["sigma_mm"] == pytest.approx(true_sigma, rel=0.15)


def test_em_log_likelihood_is_monotone(board):
    """
    The E step is an exact conditional expectation, not a Monte Carlo estimate,
    so EM must increase the observed-data log-likelihood at every step. This is
    the property that would break if the E step were wrong.
    """
    scores = simulate_scores(800, [0.0, 103.0], 15.0 ** 2 * np.eye(2),
                             board=board, seed=5)
    f = fit_from_scores(scores, board=board, max_iter=60)
    h = np.array(f["history"])
    assert (np.diff(h) > -1e-7).all()


def test_em_is_deterministic(board):
    """No Monte Carlo in the E step means the same data gives the same fit."""
    scores = simulate_scores(500, [0.0, 103.0], 15.0 ** 2 * np.eye(2),
                             board=board, seed=7)
    a = fit_from_scores(scores, board=board, max_iter=40)
    b = fit_from_scores(scores, board=board, max_iter=40)
    assert a["sigma_mm"] == b["sigma_mm"]
    assert a["mu"] == pytest.approx(b["mu"])


def test_unachievable_score_is_rejected(board):
    with pytest.raises(ValueError):
        fit_from_scores([23, 20, 20], board=board)


def test_bootstrap_reports_uncertainty(board):
    scores = simulate_scores(400, [0.0, 103.0], 15.0 ** 2 * np.eye(2),
                             board=board, seed=11)
    r = bootstrap_uncertainty(scores, n_boot=8, board_pixels=256, max_iter=40)
    assert r["sigma_mm_se"] > 0
    assert r["draws"].shape[1] == 3


def test_effective_sample_size_flags_uninformative_data():
    lots = effective_sample_size([20] * 100 + [60] * 100 + [1] * 100 + [5] * 100)
    none = effective_sample_size([60] * 400)
    assert none["distinct scores"] == 1
    assert none["entropy (nats)"] < lots["entropy (nats)"]


# --------------------------------------------------------------------------

def test_class_transitions_reduce_to_the_plain_builder():
    """Equal sigma in every class must reproduce transition_arrays exactly."""
    s = 16.0
    a = transition_arrays(256, s, point_stride=8)
    b = transition_arrays_by_class(256, {"double": s, "treble": s, "other": s},
                                   point_stride=8)
    assert np.array_equal(a["points"], b["points"])
    assert np.abs(a["probs"] - b["probs"]).max() == pytest.approx(0.0, abs=1e-12)
    assert np.abs(a["checkout_probs"] - b["checkout_probs"]).max() == pytest.approx(
        0.0, abs=1e-12)


def test_point_classes_are_sensible():
    tr = transition_arrays_by_class(256, {"double": 10.0, "treble": 10.0,
                                          "other": 10.0}, point_stride=8)
    cls = classify_points(tr["points"], 256)
    assert set(np.unique(cls)) <= {"double", "treble", "other"}
    # the trebles are a thin ring, so there must be fewer of them than "other"
    assert (cls == "treble").sum() < (cls == "other").sum()


def test_tighter_doubles_never_hurt():
    """Improving one class of target cannot make the player worse."""
    from darts.mdp_3turn import ThreeDartMDP
    base, better = 20.0, 15.0
    out = {}
    for name, smap in [("base", {"double": base, "treble": base, "other": base}),
                       ("sharp doubles", {"double": better, "treble": base,
                                          "other": base})]:
        tr = transition_arrays_by_class(256, smap, point_stride=8)
        m = ThreeDartMDP(tr["probs"], tr["checkout_probs"], tr["allowed_scores"],
                         170, dart_cost=0.0, turn_cost=1.0).solve()
        out[name] = -m.V1[170]
    assert out["sharp doubles"] <= out["base"] + 1e-9


def test_sigma_sensitivity_signs():
    sigmas = np.array([10.0, 12.0, 14.0, 16.0])
    visits = np.array([7.0, 8.0, 9.2, 10.6])       # worse with bigger sigma
    d = sigma_sensitivity(sigmas, visits)
    assert (d > 0).all()
    assert sigma_sensitivity(sigmas, visits, at=13.0) > 0


def test_leg_win_probability_is_monotone():
    """Fewer expected visits than the opponent must mean a better than even leg."""
    assert leg_win_probability(9.0, 12.0) > 0.5
    assert leg_win_probability(12.0, 9.0) < 0.5
    assert leg_win_probability(10.0, 10.0) > 0.5      # throwing first
    assert leg_win_probability(10.0, 10.0, throws_first=False) < 0.5
