"""
Tests for fitting a Student-t throw from scores alone.

The t adds a second latent variable to a problem that already had one: not only
is the landing point unobserved, so is how wide that particular dart was. The
tests below are arranged around the two things that could go wrong silently --
that the extra weight is derived incorrectly (caught by monotonicity and by the
nu = inf limit), and that a t is being fitted where the data does not support one
(caught by running the profile on Gaussian data too).
"""

import numpy as np
import pytest

from darts.dartboards import generate_dartboard
from darts.fitting import (NU_GRID, ScoreLikelihood, fit_from_scores,
                           fit_multi_target, profile_nu, simulate_scores,
                           simulate_session)

T20 = np.array([0.0, 103.0])
DESIGN = [np.array([0.0, 103.0]), np.zeros(2), np.array([-100.0, 0.0])]


@pytest.fixture(scope="module")
def board():
    return generate_dartboard(256)[0]


# --------------------------------------------------------------------------
# The Gaussian limit
# --------------------------------------------------------------------------

def test_the_weight_is_one_for_a_gaussian(board):
    """
    Every dart the same width is what a Gaussian *is*, so the mixture weight has
    to be identically one -- and at nu = inf as well, which is the case a naive
    (nu+2)/(nu+q) would return NaN for.
    """
    Sigma = 12.0 ** 2 * np.eye(2)
    for nu in (None, np.inf):
        like = ScoreLikelihood(board=board, nu=nu)
        w = like.mixture_weight(T20, Sigma)
        assert w == pytest.approx(np.ones(len(w)))

    like = ScoreLikelihood(board=board, nu=2.25)
    w = like.mixture_weight(T20, Sigma)
    # bounded above by its value at the centre, and driven to nothing far out
    assert w.max() <= (2.25 + 2) / 2.25 + 1e-12
    assert w.min() < 1e-2
    assert (w > 0).all()


def test_nu_infinity_reproduces_the_gaussian_fit(board):
    """
    The t path uses a different density, a weighted E step and a differently
    normalised M step. At nu = inf it has to land on the Gaussian answer, or the
    weighting is wrong somewhere that a recovery test would not notice.
    """
    scores = simulate_scores(500, T20 + np.array([3.0, -4.0]),
                             12.0 ** 2 * np.eye(2), board=board, seed=1)
    a = fit_from_scores(scores, board=board, mu_init=T20, tol=1e-12, max_iter=2000)
    b = fit_from_scores(scores, board=board, mu_init=T20, tol=1e-12, max_iter=2000,
                        nu=np.inf)
    assert b["sigma_mm"] == pytest.approx(a["sigma_mm"], rel=1e-6)
    assert b["mu"] == pytest.approx(a["mu"], abs=1e-5)
    assert b["log_likelihood"] == pytest.approx(a["log_likelihood"], abs=1e-9)

    sessions = simulate_session(DESIGN, 150, [2.0, -3.0], 12.0 ** 2 * np.eye(2),
                                board=board, seed=2)
    for shared in (True, False):
        c = fit_multi_target(sessions, board=board, tol=1e-12, max_iter=2000,
                             shared_bias=shared)
        d = fit_multi_target(sessions, board=board, tol=1e-12, max_iter=2000,
                             shared_bias=shared, nu=np.inf)
        assert d["sigma_mm"] == pytest.approx(c["sigma_mm"], rel=1e-6)
        assert d["log_likelihood"] == pytest.approx(c["log_likelihood"], abs=1e-9)


def test_a_gaussian_simulation_is_unchanged_by_the_nu_argument(board):
    """The draw for nu=None must be the same darts it always was."""
    a = simulate_scores(200, T20, 15.0 ** 2 * np.eye(2), board=board, seed=7)
    b = simulate_scores(200, T20, 15.0 ** 2 * np.eye(2), board=board, seed=7,
                        nu=np.inf)
    assert (a == b).all()


# --------------------------------------------------------------------------
# The t itself
# --------------------------------------------------------------------------

def test_the_t_em_increases_the_likelihood_every_step(board):
    """
    Plain EM, unaccelerated, so nothing is protecting monotonicity but the
    derivation. This is the test that would fail if the weight belonged in the
    denominator, or if the M step divided the scale by the total weight instead
    of the dart count.
    """
    scores = simulate_scores(600, T20, 8.0 ** 2 * np.eye(2), board=board,
                             seed=11, nu=2.25)
    f = fit_from_scores(scores, board=board, mu_init=T20, nu=2.25,
                        max_iter=40, accelerate=False)
    assert (np.diff(np.array(f["history"])) > -1e-7).all()


def test_the_t_fit_recovers_a_t_and_the_gaussian_does_not(board):
    """
    Simulate a known Student-t and fit it both ways. The t must recover the core
    scale; the Gaussian must not, because there is no Gaussian that is both as
    tight as the core and as wide as the tail -- it returns a compromise
    describing neither, which is the effect notebook 21 measured on real players
    (5.98mm core against 11.47mm fitted as a Gaussian).
    """
    scale, nu, b_true = 8.0, 2.25, np.array([2.0, -3.0])
    sessions = simulate_session(DESIGN, 250, b_true, scale ** 2 * np.eye(2),
                                board=board, seed=3, nu=nu)
    t = fit_multi_target(sessions, board=board, tol=1e-11, max_iter=1500, nu=nu)
    g = fit_multi_target(sessions, board=board, tol=1e-11, max_iter=1500)

    assert t["sigma_mm"] == pytest.approx(scale, rel=0.15)
    assert g["sigma_mm"] > 1.5 * scale
    # Same parameter count -- nu is fixed, not fitted -- so any gap favours the
    # t outright. On these 750 darts it is about 57 log-units, 0.076 a dart.
    assert t["log_likelihood"] > g["log_likelihood"] + 25

    # The bias is a different story and is not claimed to improve: on this design
    # both fits miss it by about 3.5mm, almost entirely in y, and which one misses
    # by less is a coin toss. What the t buys is the scale, not the aim.
    assert np.linalg.norm(t["b"] - b_true) < 5.0


def test_the_scale_and_nu_trade_off_along_a_ridge(board):
    """
    Notebook 21 found nu and the core scale correlated at +0.62 across players,
    which is why nu is profiled rather than reported as a point. The fitted scale
    must therefore rise monotonically with the nu it was fitted at.
    """
    sessions = simulate_session(DESIGN, 200, [2.0, -3.0], 8.0 ** 2 * np.eye(2),
                                board=board, seed=4, nu=2.25)
    scales = [fit_multi_target(sessions, board=board, tol=1e-10, max_iter=1200,
                               nu=nu)["sigma_mm"]
              for nu in (2.25, 3.0, 6.0, np.inf)]
    assert scales == sorted(scales)


# --------------------------------------------------------------------------
# Profiling nu, in both directions
# --------------------------------------------------------------------------

def test_the_profile_finds_the_nu_it_was_given(board):
    grid = (2.05, 2.25, 3.0, 6.0, np.inf)
    sessions = simulate_session(DESIGN, 250, [2.0, -3.0], 8.0 ** 2 * np.eye(2),
                                board=board, seed=5, nu=2.25)
    out = profile_nu(fit_multi_target, sessions, nu_grid=grid, board=board,
                     tol=1e-10, max_iter=1200)
    # Scores pin down *that* there is a tail far better than *how heavy* it is:
    # the peak wanders over the low end of the grid from sample to sample, so
    # asserting the exact nu would be asserting noise.
    assert out["best_nu"] <= 3.0
    assert out["identified"]
    # The margin over the Gaussian is the part that is not noise.
    assert out["best_vs_gaussian"] > 25
    assert [r["nu"] for r in out["profile"]] == [2.05, 2.25, 3.0, 6.0, np.inf]


def test_the_profile_does_not_invent_a_tail(board):
    """
    The mirror of the test above, and the one that makes it mean something. On
    darts that really are Gaussian the profile must land at the top of the grid
    and the tail must buy nothing worth having.
    """
    grid = (2.25, 3.0, 6.0, 20.0, np.inf)
    sessions = simulate_session(DESIGN, 250, [2.0, -3.0], 12.0 ** 2 * np.eye(2),
                                board=board, seed=6)
    out = profile_nu(fit_multi_target, sessions, nu_grid=grid, board=board,
                     tol=1e-10, max_iter=1200)
    # It may still nominate a large finite nu -- the profile is flat up there --
    # but what it cannot do is find a tail worth having. Against the +25 and more
    # that real t darts buy, a log-unit is nothing.
    assert out["best_nu"] >= 20.0
    assert out["best_vs_gaussian"] < 1.0


def test_the_default_grid_brackets_the_gaussian_and_the_fitted_range():
    """NU_GRID has to contain its own null, or the profile has nothing to say."""
    assert np.isinf(NU_GRID[-1])
    assert min(NU_GRID) < 2.1              # below the tightest player 21 fitted
    assert any(10.0 <= nu < np.inf for nu in NU_GRID)   # and above the loosest
    assert all(nu > 2.0 for nu in NU_GRID)              # finite scale matrix


# --------------------------------------------------------------------------
# The off-board mass
# --------------------------------------------------------------------------

def test_a_t_score_distribution_still_sums_to_one(board):
    """
    A t puts real mass beyond the board array and a dart landing there scores
    nothing. Booked as a miss, the probabilities are a distribution; dropped,
    they are not, and every likelihood would be quietly mis-normalised.
    """
    Sigma = 8.0 ** 2 * np.eye(2)
    for nu in (2.25, 3.0, np.inf):
        p = ScoreLikelihood(board=board, nu=nu).score_probabilities(T20, Sigma)
        assert sum(p.values()) == pytest.approx(1.0, abs=1e-9)
    heavy = ScoreLikelihood(board=board, nu=2.25).score_probabilities(T20, Sigma)
    light = ScoreLikelihood(board=board, nu=np.inf).score_probabilities(T20, Sigma)
    assert heavy[0] > light[0] + 1e-3
