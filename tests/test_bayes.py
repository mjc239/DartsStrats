"""Tests for the Bayesian live fit and recommendation, and the power rule."""

import numpy as np
import pytest

from darts.bayes import (BayesRecommender, SigmaPosterior, band_prior,
                         flat_prior, play_leg)
from darts.dartboards import generate_dartboard
from darts.design import darts_to_detect


@pytest.fixture(scope="module")
def board_and_checkouts():
    return generate_dartboard(256)


@pytest.fixture(scope="module")
def recommender():
    # a deliberately coarse grid so the module-scoped solve stays quick
    return BayesRecommender(np.array([10.0, 16.0, 22.0, 28.0]),
                            board_pixels=128, point_stride=2)


def test_band_prior_is_a_distribution_peaked_at_the_band():
    sig = np.round(np.arange(5.0, 32.1, 0.5), 1)
    p = band_prior("club", sig)
    assert p.sum() == pytest.approx(1.0)
    assert sig[p.argmax()] == pytest.approx(20.0)
    # elite is implausible under a club prior, not impossible
    assert p[np.searchsorted(sig, 6.5)] < 0.01 * p.max()


def test_likelihood_matches_score_likelihood(board_and_checkouts):
    """
    The per-dart likelihood must agree with ScoreLikelihood, which computes
    the same quantity by evaluating the density over the whole board.
    """
    from darts.fitting import ScoreLikelihood
    board, checkouts = board_and_checkouts
    like = ScoreLikelihood(board=board)
    sig = np.array([10.0, 20.0])
    post = SigmaPosterior(sig, board=board, checkouts=checkouts)
    aim_mm = np.array([0.0, 103.0])

    ll = post._log_likelihood(aim_mm, 60)
    for k, s in enumerate(sig):
        want = like.score_probabilities(aim_mm, s ** 2 * np.eye(2))[60]
        assert np.exp(ll[k]) == pytest.approx(want, rel=1e-10)


def test_checkout_observation_is_more_informative(board_and_checkouts):
    """Knowing the dart hit the double bed must never raise the likelihood
    above that of the bare score value, since the bed is a subset."""
    board, checkouts = board_and_checkouts
    sig = np.array([8.0, 16.0, 24.0])
    post = SigmaPosterior(sig, board=board, checkouts=checkouts)
    aim_mm = np.array([0.0, 166.0])       # at D20
    ll_any = post._log_likelihood(aim_mm, 40)
    ll_dbl = post._log_likelihood(aim_mm, 40, checkout=True)
    assert (ll_dbl <= ll_any + 1e-12).all()


def test_posterior_concentrates_on_the_truth(board_and_checkouts):
    """Feeding many darts at varied targets must pull the posterior mean to
    the true sigma, whatever the prior said."""
    board, checkouts = board_and_checkouts
    from darts.utils import mm_per_pixel
    px = board.shape[0]
    mmpp = mm_per_pixel(px)
    true_sigma = 20.0
    rng = np.random.default_rng(0)
    sig = np.round(np.arange(5.0, 32.1, 0.5), 1)
    post = SigmaPosterior(sig, band_prior("county", sig),  # wrong prior: 10mm
                          board=board, checkouts=checkouts)

    targets_mm = [np.array([0.0, 103.0]), np.array([-80.0, -40.0]),
                  np.array([0.0, 0.0]), np.array([100.0, 60.0])]
    for i in range(400):
        t = targets_mm[i % len(targets_mm)]
        land = t + rng.normal(0.0, true_sigma, 2)
        col = int(round(land[0] / mmpp)) + px // 2
        row = int(round(land[1] / mmpp)) + px // 2
        v = int(board[row, col]) if 0 <= row < px and 0 <= col < px else 0
        post.update(t, v, pixel=False)

    assert post.mean() == pytest.approx(true_sigma, abs=2.0)
    lo, hi = post.interval(0.9)
    assert lo < true_sigma < hi


def test_point_mass_recommendation_is_that_sigmas_policy(recommender):
    """With all posterior mass at one solved sigma, the Bayes action must be
    exactly that model's optimal action."""
    sig = recommender.solve_sigmas
    post = SigmaPosterior(sig, board=recommender.board,
                          checkouts=recommender.checkouts)
    post.log_post = np.where(sig == 16.0, 0.0, -1e9)
    m = recommender.models[list(sig).index(16.0)]
    for score, dart in [(501, 1), (170, 1), (40, 3), (61, 2)]:
        a = recommender.recommend(post, score, dart, score)
        assert a == int(np.argmax(m.q_values(score, dart, score)))


def test_bayes_action_beats_the_worst_fixed_assumption(recommender):
    """Averaged over the posterior, the Bayes action is optimal by
    construction; check the machinery agrees for a spread posterior."""
    sig = recommender.solve_sigmas
    post = SigmaPosterior(sig, board=recommender.board,
                          checkouts=recommender.checkouts)   # flat over grid
    w = recommender.weights_from(post)
    for score, dart in [(170, 1), (81, 2)]:
        q = recommender.qbar(w, score, dart, score)
        a = recommender.recommend(post, score, dart, score)
        assert q[a] == pytest.approx(q.max())
        # every fixed-sigma policy is available, so the Bayes action can be
        # no worse than any of them under the posterior-averaged Q
        for m in recommender.models:
            a_fixed = int(np.argmax(m.q_values(score, dart, score)))
            assert q[a] >= q[a_fixed] - 1e-12


def test_play_leg_finishes_and_updates(recommender):
    sig = np.round(np.arange(6.0, 30.1, 1.0), 1)
    post = SigmaPosterior(sig, band_prior("club", sig),
                          board=recommender.board, checkouts=recommender.checkouts)
    rng = np.random.default_rng(1)
    rec = []
    darts = play_leg(recommender, post, true_sigma=20.0, rng=rng, record=rec)
    assert darts == len(rec)
    assert post.n_updates == darts
    assert 9 <= darts <= 200
    assert all(r["value_loss"] >= -1e-12 for r in rec)


def test_oracle_policy_needs_no_posterior(recommender):
    sig = recommender.solve_sigmas
    post = SigmaPosterior(sig, board=recommender.board,
                          checkouts=recommender.checkouts)
    rng = np.random.default_rng(2)
    darts = play_leg(recommender, post, true_sigma=16.0, rng=rng, policy="oracle")
    assert 9 <= darts <= 200


# --------------------------------------------------------------------------

def test_darts_to_detect_scalings():
    n = darts_to_detect(10.0, 1.0)
    assert n == pytest.approx(2 * (10.0 * 2.8016) ** 2, rel=1e-3)
    # quarter the improvement -> sixteen times the darts
    assert darts_to_detect(10.0, 0.25) == pytest.approx(16 * n, rel=1e-9)
    # against a known baseline, half the darts
    assert darts_to_detect(10.0, 1.0, sessions=1) == pytest.approx(n / 2, rel=1e-9)
    # a league player at the best target: S ~ 10.8, so ~1800 darts a session
    assert 1500 < darts_to_detect(10.83, 1.0) < 2000
