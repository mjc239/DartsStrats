"""Tests for the anisotropic, biased throwing model."""

import numpy as np
import pytest

from darts.bayes import SigmaPosterior
from darts.dartboards import generate_dartboard
from darts.mdp_3turn import ThreeDartMDP
from darts.throw_shape import (ShapeRecommender, ThrowPosterior, bias_grid,
                               bias_prior, isotropic_equivalent, play_leg,
                               shape_prior, shift_indices, sigma_matrices)
from darts.transitions import transition_arrays
from darts.utils import mm_per_pixel


@pytest.fixture(scope="module")
def board_and_checkouts():
    return generate_dartboard(256)


@pytest.fixture(scope="module")
def small_recommender():
    Sigmas, _ = sigma_matrices([12.0, 20.0], [12.0, 20.0])
    return ShapeRecommender(Sigmas, board_pixels=128, point_stride=2)


def test_likelihood_matches_the_isotropic_implementation(board_and_checkouts):
    """With a spherical Sigma and no bias this must reproduce SigmaPosterior,
    which computes the same quantity a different way."""
    board, co = board_and_checkouts
    sig = np.array([10.0, 16.0])
    iso = np.stack([s ** 2 * np.eye(2) for s in sig])
    tp = ThrowPosterior(iso, board=board, checkouts=co)
    sp = SigmaPosterior(sig, board=board, checkouts=co)
    aim = np.array([0.0, 103.0])
    assert tp._log_likelihood(aim, 60)[:, 0] == pytest.approx(
        sp._log_likelihood(aim, 60), abs=1e-12)


def test_bias_is_equivalent_to_moving_the_aim(board_and_checkouts):
    """N(t + b, Sigma) evaluated at t is N(t', Sigma) evaluated at t' = t + b.
    This identity is what the whole shift theorem rests on."""
    board, co = board_and_checkouts
    iso = (14.0 ** 2 * np.eye(2))[None]
    b = np.array([5.0, -7.0])
    biased = ThrowPosterior(iso, biases=b[None], board=board, checkouts=co)
    plain = ThrowPosterior(iso, board=board, checkouts=co)
    aim = np.array([0.0, 103.0])
    assert biased._log_likelihood(aim, 60)[0, 0] == pytest.approx(
        plain._log_likelihood(aim + b, 60)[0, 0], abs=1e-12)


def test_anisotropy_is_visible_in_the_scores(board_and_checkouts):
    """A tall narrow throw and a short wide one with the same isotropic
    equivalent must give different score likelihoods -- otherwise no amount of
    data could tell them apart."""
    board, co = board_and_checkouts
    tall = np.array([[12.0 ** 2, 0.0], [0.0, 20.0 ** 2]])
    wide = np.array([[20.0 ** 2, 0.0], [0.0, 12.0 ** 2]])
    assert isotropic_equivalent(tall) == pytest.approx(isotropic_equivalent(wide))
    tp = ThrowPosterior(np.stack([tall, wide]), board=board, checkouts=co)
    ll = tp._log_likelihood(np.array([0.0, 103.0]), 60)[:, 0]
    assert abs(ll[0] - ll[1]) > 0.05


def test_posterior_recovers_an_anisotropic_throw(board_and_checkouts):
    """Fed darts from a tall narrow thrower at varied targets, the posterior
    must put its mass on tall narrow covariances."""
    board, co = board_and_checkouts
    px = board.shape[0]
    mmpp = mm_per_pixel(px)
    true_S = np.array([[11.0 ** 2, 0.0], [0.0, 21.0 ** 2]])
    Sigmas, labels = sigma_matrices(np.arange(8.0, 25.0, 2.0),
                                    np.arange(8.0, 25.0, 2.0))
    post = ThrowPosterior(Sigmas, board=board, checkouts=co)

    rng = np.random.default_rng(0)
    L = np.linalg.cholesky(true_S)
    targets = [np.array([0.0, 103.0]), np.array([-80.0, -40.0]),
               np.array([0.0, 0.0]), np.array([100.0, 60.0])]
    for i in range(500):
        t = targets[i % len(targets)]
        land = t + L @ rng.standard_normal(2)
        col = int(round(land[0] / mmpp)) + px // 2
        row = int(round(land[1] / mmpp)) + px // 2
        v = int(board[row, col]) if 0 <= row < px and 0 <= col < px else 0
        post.update(t, v, pixel=False)

    sx, sy = post.mean_sigma_xy()
    assert sx == pytest.approx(11.0, abs=3.0)
    assert sy == pytest.approx(21.0, abs=3.0)
    assert post.mean_ratio() > 1.3          # the shape, not just the size


def test_posterior_recovers_a_bias(board_and_checkouts):
    board, co = board_and_checkouts
    px = board.shape[0]
    mmpp = mm_per_pixel(px)
    true_b = np.array([6.0, -9.0])
    Sigmas = np.stack([s ** 2 * np.eye(2) for s in np.arange(10.0, 25.0, 2.0)])
    biases = bias_grid(np.arange(-12.0, 12.1, 3.0), np.arange(-12.0, 12.1, 3.0))
    post = ThrowPosterior(Sigmas, biases, board=board, checkouts=co)

    rng = np.random.default_rng(1)
    targets = [np.array([0.0, 103.0]), np.array([-80.0, -40.0]),
               np.array([0.0, 0.0]), np.array([100.0, 60.0])]
    for i in range(500):
        t = targets[i % len(targets)]
        land = t + true_b + rng.normal(0.0, 16.0, 2)
        col = int(round(land[0] / mmpp)) + px // 2
        row = int(round(land[1] / mmpp)) + px // 2
        v = int(board[row, col]) if 0 <= row < px and 0 <= col < px else 0
        post.update(t, v, pixel=False)

    assert post.mean_bias() == pytest.approx(true_b, abs=4.0)


def test_priors_are_distributions():
    Sigmas, _ = sigma_matrices([10.0, 16.0, 22.0], [10.0, 16.0, 22.0])
    p = shape_prior(Sigmas, "league")
    assert p.sum() == pytest.approx(1.0)
    # the round covariance nearest the band should be the most likely
    best = Sigmas[int(np.argmax(p))]
    assert isotropic_equivalent(best) == pytest.approx(16.0, abs=3.0)
    b = bias_grid([-6.0, 0.0, 6.0], [-6.0, 0.0, 6.0])
    pb = bias_prior(b)
    assert pb.sum() == pytest.approx(1.0)
    assert pb[np.argmax(pb)] == pytest.approx(pb[(np.abs(b).sum(axis=1) == 0)][0])


def test_shift_indices_are_a_translation():
    pts = np.stack(np.meshgrid(np.arange(0, 40, 2), np.arange(0, 40, 2),
                               indexing="ij"), axis=-1).reshape(-1, 2)
    idx, exact = shift_indices(pts, np.array([4.0, -2.0]), 40)
    interior = (pts[:, 0] < 36) & (pts[:, 1] >= 2)
    assert exact[interior].all()
    assert (pts[idx][interior] == (pts + np.array([4, -2]))[interior]).all()


def test_a_known_bias_is_almost_free(board_and_checkouts):
    """
    The shift theorem: aiming at ``a`` under bias ``b`` lands where aiming at
    ``a + b`` would with no bias, so the biased game is the unbiased game with
    a translated action set. Values must therefore agree, up to the candidate
    grid being finite -- points near the rim translate off it.
    """
    px = 256
    S = 16.0 ** 2 * np.eye(2)
    tr = transition_arrays(px, 0.0, margin_mm=10.0, point_stride=2, Sigma_mm=S)
    P, CP, SC, pts = (tr["probs"], tr["checkout_probs"],
                      tr["allowed_scores"], tr["points"])
    base = ThreeDartMDP(P, CP, SC, 501, dart_cost=0.0, turn_cost=1.0).solve()

    mmpp = mm_per_pixel(px)
    for bx, by, tol in [(2 * mmpp, 0.0, 1e-9), (4 * mmpp, -4 * mmpp, 5e-3)]:
        idx, _ = shift_indices(pts, np.array([by, bx]) / mmpp, px)
        m = ThreeDartMDP(np.ascontiguousarray(P[idx]),
                         np.ascontiguousarray(CP[idx]), SC, 501,
                         dart_cost=0.0, turn_cost=1.0).solve()
        assert -m.V1[501] == pytest.approx(-base.V1[501], abs=tol)
        assert -m.V1[501] >= -base.V1[501] - 1e-9      # never better


def test_recommender_reduces_to_the_right_policy(small_recommender):
    """A point-mass posterior must reproduce that covariance's own policy, and
    a zero bias must leave the Q-values untouched."""
    rec = small_recommender
    post = ThrowPosterior(rec.Sigmas, board=rec.board, checkouts=rec.checkouts)
    post.log_post = np.full(post.log_post.shape, -1e9)
    post.log_post[2, 0] = 0.0
    for score, dart in [(501, 1), (170, 1), (40, 3)]:
        a = rec.recommend(post, score, dart, score)
        assert a == int(np.argmax(rec.models[2].q_values(score, dart, score)))
        assert rec.q_biased(2, np.zeros(2), score, dart, score) == pytest.approx(
            rec.models[2].q_values(score, dart, score))


def test_biased_recommendation_aims_off(small_recommender):
    """
    Under a known bias the recommended aim point should be displaced roughly
    opposite the bias -- the model tells the player to aim off.
    """
    rec = small_recommender
    b = np.array([10.0, 0.0])                      # pulls right
    unbiased = int(np.argmax(rec.q_biased(0, np.zeros(2), 501, 1, 501)))
    biased = int(np.argmax(rec.q_biased(0, b, 501, 1, 501)))
    dx = (rec.points[biased][1] - rec.points[unbiased][1]) * rec.mm_per_pixel
    assert dx < 0                                   # aim left of the target


def test_play_leg_runs_and_records(small_recommender):
    rec = small_recommender
    biases = bias_grid([-6.0, 0.0, 6.0], [-6.0, 0.0, 6.0])
    post = ThrowPosterior(rec.Sigmas, biases, board=rec.board,
                          checkouts=rec.checkouts)
    rng = np.random.default_rng(3)
    log = []
    true_S = np.array([[12.0 ** 2, 0.0], [0.0, 20.0 ** 2]])
    darts = play_leg(rec, post, true_S, np.array([4.0, -4.0]), rng, record=log)
    assert darts == len(log) == post.n_updates
    assert all(r["value_loss"] >= -1e-12 for r in log)
    for policy in ("oracle", "isotropic", "nobias"):
        rng2 = np.random.default_rng(4)
        p2 = ThrowPosterior(rec.Sigmas, biases, board=rec.board,
                            checkouts=rec.checkouts)
        assert play_leg(rec, p2, true_S, np.array([4.0, -4.0]), rng2,
                        policy=policy) > 0


def test_weights_from_aggregates_a_finer_grid(small_recommender):
    rec = small_recommender
    fine, _ = sigma_matrices([11.0, 13.0, 19.0, 21.0], [11.0, 13.0, 19.0, 21.0])
    post = ThrowPosterior(fine, board=rec.board, checkouts=rec.checkouts)
    w = rec.weights_from(post)
    assert w.shape == (len(rec.Sigmas), 1)
    assert w.sum() == pytest.approx(1.0)
