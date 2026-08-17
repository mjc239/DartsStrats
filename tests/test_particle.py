"""Tests for the particle filter used for online play."""

import numpy as np
import pytest

from darts.dartboards import generate_dartboard
from darts.throw_shape import (ParticleThrowPosterior, ShapeRecommender,
                               ThrowPosterior, bias_grid, bias_prior, play_leg,
                               shape_prior, sigma_matrices)
from darts.utils import mm_per_pixel

TRUE_S = np.array([[12.0 ** 2, 0.0], [0.0, 20.0 ** 2]])
TRUE_B = np.array([4.0, -9.0])


@pytest.fixture(scope="module")
def board_and_checkouts():
    return generate_dartboard(256)


@pytest.fixture(scope="module")
def darts(board_and_checkouts):
    """One fixed sequence of (target, score), so every filter sees the same data."""
    board, _ = board_and_checkouts
    px = board.shape[0]
    mmpp = mm_per_pixel(px)
    L = np.linalg.cholesky(TRUE_S)
    rng = np.random.default_rng(0)
    targets = [np.array([0.0, 103.0]), np.array([-80.0, -40.0]),
               np.array([0.0, 0.0]), np.array([100.0, 60.0])]
    out = []
    for i in range(600):
        t = targets[i % len(targets)]
        land = t + TRUE_B + L @ rng.standard_normal(2)
        col = int(round(land[0] / mmpp)) + px // 2
        row = int(round(land[1] / mmpp)) + px // 2
        v = int(board[row, col]) if 0 <= row < px and 0 <= col < px else 0
        out.append((t, v))
    return out


@pytest.fixture(scope="module")
def small_recommender():
    Sigmas, _ = sigma_matrices([12.0, 20.0], [12.0, 20.0])
    return ShapeRecommender(Sigmas, board_pixels=128, point_stride=2)


def _run(post, darts):
    for t, v in darts:
        post.update(t, v, pixel=False)
    return post


def test_likelihood_matches_the_grid_for_the_same_parameters(board_and_checkouts):
    """
    The filter and the grid must agree on what a dart is worth. Give each a
    single identical (Sigma, bias) and demand the same log likelihood.
    """
    board, co = board_and_checkouts
    theta = np.array([[np.log(12.0), np.log(20.0), 0.0, 4.0, -9.0]])
    pf = ParticleThrowPosterior(particles=theta, board=board, checkouts=co)
    grid = ThrowPosterior(TRUE_S[None], TRUE_B[None], board=board, checkouts=co)
    aim = np.array([0.0, 103.0])
    for score, checkout in [(60, False), (20, False), (40, True)]:
        assert pf._log_likelihood(aim, score, checkout)[0] == pytest.approx(
            grid._log_likelihood(aim, score, checkout)[0, 0], abs=1e-10)


def test_filter_recovers_the_truth(board_and_checkouts, darts):
    board, co = board_and_checkouts
    pf = _run(ParticleThrowPosterior(600, band="league", board=board,
                                     checkouts=co, rng=np.random.default_rng(1)),
              darts)
    sx, sy = pf.mean_sigma_xy()
    assert sx == pytest.approx(12.0, abs=3.0)
    assert sy == pytest.approx(20.0, abs=3.0)
    assert pf.mean_bias() == pytest.approx(TRUE_B, abs=4.0)
    assert pf.mean_ratio() > 1.2


def test_filter_agrees_with_the_exact_grid(board_and_checkouts, darts):
    """
    The filter is an approximation to the grid posterior, so on identical data
    with a matched prior the two should reach the same place.
    """
    board, co = board_and_checkouts
    S, _ = sigma_matrices(np.arange(8.0, 24.1, 2.0), np.arange(8.0, 24.1, 2.0))
    B = bias_grid(np.arange(-12.0, 12.1, 3.0), np.arange(-15.0, 15.1, 3.0))
    prior = np.outer(shape_prior(S, "league"), bias_prior(B, sd_mm=8.0))
    grid = _run(ThrowPosterior(S, B, prior=prior, board=board, checkouts=co), darts)
    pf = _run(ParticleThrowPosterior(600, band="league", board=board,
                                     checkouts=co, rng=np.random.default_rng(1)),
              darts)
    gx, gy = grid.mean_sigma_xy()
    px_, py_ = pf.mean_sigma_xy()
    assert px_ == pytest.approx(gx, abs=2.5)
    assert py_ == pytest.approx(gy, abs=2.5)
    assert pf.mean_bias() == pytest.approx(grid.mean_bias(), abs=3.5)


def test_rejuvenation_prevents_degeneracy(board_and_checkouts, darts):
    """
    The parameters are static, so plain resampling collapses the particles onto
    a handful of duplicated values and the belief reports false certainty. The
    Liu-West shrink-and-jitter is what stops that, and turning it off (delta=1,
    which makes the shrinkage a=1 and the jitter zero) should show the failure.
    """
    board, co = board_and_checkouts
    kw = dict(board=board, checkouts=co)
    alive = _run(ParticleThrowPosterior(400, band="league", delta=0.98,
                                        rng=np.random.default_rng(2), **kw), darts)
    dead = _run(ParticleThrowPosterior(400, band="league", delta=1.0,
                                       rng=np.random.default_rng(2), **kw), darts)
    # Count distinct particles coarsely: delta=1 leaves only the 1e-12 ridge
    # used to keep the Cholesky well posed, so the duplicates differ by about
    # 1e-6 and are not distinct in any way that matters.
    n_alive = len(np.unique(alive._theta.round(3), axis=0))
    n_dead = len(np.unique(dead._theta.round(3), axis=0))
    assert n_alive > 0.9 * len(alive._theta)
    assert n_dead < 0.1 * n_alive

    # The symptom that matters is not the bookkeeping but the false certainty:
    # a collapsed filter reports a far tighter belief than the data support.
    assert dead.sd_ratio() < 0.2 * alive.sd_ratio()
    assert dead.sd_bias()[1] < 0.2 * alive.sd_bias()[1]


def test_resampling_keeps_the_ess_up(board_and_checkouts, darts):
    board, co = board_and_checkouts
    pf = _run(ParticleThrowPosterior(400, band="league", ess_fraction=0.5,
                                     board=board, checkouts=co,
                                     rng=np.random.default_rng(3)), darts)
    assert pf.n_resamples > 0
    assert pf.ess() > 0.2 * 400


def test_drift_keeps_the_belief_open(board_and_checkouts, darts):
    """
    With drift the filter is a tracker rather than an estimator: it must not
    keep sharpening indefinitely on a fixed player.
    """
    board, co = board_and_checkouts
    kw = dict(band="league", board=board, checkouts=co)
    still = _run(ParticleThrowPosterior(400, rng=np.random.default_rng(4), **kw),
                 darts)
    moving = _run(ParticleThrowPosterior(400, drift={"sigma": 0.004, "bias": 0.3},
                                         rng=np.random.default_rng(4), **kw), darts)
    assert moving.sd_ratio() > still.sd_ratio()


def test_drift_is_named_because_the_units_differ(board_and_checkouts):
    """
    The parameters live in a transformed space where log-sigma and millimetre
    components are not comparable, so a scalar drift means different things to
    each. The dict form must put each number where it belongs.
    """
    board, co = board_and_checkouts
    kw = dict(board=board, checkouts=co, rng=np.random.default_rng(0))
    named = ParticleThrowPosterior(20, drift={"sigma": 0.002, "bias": 0.25}, **kw)
    assert named.drift == pytest.approx([0.002, 0.002, 0.0, 0.25, 0.25])
    assert ParticleThrowPosterior(20, drift=0.01, **kw).drift == pytest.approx(
        [0.01] * 5)


def test_drift_actually_tracks_a_changing_player(board_and_checkouts):
    """
    The point of drift, and the thing a scalar gets wrong: on a player whose
    pull grows through a session, the tracker must follow it and the plain
    estimator must lag.
    """
    board, co = board_and_checkouts
    px = board.shape[0]
    mmpp = mm_per_pixel(px)
    L = np.linalg.cholesky(TRUE_S)
    rng = np.random.default_rng(7)
    targets = [np.array([0.0, 103.0]), np.array([-80.0, -40.0]),
               np.array([0.0, 0.0]), np.array([100.0, 60.0])]
    seq, truth = [], []
    n = 800
    for i in range(n):
        b = np.array([0.0, -14.0 * i / n])          # pull grows to 14 mm
        t = targets[i % len(targets)]
        land = t + b + L @ rng.standard_normal(2)
        col = int(round(land[0] / mmpp)) + px // 2
        row = int(round(land[1] / mmpp)) + px // 2
        seq.append((t, int(board[row, col]) if 0 <= row < px and 0 <= col < px else 0))
        truth.append(b[1])
    truth = np.array(truth)

    err = {}
    for label, drift in [("estimator", 0.0),
                         ("tracker", {"sigma": 0.002, "bias": 0.25})]:
        pf = ParticleThrowPosterior(400, band="league", drift=drift, board=board,
                                    checkouts=co, rng=np.random.default_rng(5))
        trace = []
        for t, v in seq:
            pf.update(t, v, pixel=False)
            trace.append(pf.mean_bias()[1])
        err[label] = np.abs(np.array(trace[-200:]) - truth[-200:]).mean()
    assert err["tracker"] < 0.5 * err["estimator"]


def test_tilt_is_off_unless_asked_for(board_and_checkouts):
    board, co = board_and_checkouts
    flat = ParticleThrowPosterior(100, board=board, checkouts=co,
                                  rng=np.random.default_rng(5))
    assert np.all(flat.rho == 0.0)
    tilted = ParticleThrowPosterior(100, tilt=True, board=board, checkouts=co,
                                    rng=np.random.default_rng(5))
    assert np.abs(tilted.rho).max() > 0.0
    assert np.abs(tilted.rho).max() < 1.0
    assert (np.linalg.det(tilted.Sigmas) > 0).all()


def test_support_groups_are_a_valid_smaller_belief(small_recommender,
                                                   board_and_checkouts, darts):
    """
    The groups must be a probability distribution over solved covariances, and
    there must be far fewer of them than particles -- that collapse is what
    makes the recommendation cheap, not just the belief update.
    """
    rec = small_recommender
    pf = _run(ParticleThrowPosterior(400, band="league", board=rec.board,
                                     checkouts=rec.checkouts,
                                     rng=np.random.default_rng(6)), darts)
    groups = rec.support(pf)
    assert sum(w for _, _, w in groups) == pytest.approx(1.0)
    assert all(0 <= k < len(rec.Sigmas) for k, _, _ in groups)
    assert len(groups) < 0.5 * 400


def test_particle_and_grid_recommend_alike(board_and_checkouts, darts):
    """Once both have seen the same data they should advise the same shots."""
    Sigmas, _ = sigma_matrices([8.0, 12.0, 16.0, 20.0], [8.0, 12.0, 16.0, 20.0])
    rec = ShapeRecommender(Sigmas, board_pixels=128, point_stride=2)
    S, _ = sigma_matrices(np.arange(8.0, 21.1, 2.0), np.arange(8.0, 21.1, 2.0))
    B = bias_grid(np.arange(-12.0, 12.1, 3.0), np.arange(-12.0, 12.1, 3.0))
    prior = np.outer(shape_prior(S, "league"), bias_prior(B, sd_mm=8.0))
    grid = _run(ThrowPosterior(S, B, prior=prior, board=rec.board,
                               checkouts=rec.checkouts), darts)
    pf = _run(ParticleThrowPosterior(600, band="league", board=rec.board,
                                     checkouts=rec.checkouts,
                                     rng=np.random.default_rng(7)), darts)
    agree = 0
    states = [(501, 1), (301, 1), (170, 1), (81, 2), (40, 3), (32, 3)]
    for score, dart in states:
        a_g = rec.recommend(grid, score, dart, score)
        a_p = rec.recommend(pf, score, dart, score)
        d = np.linalg.norm(rec.points[a_g] - rec.points[a_p]) * rec.mm_per_pixel
        agree += d < 12.0        # within a bed's width of each other
    assert agree >= len(states) - 1


def test_filter_is_a_drop_in_for_play_leg(small_recommender):
    rec = small_recommender
    pf = ParticleThrowPosterior(200, band="league", board=rec.board,
                                checkouts=rec.checkouts,
                                rng=np.random.default_rng(8))
    log = []
    darts_thrown = play_leg(rec, pf, TRUE_S, TRUE_B, np.random.default_rng(9),
                            record=log)
    assert darts_thrown == len(log) == pf.n_updates
    assert all(r["value_loss"] >= -1e-12 for r in log)
    assert np.isfinite(log[-1]["sigma_x"]) and np.isfinite(log[-1]["bias_y"])


def test_filter_is_faster_than_the_grid(board_and_checkouts, darts):
    """
    The whole point. Both halves of the per-dart cost must come down: the
    belief update, and the Q-average the recommender does over the support.
    """
    import time
    board, co = board_and_checkouts
    S, _ = sigma_matrices(np.arange(8.0, 24.1, 2.0), np.arange(8.0, 24.1, 2.0))
    B = bias_grid(np.arange(-12.0, 12.1, 3.0), np.arange(-15.0, 15.1, 3.0))
    grid = ThrowPosterior(S, B, board=board, checkouts=co)
    pf = ParticleThrowPosterior(300, band="league", board=board, checkouts=co,
                                rng=np.random.default_rng(10))
    sub = darts[:40]
    t0 = time.perf_counter()
    for t, v in sub:
        grid.update(t, v, pixel=False)
    t_grid = time.perf_counter() - t0
    t0 = time.perf_counter()
    for t, v in sub:
        pf.update(t, v, pixel=False)
    t_pf = time.perf_counter() - t0
    assert t_pf < t_grid / 5
