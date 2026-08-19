"""Calibrating the throw model against scores that carry no aim point."""
import numpy as np
import pytest

from darts.calibration import (MAX_VISIT, SCORING_FLOOR, T20_MM, DoubleAttempts,
                               ScoringVisits, aggregate_consistency,
                               fit_from_aggregates, fit_from_visits)
from darts.dartboards import generate_dartboard
from darts.throw_shape import rotated_sigma
from darts.utils import mm_per_pixel

PIXELS = 512


@pytest.fixture(scope="module")
def board():
    return generate_dartboard(PIXELS)


@pytest.fixture(scope="module")
def visits(board):
    return ScoringVisits(board_pixels=PIXELS, board=board[0])


@pytest.fixture(scope="module")
def doubles(board):
    return DoubleAttempts(board_pixels=PIXELS, board=board[0], checkouts=board[1])


def test_dart_pmf_is_a_distribution(visits):
    pmf = visits.dart_pmf(T20_MM, 9.0 ** 2 * np.eye(2))
    assert pmf.shape == (61,)
    assert pmf.min() >= 0
    assert pmf.sum() == pytest.approx(1.0, abs=1e-12)


def test_visit_pmf_is_a_distribution(visits):
    pmf = visits.visit_pmf(T20_MM, 9.0 ** 2 * np.eye(2))
    assert pmf.shape == (MAX_VISIT + 1,)
    assert pmf.min() >= 0
    assert pmf.sum() == pytest.approx(1.0, abs=1e-12)


def test_visit_pmf_matches_brute_force(visits, board):
    """The convolution is only valid because the three darts are i.i.d."""
    sigma = 9.0
    pmf = visits.visit_pmf(T20_MM, sigma ** 2 * np.eye(2))

    arr, mm, centre = board[0], mm_per_pixel(PIXELS), PIXELS // 2
    rng = np.random.default_rng(0)
    n = 200_000
    land = T20_MM + sigma * rng.standard_normal((3 * n, 2))
    col = np.rint(land[:, 0] / mm).astype(int) + centre
    row = np.rint(land[:, 1] / mm).astype(int) + centre
    ok = (row >= 0) & (row < PIXELS) & (col >= 0) & (col < PIXELS)
    v = np.zeros(3 * n, dtype=np.int64)
    v[ok] = arr[row[ok], col[ok]]
    empirical = np.bincount(v.reshape(n, 3).sum(axis=1), minlength=MAX_VISIT + 1) / n

    # 4 Monte Carlo standard errors on the worst cell
    assert np.abs(pmf - empirical).max() < 4 * np.sqrt(0.25 / n)
    assert ((pmf * np.arange(MAX_VISIT + 1)).sum()
            == pytest.approx(v.reshape(n, 3).sum(axis=1).mean(), rel=2e-3))


def test_a_tighter_throw_scores_more(visits):
    """Monotonicity: the whole calibration is meaningless without it."""
    avg = [visits.statistics(T20_MM, s ** 2 * np.eye(2))["three_dart_average"]
           for s in (6.0, 8.0, 12.0, 18.0)]
    assert avg == sorted(avg, reverse=True)


def test_usable_rejects_visits_that_left_the_scoring_phase(visits):
    before = np.array([501, 400, 300, 260])
    scored = np.array([60, 100, 60, 60])          # ends at 441, 300, 240, 200
    keep = visits.usable(before, scored)
    assert list(keep) == [True, True, False, False]


def test_usable_rejects_checkout_visits(visits):
    before = np.array([501, 501])
    scored = np.array([60, 60])
    keep = visits.usable(before, scored, darts_used=np.array([3, 2]))
    assert list(keep) == [True, False]


def test_truncated_likelihood_normalises(visits):
    """Selecting on the outcome must be matched by conditioning on it."""
    Sigma = 9.0 ** 2 * np.eye(2)
    pmf = visits.visit_pmf(T20_MM, Sigma)
    before = 320                                   # cut at 320 - 250 = 70
    cut = before - SCORING_FLOOR
    total = sum(np.exp(visits.log_likelihood(T20_MM, Sigma, [before], [t]))
                for t in range(cut + 1))
    assert total == pytest.approx(1.0, rel=1e-9)
    # and it really is a reweighting of the untruncated pmf
    lone = np.exp(visits.log_likelihood(T20_MM, Sigma, [before], [60]))
    assert lone == pytest.approx(pmf[60] / pmf[:cut + 1].sum(), rel=1e-9)


def test_untruncated_when_the_score_is_high(visits):
    Sigma = 9.0 ** 2 * np.eye(2)
    pmf = visits.visit_pmf(T20_MM, Sigma)
    lone = np.exp(visits.log_likelihood(T20_MM, Sigma, [501], [100]))
    assert lone == pytest.approx(pmf[100], rel=1e-12)


def test_likelihood_refuses_unfiltered_data(visits):
    with pytest.raises(ValueError, match="filter with usable"):
        visits.log_likelihood(T20_MM, 9.0 ** 2 * np.eye(2), [300], [140])


def test_fit_recovers_a_known_sigma(board):
    """The whole point: a scoresheet with no aim point still measures a player."""
    arr, mm, centre = board[0], mm_per_pixel(PIXELS), PIXELS // 2
    sigma = 9.0
    rng = np.random.default_rng(3)
    before, scored = [], []
    for _ in range(300):
        score = 501
        while score > SCORING_FLOOR:
            land = T20_MM + sigma * rng.standard_normal((3, 2))
            col = np.rint(land[:, 0] / mm).astype(int) + centre
            row = np.rint(land[:, 1] / mm).astype(int) + centre
            ok = (row >= 0) & (row < PIXELS) & (col >= 0) & (col < PIXELS)
            v = np.where(ok, arr[np.clip(row, 0, PIXELS - 1),
                                 np.clip(col, 0, PIXELS - 1)], 0).sum()
            before.append(score)
            scored.append(int(v))
            score -= int(v)

    fit = fit_from_visits(before, scored, board_pixels=PIXELS, board=arr,
                          sigma_init=12.0, fix_mu=True)
    # a sigma=9mm player takes ~2.2 scoring visits to fall from 501 to the floor
    assert fit["n_darts"] > 1800
    assert fit["sigma_mm"] == pytest.approx(sigma, rel=0.05)


def test_aggregates_invert_the_statistics(board):
    """fit_from_aggregates is the inverse of statistics, so a round trip closes."""
    sv = ScoringVisits(board_pixels=PIXELS, board=board[0])
    truth = 7.5
    stats = sv.statistics(T20_MM, truth ** 2 * np.eye(2))
    fit = fit_from_aggregates(three_dart_average=stats["three_dart_average"],
                              p_180=stats["p_180"], board_pixels=PIXELS,
                              board=board[0])
    assert fit["sigma_mm"] == pytest.approx(truth, rel=1e-3)


def test_consistent_statistics_imply_one_sigma(board):
    """A player the model can actually produce must not look inconsistent."""
    sv = ScoringVisits(board_pixels=PIXELS, board=board[0])
    truth = 8.0
    stats = sv.statistics(T20_MM, truth ** 2 * np.eye(2))
    implied = aggregate_consistency(
        {k: stats[k] for k in ("three_dart_average", "p_180", "p_140_plus")},
        board_pixels=PIXELS, board=board[0])
    for name, sigma in implied.items():
        assert sigma == pytest.approx(truth, rel=1e-2), name


def test_isotropic_makes_every_double_equally_hard(doubles):
    """The claim the real checkout data would test."""
    rates = [doubles.hit_probability(8.0 ** 2 * np.eye(2), n)
             for n in (20, 6, 3, 11, 16, 8, 19, 5)]
    assert max(rates) - min(rates) < 0.005


def test_a_tall_throw_prefers_the_side_doubles(doubles):
    """And this is why the flat prediction is a real test rather than a tautology."""
    tall = rotated_sigma(11.0, 6.0, 90.0)
    top = doubles.hit_probability(tall, 20)        # at 90 degrees
    side = doubles.hit_probability(tall, 6)        # at 0 degrees
    assert side > top * 1.4


def test_double_fit_recovers_sigma_from_binomial_counts(doubles):
    truth = 9.0
    p = doubles.hit_probability(truth ** 2 * np.eye(2), 20)
    n = 200_000
    fit = doubles.fit_sigma(attempts=n, hits=int(round(p * n)), number=20)
    assert fit["sigma_mm"] == pytest.approx(truth, rel=0.02)
    assert fit["predicted_rate"] == pytest.approx(p, rel=1e-3)
