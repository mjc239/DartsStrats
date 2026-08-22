"""
Tests for measurement design under a Student-t throw.

``design.py`` builds the score function of a *likelihood*, not merely of a
throw, so switching distributions changes it in kind: the Gaussian score lets a
far dart dominate, because a wide player is the only way to explain it, while
the t's discounts it, because a wide dart will do. The tests below check that
the new score function is the derivative of the density it claims to be
(against an implementation that shares no code with it), and then ask the two
questions the module exists to answer -- where to aim, and for how long.
"""

import numpy as np
import pytest

from darts.dartboards import generate_dartboard
from darts.design import (best_single_target, c_criterion, candidate_targets,
                          darts_to_detect, design_information,
                          information_at_points, information_maps,
                          kernel_derivatives, sigma_gradient,
                          sigma_standard_error)
from darts.fitting import ScoreLikelihood
from darts.transitions import _correlate_fft
from darts.utils import mm_per_pixel, region_label

PX = 256
NU = 2.25


@pytest.fixture(scope="module")
def board():
    return generate_dartboard(PX)[0]


def test_t_kernel_derivatives_match_finite_differences(board):
    """
    The same cross-implementation check the Gaussian gets, for the t.

    ``ScoreLikelihood`` evaluates the t density directly on the pixel grid and
    books the off-board mass as a miss; the design code correlates score masks
    with an analytically differentiated kernel. They share no code beyond the
    board, so agreeing to 1e-9 says the ``(nu+2)/(nu+q)`` factor is genuinely
    the derivative of ``log(1 + q/nu)^{-(nu+2)/2}`` and not merely a plausible
    weight.
    """
    like = ScoreLikelihood(board=board, nu=NU)
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

    _, dK = kernel_derivatives(PX, Sigma, mm, nu=NU)
    masks = np.stack([(board == s).astype(float) for s in allowed])
    analytic = np.stack([_correlate_fft(masks, d, pad=True)[:, target[0], target[1]]
                         for d in dK])
    # score 0 also absorbs whatever left the array, so its derivative is minus
    # everything else's -- the same rule information_maps applies
    zero = int(np.flatnonzero(allowed == 0)[0])
    analytic[:, zero] = -(np.delete(analytic, zero, axis=1)).sum(axis=1)

    for i in range(5):
        assert analytic[i] == pytest.approx(numeric[i], abs=2e-9)


def test_t_probability_derivatives_sum_to_zero(board):
    """The score probabilities sum to 1, so their derivatives must sum to 0."""
    maps = information_maps(PX, 14.0, board=board, nu=NU)
    cands = candidate_targets(PX, point_stride=4)
    p = maps["probs"][:, cands[:, 0], cands[:, 1]]
    assert p.sum(axis=0) == pytest.approx(np.ones(len(cands)), abs=1e-10)


def test_nu_infinity_reproduces_the_gaussian_where_anyone_aims(board):
    """
    The t path normalises analytically and pads the transform; the Gaussian path
    normalises over the window and wraps. Those differ at the corners of the
    array, which are not targets. Over the aiming region they must agree, or the
    Student-t design numbers are not comparable with the published ones.
    """
    g = information_maps(PX, 12.0, board=board)
    i = information_maps(PX, 12.0, board=board, nu=np.inf)
    cands = candidate_targets(PX, point_stride=2)
    c = sigma_gradient(12.0 ** 2 * np.eye(2))
    vg = c_criterion(information_at_points(g, cands), c)
    vi = c_criterion(information_at_points(i, cands), c)
    assert np.abs(g["probs"] - i["probs"])[:, cands[:, 0], cands[:, 1]].max() < 1e-12
    assert (np.abs(vg - vi) / vg).max() < 1e-6


def test_the_analytic_and_discrete_normalisers_agree_for_a_gaussian():
    """
    The Gaussian path corrects for the normaliser with an expectation under K;
    the t path uses d log(2 pi sqrt(det Sigma))/dtheta. Those are the same thing,
    and a Gaussian is where both are computable -- so this is the check on the
    algebra of the new one.
    """
    mm = mm_per_pixel(PX)
    for Sigma in (12.0 ** 2 * np.eye(2), np.array([[256.0, 40.0], [40.0, 169.0]])):
        for params in ("full", "isotropic"):
            _, a = kernel_derivatives(PX, Sigma, mm, params=params)
            _, b = kernel_derivatives(PX, Sigma, mm, params=params, nu=np.inf)
            assert np.abs(a - b).max() < 1e-15


def test_a_heavier_tail_costs_information_about_the_core(board):
    """
    The t discounts exactly the darts that a Gaussian reads as evidence of
    spread, so it should learn the scale *more slowly* from the same darts.
    Whether that was true was not obvious before it was computed -- heavy tails
    can carry more information about a scale, not less -- so this records which
    way it went.
    """
    cands = candidate_targets(PX, point_stride=4)
    scale = 12.0
    c = sigma_gradient(scale ** 2 * np.eye(2))
    best = {}
    for nu in (None, 5.0, NU):
        maps = information_maps(PX, scale, board=board, nu=nu)
        idx, val, _ = best_single_target(information_at_points(maps, cands), c)
        best[nu] = (region_label(cands[idx], PX), val)
    assert best[NU][1] > best[5.0][1] > best[None][1]
    # and the practical consequence: more darts to prove the same millimetre
    n_gauss = darts_to_detect(np.sqrt(best[None][1]), 1.0)
    n_t = darts_to_detect(np.sqrt(best[NU][1]), 1.0)
    assert n_t > n_gauss


def test_the_predicted_standard_error_is_not_wildly_wrong(board):
    """
    A cheap sanity version of the simulation check: the asymptotic standard
    error must be in the right place, and must fall as 1/sqrt(n).
    """
    maps = information_maps(PX, 12.0, board=board, nu=NU)
    mm = mm_per_pixel(PX)
    t20 = np.array([[PX // 2 + int(round(103 / mm)), PX // 2]])
    M = design_information(information_at_points(maps, t20), [1.0])
    se_400 = sigma_standard_error(M, 400, 12.0 ** 2 * np.eye(2))
    se_1600 = sigma_standard_error(M, 1600, 12.0 ** 2 * np.eye(2))
    assert 0.05 < se_400 < 5.0
    assert se_1600 == pytest.approx(se_400 / 2, rel=1e-9)
