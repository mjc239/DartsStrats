"""Tests for the FFT-based transition probability builder."""

import numpy as np
import pytest

from darts.dartboards import DARTBOARD_CONSTANTS, generate_dartboard
from darts.mdp import compute_transition_probs_from_point_njit
from darts.transitions import aim_points, gaussian_kernel, transition_arrays


def test_kernel_is_centred_and_normalised():
    Sigma = 9.0 * np.eye(2)
    k = gaussian_kernel(64, Sigma)
    assert k.sum() == pytest.approx(1.0)
    assert np.unravel_index(k.argmax(), k.shape) == (32, 32)
    # Spherical covariance => symmetric about the centre pixel.
    assert k[30, 32] == pytest.approx(k[34, 32])
    assert k[32, 30] == pytest.approx(k[32, 34])


def test_fft_matches_pointwise_gaussian():
    """
    The FFT maps must agree with evaluating the Gaussian at each aim point.

    The two treat the sliver of probability mass that falls outside the board
    array differently (the pointwise version redistributes it over all scores,
    the FFT wraps it round to the zero-scoring far edge), so the board array
    has to extend a good few sigma beyond the aiming region for them to agree.
    """
    board_pixels, sigma_mm = 128, 10.0
    tr = transition_arrays(board_pixels, sigma_mm, point_stride=8)

    board, checkouts = generate_dartboard(board_pixels)
    mm_per_pixel = 2 * DARTBOARD_CONSTANTS["DARTBOARD_RADIUS_MM"] / board_pixels
    Sigma = (sigma_mm / mm_per_pixel) ** 2 * np.eye(2)
    centre = np.array([board_pixels // 2, board_pixels // 2])
    scores = tr["allowed_scores"]

    rng = np.random.default_rng(0)
    for idx in rng.choice(len(tr["points"]), size=12, replace=False):
        point = tr["points"][idx]
        p, cp = compute_transition_probs_from_point_njit(
            board, point - centre, Sigma, scores, checkouts
        )
        direct = np.array([p[np.int32(s)] for s in scores])
        direct_co = np.array([cp[np.int32(s)] for s in scores])
        assert tr["probs"][idx] == pytest.approx(direct, abs=1e-7)
        assert tr["checkout_probs"][idx] == pytest.approx(direct_co, abs=1e-7)


def test_probabilities_are_valid():
    tr = transition_arrays(64, 20.0, point_stride=3)
    probs, co = tr["probs"], tr["checkout_probs"]
    assert (probs >= 0).all()
    assert (co >= -1e-12).all()
    assert (co <= probs + 1e-12).all()
    assert probs.sum(axis=1) == pytest.approx(np.ones(probs.shape[0]))
    # A checkout must be a double or the inner bull, so odd scores above 1 and
    # scores that are not doubles can never be checkouts.
    for j, s in enumerate(tr["allowed_scores"]):
        if s % 2 == 1 or s == 0:
            assert co[:, j].max() == pytest.approx(0.0, abs=1e-12)


def test_aim_points_inside_board():
    pts = aim_points(128, margin=2.0, point_stride=1)
    r = np.linalg.norm(pts - 64, axis=1)
    radius = int(
        DARTBOARD_CONSTANTS["DOUBLE_OUTER_RADIUS"]
        / DARTBOARD_CONSTANTS["DARTBOARD_RADIUS_MM"]
        * 128
        / 2
    )
    assert r.max() < radius + 2.0
    assert len(np.unique(pts, axis=0)) == len(pts)


def test_labels_on_the_negative_x_axis():
    """
    Points exactly on the negative x axis must be labelled as being on the
    board.

    arctan2 returns exactly +pi there and the segment intervals are half-open,
    so an unwrapped angle matches no segment. generate_dartboard was fixed for
    this; region_label and aim_description had the same latent bug, which an
    earlier validation missed because it ran before the board itself was fixed,
    at which point the two agreed on the wrong answer.
    """
    from darts.utils import aim_description, region_label

    for px in (128, 256, 512):
        centre = px // 2
        mmpp = 2 * DARTBOARD_CONSTANTS["DARTBOARD_RADIUS_MM"] / px
        board, _ = generate_dartboard(px)
        for r_mm, expect in [(120.0, "11"), (166.0, "D11"), (103.0, "T11")]:
            col = centre - int(round(r_mm / mmpp))
            assert region_label([centre, col], px) == expect
            assert int(board[centre, col]) == {"11": 11, "D11": 22, "T11": 33}[expect]
        # just outside the double ring, the same axis
        col = centre - int(round(172.0 / mmpp))
        assert aim_description([centre, col], px) == "outside D11"


def test_region_label_agrees_with_the_board_everywhere():
    from darts.utils import region_label

    for px in (128, 256):
        tr = transition_arrays(px, 10.0, point_stride=1)
        board = tr["board"]
        for p in tr["points"]:
            lab = region_label(p, px)
            v = {"BULL": 50, "25": 25, "miss": 0}.get(lab)
            if v is None:
                v = (3 * int(lab[1:]) if lab[0] == "T"
                     else 2 * int(lab[1:]) if lab[0] == "D" else int(lab))
            assert v == int(board[p[0], p[1]]), (p, lab, board[p[0], p[1]])
