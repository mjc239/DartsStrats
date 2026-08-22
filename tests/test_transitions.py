"""Tests for the FFT-based transition probability builder."""

import numpy as np
import pytest

from darts.dartboards import DARTBOARD_CONSTANTS, generate_dartboard
from darts.mdp import compute_transition_probs_from_point_njit
from darts.transitions import (
    _correlate_fft,
    aim_points,
    gaussian_kernel,
    matched_scale,
    scoring_average,
    student_t_kernel,
    transition_arrays,
)


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


# --------------------------------------------------------------------------
# The Student-t kernel
# --------------------------------------------------------------------------

def test_student_t_normalises_over_the_plane():
    """
    A t is normalised analytically, so its sum over the window is 1 *minus* what
    fell outside -- and how much that is, is the whole reason for the padding
    below. Quantifying it here means the number is on the record rather than
    assumed small.
    """
    Sigma = 81.0 * np.eye(2)  # 9 pixels of scale
    off = {}
    for nu in (2.25, 3.0, 5.0, 12.0):
        k = student_t_kernel(512, Sigma, nu)
        assert (k > 0).all()
        off[nu] = 1.0 - k.sum()
    assert 5e-4 < off[2.25] < 5e-3
    # Heavier tails leave more outside, monotonically.
    assert off[2.25] > off[3.0] > off[5.0] > off[12.0]
    assert off[12.0] < 1e-8
    # A Gaussian of the same scale leaves nothing measurable out there at all.
    assert 1.0 - gaussian_kernel(512, Sigma).sum() < 1e-15


def test_student_t_becomes_the_gaussian_as_nu_grows():
    """
    The t nests the Gaussian, so at nu = infinity the two kernels must be the
    same array -- not merely similar. This is the continuity check that says the
    new path cannot have quietly changed the old model.
    """
    Sigma = np.array([[81.0, 12.0], [12.0, 49.0]])
    g = gaussian_kernel(256, Sigma)
    assert np.abs(student_t_kernel(256, Sigma, np.inf) - g).max() / g.max() < 1e-14
    # and approaches it from a finite nu, at the rate 1/nu
    err = [np.abs(student_t_kernel(256, Sigma, nu) - g).max() / g.max()
           for nu in (1e3, 1e5, 1e7)]
    assert err[0] > err[1] > err[2]
    assert err[2] < 1e-6


def test_padded_correlation_is_linear():
    """The padded FFT must equal the correlation written out by hand."""
    n = 32
    rng = np.random.default_rng(0)
    mask = (rng.random((n, n)) < 0.4).astype(np.float64)
    kernel = student_t_kernel(n, 4.0 * np.eye(2), 2.25)

    c = n // 2
    brute = np.zeros((n, n))
    for p0 in range(n):
        for p1 in range(n):
            for x0 in range(n):
                for x1 in range(n):
                    o0, o1 = p0 - x0 + c, p1 - x1 + c
                    if 0 <= o0 < n and 0 <= o1 < n:
                        brute[p0, p1] += mask[x0, x1] * kernel[o0, o1]

    padded = _correlate_fft(mask[None], kernel, pad=True)[0]
    assert padded == pytest.approx(brute, abs=1e-14)

    # The circular version differs precisely where the wrap-around bites: at the
    # edges, where a dart leaving one side is scored as arriving at the other.
    circular = _correlate_fft(mask[None], kernel, pad=False)[0]
    assert np.abs(circular - padded)[:4, :].max() > 1e-3


def test_padding_leaves_the_gaussian_alone_where_anyone_aims():
    """
    Padding changes the answer for aim points near the corners of the array,
    which are off the board. Over the aiming region a Gaussian's wrap-around is
    ~1e-12, which is why nobody has ever had to think about it.
    """
    board, _ = generate_dartboard(512)
    mm_per_pixel = 2 * DARTBOARD_CONSTANTS["DARTBOARD_RADIUS_MM"] / 512
    Sigma = (8.0 / mm_per_pixel) ** 2 * np.eye(2)
    kernel = gaussian_kernel(512, Sigma)
    masks = np.stack([(board == s).astype(np.float64) for s in np.unique(board)])

    circular = _correlate_fft(masks, kernel, pad=False)
    padded = _correlate_fft(masks, kernel, pad=True)
    pts = aim_points(512, margin=2.0, point_stride=4)
    at_points = np.abs(circular - padded)[:, pts[:, 0], pts[:, 1]]
    assert at_points.max() < 1e-10


def test_nu_none_and_nu_infinity_give_the_same_transitions():
    """
    ``nu=None`` takes the original code path and ``nu=inf`` takes the new one,
    through a different kernel, a padded transform and an explicit off-board
    term. They must still land on the same matrix, or every Gaussian result in
    the project is not what the new path would reproduce.
    """
    a = transition_arrays(128, 10.0, point_stride=6)
    b = transition_arrays(128, 10.0, point_stride=6, nu=np.inf)
    assert np.abs(a["probs"] - b["probs"]).max() < 1e-12
    assert np.abs(a["checkout_probs"] - b["checkout_probs"]).max() < 1e-12


def test_student_t_transitions_are_a_valid_distribution():
    """
    Off-board mass is booked as a miss rather than renormalised away, so the
    rows still sum to one -- but with a real probability of scoring nothing,
    which is the point.
    """
    tr = transition_arrays(256, 6.5, point_stride=6, nu=2.25)
    probs, co = tr["probs"], tr["checkout_probs"]
    assert (probs >= 0).all()
    assert (co <= probs + 1e-12).all()
    assert probs.sum(axis=1) == pytest.approx(np.ones(probs.shape[0]), abs=1e-10)

    zero = list(tr["allowed_scores"]).index(0)
    scores = tr["allowed_scores"]
    best = (probs * scores).sum(axis=1).argmax()
    gauss = transition_arrays(256, 6.5, point_stride=6)
    # Aiming at the best target, a t misses the board sometimes and a Gaussian
    # essentially never.
    assert probs[best, zero] > 1e-3
    assert gauss["probs"][gauss["probs"].dot(scores).argmax(), zero] < 1e-6


def test_matched_scale_equalises_the_three_dart_average():
    """
    Matching is on the three-dart average, because that is what the ability
    bands are named for. The scale that results is smaller than the Gaussian
    sigma -- a t of equal scale is a worse player -- and lands close to what
    notebook 21 fitted to real professionals, which the per-axis-SD convention
    does not.
    """
    board, _ = generate_dartboard(256)
    target = scoring_average(8.0, board=board)
    for nu in (2.25, 5.0):
        s = matched_scale(8.0, nu, board=board)
        assert scoring_average(s, nu=nu, board=board) == pytest.approx(target, abs=0.01)
        assert s < 8.0
    # Heavier tails need a tighter core to score the same.
    assert matched_scale(8.0, 2.25, board=board) < matched_scale(8.0, 5.0, board=board)
    # A pro's matched core at the nu real darts prefer is a few mm, not the
    # ~2mm that matching per-axis SD through the nu/(nu-2) inflation would give.
    assert 5.0 < matched_scale(8.0, 2.25, board=board) < 7.5
    assert matched_scale(8.0, None) == 8.0


def test_matched_scale_is_monotone_in_ability():
    board, _ = generate_dartboard(256)
    scales = [matched_scale(s, 2.25, board=board) for s in (6.5, 10.0, 16.0, 28.0)]
    assert scales == sorted(scales)


def test_student_t_kernel_rejects_a_non_positive_nu():
    with pytest.raises(ValueError):
        student_t_kernel(32, np.eye(2), 0.0)


def test_a_matched_t_is_better_at_beds_and_worse_at_sectors():
    """
    Notebook 22's mechanism, restated. Matched on the three-dart average, a t has
    a tighter core and a few darts that go anywhere, so it must beat the Gaussian
    on an 8mm bed and lose to it on a whole sector. Every conclusion in that
    notebook about which bands the t helps is downstream of those two signs.

    The treble is the clean case and is strictly better at every ability. The
    double is not, and the notebook says so: at the middle abilities the t's
    advantage there is 1.002, which is a tie the grid cannot resolve. So the
    double is asserted as "not worse", and strictly better only at the two ends
    where the effect is real. (256 pixels here, against the notebook's 512, so
    the numbers differ in the third place -- the signs are the claim.)
    """
    board, _ = generate_dartboard(256)
    d20 = {}
    for sigma in (8.0, 16.0, 28.0):
        g = transition_arrays(256, sigma, point_stride=4)
        t = transition_arrays(256, matched_scale(sigma, 2.25, board=board),
                              point_stride=4, nu=2.25)
        sc = g['allowed_scores']
        i40, i60 = list(sc).index(40), list(sc).index(60)
        d20[sigma] = (g['checkout_probs'][:, i40].max(),
                      t['checkout_probs'][:, i40].max())
        assert d20[sigma][1] > 0.99 * d20[sigma][0]
        assert t['probs'][:, i60].max() > g['probs'][:, i60].max()
        # and the other way on a target a whole sector satisfies
        sector = lambda tr: tr['probs'][:, sc >= 20].sum(1).max()
        assert sector(t) < sector(g)
    for sigma in (8.0, 28.0):
        assert d20[sigma][1] > d20[sigma][0]


def test_a_matched_t_misses_the_board_at_about_the_observed_rate():
    """
    Nothing in the matching uses the off-board rate -- the scale is chosen to
    reproduce a three-dart average -- so what the t predicts there is a free
    check. Cleaned professional darts miss on 0.32% of pure-scoring throws
    (0.012% on the first dart of a visit, 0.60% on the third). A Gaussian says
    3e-17, which is not a near miss.
    """
    board, _ = generate_dartboard(256)
    g = transition_arrays(256, 8.0, point_stride=4)
    t = transition_arrays(256, matched_scale(8.0, 2.25, board=board),
                          point_stride=4, nu=2.25)
    sc = g['allowed_scores']
    zero = list(sc).index(0)
    at_best = lambda tr: tr['probs'][(tr['probs'] @ sc).argmax(), zero]
    assert at_best(g) < 1e-10
    assert 1e-3 < at_best(t) < 2e-2
