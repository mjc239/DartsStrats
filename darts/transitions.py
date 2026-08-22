"""
Fast construction of darts transition probabilities.

For a player with throwing distribution ``N(p, Sigma)`` aiming at point ``p``,
the probability of scoring ``v`` is

    P(score = v | aim at p) = sum_x f_{p,Sigma}(x) * 1[board(x) == v]

which is a cross-correlation of the Gaussian pdf with the indicator mask of
score ``v``. Evaluating it point by point costs ``O(n_points * n_pixels)``;
evaluating it for *every* aiming point at once with an FFT costs
``O(n_scores * n_pixels * log n_pixels)``, which is thousands of times cheaper
and lets much finer boards be used.

This is the same trick :func:`darts.stats.expected_score` already uses for the
expected score map, applied to each score's indicator mask (and to the checkout
regions) rather than to the board itself.

Nothing downstream of here knows what distribution the dart came from. The
solvers, the checkout charts and the match model all take the ``(n_points,
n_scores)`` matrix this module produces and never see ``Sigma`` again, so a
different throw distribution is a different kernel and nothing else. Notebook 21
found on real competition darts that a Student-t beats a Gaussian by a wide
margin, so ``nu`` selects that kernel instead; ``nu=None`` is the Gaussian and
is the default everywhere.
"""

import numpy as np
from scipy.optimize import brentq

from darts.dartboards import generate_dartboard, DARTBOARD_CONSTANTS
from darts.players import BOARD_PIXELS


def gaussian_kernel(pixels, Sigma):
    """
    Gaussian pdf sampled on an integer pixel grid, centred exactly on pixel
    ``pixels // 2`` and normalised to sum to 1.

    ``Sigma`` follows the convention used by :func:`darts.stats.gaussian_filter`:
    it is expressed in (x, y) = (column, row) coordinates.
    """
    Sigma = np.asarray(Sigma, dtype=np.float64)
    c = pixels // 2
    offs = np.arange(pixels) - c
    x = offs[None, :]  # column offset
    y = offs[:, None]  # row offset
    det = Sigma[0, 0] * Sigma[1, 1] - Sigma[0, 1] ** 2
    quad = (
        Sigma[1, 1] * x * x - 2 * Sigma[0, 1] * x * y + Sigma[0, 0] * y * y
    ) / det
    kernel = np.exp(-quad / 2.0)
    return kernel / kernel.sum()


def _quadratic_form(pixels, Sigma):
    """``x' Sigma^-1 x`` at every pixel offset from the centre, and ``det Sigma``."""
    Sigma = np.asarray(Sigma, dtype=np.float64)
    c = pixels // 2
    offs = np.arange(pixels) - c
    x = offs[None, :]  # column offset
    y = offs[:, None]  # row offset
    det = Sigma[0, 0] * Sigma[1, 1] - Sigma[0, 1] ** 2
    quad = (
        Sigma[1, 1] * x * x - 2 * Sigma[0, 1] * x * y + Sigma[0, 0] * y * y
    ) / det
    return quad, det


def student_t_kernel(pixels, Sigma, nu):
    """
    Bivariate Student-t pdf sampled on an integer pixel grid, centred exactly on
    pixel ``pixels // 2``.

    ``Sigma`` is the **scale** matrix, not the covariance: a t has covariance
    ``nu / (nu - 2) * Sigma``, which at the ``nu = 2.25`` real darts prefer is
    nine times larger. It follows the same (x, y) = (column, row) convention as
    :func:`gaussian_kernel`, and ``nu = np.inf`` recovers that function exactly.

    Unlike the Gaussian this is normalised **analytically**, over the plane,
    rather than by dividing through by its own sum over the window. That is the
    whole difficulty of the distribution in one line. A Gaussian with a realistic
    sigma has ~1e-307 of its mass outside a 512-pixel board, so summing the
    window and summing the plane are the same number; a t at ``nu = 2.25`` leaves
    between 7e-4 and 4e-3 out there. Dividing by the window sum would quietly
    push that mass back onto the board and inflate every score. Left alone, it
    stays where it belongs, and :func:`transition_maps` books it as a miss.
    """
    if nu <= 0:
        raise ValueError("nu must be positive")
    quad, det = _quadratic_form(pixels, Sigma)
    if np.isinf(nu):
        log_profile = -quad / 2.0
    else:
        # log1p keeps the large-nu limit accurate, where quad / nu underflows the
        # difference between 1 + q/nu and 1.
        log_profile = -0.5 * (nu + 2.0) * np.log1p(quad / nu)
    return np.exp(log_profile) / (2.0 * np.pi * np.sqrt(det))


def _correlate_fft(mask_stack, kernel, pad=False):
    """Cross-correlate each mask in ``mask_stack`` with a centred ``kernel``.

    The FFT is circular, so by default mass leaving one edge of the board
    re-enters at the opposite one. For a Gaussian that is a fiction about 1e-307
    of the throw and nobody has ever cared. For a polynomial tail it is not: a
    dart wide enough to fall off the left of the array would be scored as though
    it had hit the right of the board. ``pad=True`` zero-pads to twice the
    board before transforming, which makes the correlation linear -- mass that
    leaves the array is then genuinely gone, and the caller can account for it.
    """
    n = mask_stack.shape[-1]
    if not pad:
        kernel_ft = np.fft.fft2(np.fft.ifftshift(kernel))
        out = np.empty_like(mask_stack, dtype=np.float64)
        for k in range(mask_stack.shape[0]):
            out[k] = np.real(np.fft.ifft2(np.fft.fft2(mask_stack[k]) * kernel_ft))
        return out

    # Offsets p - x run over (-n, n), so a period of 2n is enough for every one
    # of them to land on its own index. ifftshift would fold the negative
    # offsets into the middle of the padded array rather than its end, so roll
    # the kernel into place after padding instead of before.
    m = 2 * n
    c = n // 2
    padded = np.zeros((m, m), dtype=np.float64)
    padded[:n, :n] = kernel
    kernel_ft = np.fft.fft2(np.roll(padded, (-c, -c), axis=(0, 1)))
    out = np.empty_like(mask_stack, dtype=np.float64)
    for k in range(mask_stack.shape[0]):
        full = np.fft.ifft2(np.fft.fft2(mask_stack[k], s=(m, m)) * kernel_ft)
        out[k] = np.real(full[:n, :n])
    return out


def transition_maps(board, checkouts, Sigma, nu=None):
    """
    Score and checkout probability maps for every pixel of the board as an
    aiming point.

    Args:
        board (np.ndarray): (n, n) array of the score at each pixel.
        checkouts (np.ndarray): (n, n) boolean mask of checkout regions.
        Sigma (np.ndarray): 2x2 matrix in pixel units -- the covariance of the
            Gaussian, or the scale of the Student-t.
        nu (float): degrees of freedom of a Student-t throw. ``None`` (the
            default) means a Gaussian, and takes exactly the path it always did.

    Returns:
        tuple:
            - prob_maps (np.ndarray): (n_scores, n, n), probability of each
              board score when aiming at each pixel.
            - checkout_maps (np.ndarray): (n_scores, n, n), probability of
              hitting each score in a checkout region.
            - allowed_scores (np.ndarray): (n_scores,) sorted board scores.
    """
    allowed_scores = np.unique(board).astype(np.int32)
    pad = nu is not None
    if pad:
        kernel = student_t_kernel(board.shape[0], Sigma, nu)
    else:
        kernel = gaussian_kernel(board.shape[0], Sigma)

    masks = np.stack([(board == s).astype(np.float64) for s in allowed_scores])
    prob_maps = _correlate_fft(masks, kernel, pad=pad)

    co_masks = masks * checkouts[None, :, :]
    checkout_maps = _correlate_fft(co_masks, kernel, pad=pad)

    # Numerical noise from the FFT can produce tiny negative probabilities.
    np.clip(prob_maps, 0.0, None, out=prob_maps)
    np.clip(checkout_maps, 0.0, None, out=checkout_maps)
    np.minimum(checkout_maps, prob_maps, out=checkout_maps)

    if not pad:
        prob_maps /= prob_maps.sum(axis=0, keepdims=True)
        return prob_maps, checkout_maps, allowed_scores

    # The masks tile the array, so what the scores do not account for is the
    # mass that left it -- a dart landing more than a board's width from where it
    # was aimed. That is a miss, and scores zero. Booking it explicitly is the
    # point of padding: renormalising instead, as the Gaussian path can afford
    # to, would spread a t's off-board mass back over the trebles.
    zero = int(np.searchsorted(allowed_scores, 0))
    if allowed_scores[zero] != 0:
        raise ValueError("board has no zero-scoring region to absorb misses")
    off_board = 1.0 - prob_maps.sum(axis=0)
    prob_maps[zero] += np.clip(off_board, 0.0, None)

    return prob_maps, checkout_maps, allowed_scores


def expected_score_map(board, Sigma, nu=None):
    """
    ``E[score]`` for every pixel as an aiming point, in a single correlation.

    :func:`transition_maps` gets the same thing by summing ``s * p_s`` over
    sixty-odd masks. Correlating the board itself instead costs one transform
    rather than sixty, which is what makes the scale matching below cheap enough
    to run inside a root-find. Off-board pixels are already zero in ``board``,
    and under the padded correlation so is everything past its edge, so a wide
    dart contributes nothing without any special handling.
    """
    kernel = (
        gaussian_kernel(board.shape[0], Sigma)
        if nu is None
        else student_t_kernel(board.shape[0], Sigma, nu)
    )
    stack = board.astype(np.float64)[None, :, :]
    return _correlate_fft(stack, kernel, pad=nu is not None)[0]


def scoring_average(sigma_mm, nu=None, board_pixels=BOARD_PIXELS, quadro=False,
                    board=None):
    """
    The three-dart average a throw produces in the pure scoring phase: three
    darts at whichever single target maximises expected score.

    This is the statistic the ability bands are named for, and it is the one
    :func:`matched_scale` holds fixed. It is not identical to
    ``players.BAND_AVERAGES``, which come from a full MDP solve and so include
    the checkout phase dragging the number down; it is the same quantity
    measured without solving anything, which is what makes it usable as a
    matching target.
    """
    if board is None:
        board, _ = generate_dartboard(board_pixels, quadro=quadro)
    mm_per_pixel = 2 * DARTBOARD_CONSTANTS["DARTBOARD_RADIUS_MM"] / board.shape[0]
    Sigma = float(sigma_mm) ** 2 * np.eye(2) / mm_per_pixel**2
    return 3.0 * float(expected_score_map(board, Sigma, nu=nu).max())


def matched_scale(sigma_mm, nu, board_pixels=BOARD_PIXELS, quadro=False, board=None):
    """
    The Student-t scale in mm that plays as well as a Gaussian of ``sigma_mm``.

    A t and a Gaussian carrying the same scale are not the same player, so
    comparing them at equal ``sigma`` compares a good thrower with a bad one and
    calls the difference a distributional effect. Something has to be held fixed
    instead, and the choice is not free:

    * **Per-axis standard deviation** is what a statistician reaches for and is
      wrong here. At ``nu = 2.25`` the variance inflation is nine-fold, so an
      elite player comes out with a 2.2mm core -- a third of what notebook 21
      actually fitted, and a caricature of a dart.
    * **P(treble 20)** and **expected score** disagree materially: matched on the
      treble the t scores 2.8-5.1% low, matched on the score it hits 5-17% more
      trebles. The tail has to come from somewhere, and which end of the
      distribution you pin decides where.
    * **The three-dart average** is what the bands already mean -- ``players.py``
      names them for it -- and it is the number a darts player recognises. So it
      is the one pinned here, and the treble rate is then a result rather than an
      assumption.

    ``nu=None`` returns ``sigma_mm`` unchanged, so callers can match
    unconditionally.
    """
    if nu is None:
        return float(sigma_mm)
    if board is None:
        board, _ = generate_dartboard(board_pixels, quadro=quadro)
    target = scoring_average(sigma_mm, nu=None, board=board)

    def gap(log_scale):
        return scoring_average(np.exp(log_scale), nu=nu, board=board) - target

    lo, hi = np.log(0.02 * sigma_mm), np.log(2.0 * sigma_mm)
    if gap(lo) < 0 or gap(hi) > 0:
        raise ValueError(f"no t scale within [0.02, 2] x sigma matches "
                         f"sigma={sigma_mm}, nu={nu}")
    return float(np.exp(brentq(gap, lo, hi, xtol=1e-6)))


def aim_points(board_pixels, margin, point_stride=1):
    """
    Grid of candidate aiming points: every ``point_stride``-th pixel within the
    outer edge of the double ring, plus ``margin`` pixels.

    Returns:
        np.ndarray: (n_points, 2) integer pixel coordinates.
    """
    centre = np.array([board_pixels // 2, board_pixels // 2])
    radius_pixels = int(
        DARTBOARD_CONSTANTS["DOUBLE_OUTER_RADIUS"]
        / DARTBOARD_CONSTANTS["DARTBOARD_RADIUS_MM"]
        * board_pixels
        / 2
    )
    idx = np.arange(0, board_pixels, point_stride)
    ii, jj = np.meshgrid(idx, idx, indexing="ij")
    r = np.sqrt((ii - centre[0]) ** 2 + (jj - centre[1]) ** 2)
    keep = r < radius_pixels + margin
    return np.stack([ii[keep], jj[keep]], axis=1).astype(np.int32)


def transition_arrays(
    board_pixels,
    sigma_mm,
    margin_mm=None,
    point_stride=1,
    quadro=False,
    Sigma_mm=None,
    nu=None,
):
    """
    Build everything the MDP solvers need for a given player and board.

    Args:
        board_pixels (int): resolution of the square board array.
        sigma_mm (float): throwing standard deviation in mm (ignored if
            ``Sigma_mm`` is given). When ``nu`` is set this is the t's **scale**
            and not its standard deviation -- see below.
        margin_mm (float): extra mm beyond the double ring to allow as aiming
            points. Defaults to a quarter of sigma.
        point_stride (int): stride of the aiming-point grid, in pixels.
        quadro (bool): use the Quadro board variant.
        Sigma_mm (np.ndarray): 2x2 matrix in mm^2, for players whose throw is
            not spherically symmetric.
        nu (float): degrees of freedom of a Student-t throw. ``None`` is the
            Gaussian and reproduces every result in the project unchanged.

    A Student-t's scale is not comparable to a Gaussian's sigma -- the same
    number is a much wider player -- so a t built straight from an ability band's
    sigma is not that band's player. :func:`matched_scale` converts, by holding
    fixed the thing the bands actually mean.

    Returns:
        dict with keys ``probs``, ``checkout_probs`` ((n_points, n_scores)
        arrays), ``allowed_scores``, ``points``, ``board``, ``checkouts``,
        ``mm_per_pixel``.
    """
    board, checkouts = generate_dartboard(board_pixels, quadro=quadro)
    mm_per_pixel = 2 * DARTBOARD_CONSTANTS["DARTBOARD_RADIUS_MM"] / board_pixels

    if Sigma_mm is None:
        Sigma_mm = float(sigma_mm) ** 2 * np.eye(2)
    Sigma = np.asarray(Sigma_mm, dtype=np.float64) / mm_per_pixel**2

    if margin_mm is None:
        margin_mm = 0.25 * np.sqrt(np.max(np.diag(Sigma_mm)))
    margin = margin_mm / mm_per_pixel

    prob_maps, checkout_maps, allowed_scores = transition_maps(
        board, checkouts, Sigma, nu=nu
    )
    points = aim_points(board_pixels, margin, point_stride)

    probs = np.ascontiguousarray(prob_maps[:, points[:, 0], points[:, 1]].T)
    checkout_probs = np.ascontiguousarray(
        checkout_maps[:, points[:, 0], points[:, 1]].T
    )

    return {
        "probs": probs,
        "checkout_probs": checkout_probs,
        "allowed_scores": allowed_scores,
        "points": points,
        "board": board,
        "checkouts": checkouts,
        "mm_per_pixel": mm_per_pixel,
    }
