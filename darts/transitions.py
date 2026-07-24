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
"""

import numpy as np

from darts.dartboards import generate_dartboard, DARTBOARD_CONSTANTS


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


def _correlate_fft(mask_stack, kernel):
    """Cross-correlate each mask in ``mask_stack`` with a centred ``kernel``."""
    kernel_ft = np.fft.fft2(np.fft.ifftshift(kernel))
    out = np.empty_like(mask_stack, dtype=np.float64)
    for k in range(mask_stack.shape[0]):
        out[k] = np.real(np.fft.ifft2(np.fft.fft2(mask_stack[k]) * kernel_ft))
    return out


def transition_maps(board, checkouts, Sigma):
    """
    Score and checkout probability maps for every pixel of the board as an
    aiming point.

    Args:
        board (np.ndarray): (n, n) array of the score at each pixel.
        checkouts (np.ndarray): (n, n) boolean mask of checkout regions.
        Sigma (np.ndarray): 2x2 covariance matrix, in pixel units.

    Returns:
        tuple:
            - prob_maps (np.ndarray): (n_scores, n, n), probability of each
              board score when aiming at each pixel.
            - checkout_maps (np.ndarray): (n_scores, n, n), probability of
              hitting each score in a checkout region.
            - allowed_scores (np.ndarray): (n_scores,) sorted board scores.
    """
    allowed_scores = np.unique(board).astype(np.int32)
    kernel = gaussian_kernel(board.shape[0], Sigma)

    masks = np.stack([(board == s).astype(np.float64) for s in allowed_scores])
    prob_maps = _correlate_fft(masks, kernel)

    co_masks = masks * checkouts[None, :, :]
    checkout_maps = _correlate_fft(co_masks, kernel)

    # Numerical noise from the FFT can produce tiny negative probabilities.
    np.clip(prob_maps, 0.0, None, out=prob_maps)
    np.clip(checkout_maps, 0.0, None, out=checkout_maps)
    np.minimum(checkout_maps, prob_maps, out=checkout_maps)
    prob_maps /= prob_maps.sum(axis=0, keepdims=True)

    return prob_maps, checkout_maps, allowed_scores


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
    board_pixels, sigma_mm, margin_mm=None, point_stride=1, quadro=False, Sigma_mm=None
):
    """
    Build everything the MDP solvers need for a given player and board.

    Args:
        board_pixels (int): resolution of the square board array.
        sigma_mm (float): throwing standard deviation in mm (ignored if
            ``Sigma_mm`` is given).
        margin_mm (float): extra mm beyond the double ring to allow as aiming
            points. Defaults to a quarter of sigma.
        point_stride (int): stride of the aiming-point grid, in pixels.
        quadro (bool): use the Quadro board variant.
        Sigma_mm (np.ndarray): 2x2 covariance matrix in mm^2, for players whose
            throw is not spherically symmetric.

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

    prob_maps, checkout_maps, allowed_scores = transition_maps(board, checkouts, Sigma)
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
