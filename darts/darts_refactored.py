import numpy as np
import functools
from scipy.stats import multivariate_normal
from collections import defaultdict
from typing import Callable

DARTBOARD_CONSTANTS = {
    "DARTBOARD_RADIUS_MM": 225.5,
    "INNER_BULLSEYE_RADIUS_MM": 6.35,
    "OUTER_BULLSEYE_RADIUS_MM": 15.9,
    "TRIPLE_INNER_RADIUS": 99,
    "TRIPLE_OUTER_RADIUS": 107,
    "DOUBLE_INNER_RADIUS": 162,
    "DOUBLE_OUTER_RADIUS": 170,
    "SEGMENTS": {
        20: [(9 / 20, 11 / 20)],
        1: [(7 / 20, 9 / 20)],
        18: [(5 / 20, 7 / 20)],
        4: [(3 / 20, 5 / 20)],
        13: [(1 / 20, 3 / 20)],
        6: [(-1 / 20, 1 / 20)],
        10: [(-3 / 20, -1 / 20)],
        15: [(-5 / 20, -3 / 20)],
        2: [(-7 / 20, -5 / 20)],
        17: [(-9 / 20, -7 / 20)],
        3: [(-11 / 20, -9 / 20)],
        19: [(-13 / 20, -11 / 20)],
        7: [(-15 / 20, -13 / 20)],
        16: [(-17 / 20, -15 / 20)],
        8: [(-19 / 20, -17 / 20)],
        11: [(19 / 20, 1), (-1, -19 / 20)],
        14: [(17 / 20, 19 / 20)],
        9: [(15 / 20, 17 / 20)],
        12: [(13 / 20, 15 / 20)],
        5: [(11 / 20, 13 / 20)],
    },
}


def generate_dartboard(pixels: int) -> np.ndarray:
    """Generate a dartboard as a numpy array, with each entry indicating the
    score at that pixel.

    Args:
        pixels (int): Dimension of the square array, indicating how pixels
        each side of the image should have.

    Returns:
        np.ndarray: numpy array, with each entry indicating the score at
        that pixel.
    """
    radius = DARTBOARD_CONSTANTS["DARTBOARD_RADIUS_MM"]

    # Create grid covering full dartboard area, with the
    # correct number of pixels
    x, y = np.meshgrid(
        np.linspace(-radius, radius, pixels), np.linspace(-radius, radius, pixels)
    )

    # Convert to polar coordinates
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)

    # Score multiplier mask
    bi = DARTBOARD_CONSTANTS["INNER_BULLSEYE_RADIUS_MM"]
    bo = DARTBOARD_CONSTANTS["OUTER_BULLSEYE_RADIUS_MM"]
    di = DARTBOARD_CONSTANTS["DOUBLE_INNER_RADIUS"]
    do = DARTBOARD_CONSTANTS["DOUBLE_OUTER_RADIUS"]
    ti = DARTBOARD_CONSTANTS["TRIPLE_INNER_RADIUS"]
    to = DARTBOARD_CONSTANTS["TRIPLE_OUTER_RADIUS"]

    single_rings = (((r >= bo) & (r < ti)) | ((r >= ti) & (r < di))).astype(int)
    double_rings = ((r >= di) & (r < do)).astype(int)
    triple_rings = ((r >= ti) & (r < to)).astype(int)
    outer_bull = ((r >= bi) & (r < bo)).astype(int)
    inner_bull = (r < bi).astype(int)
    multiplier_mask = single_rings + 2 * double_rings + 3 * triple_rings

    score_arr = np.zeros([pixels, pixels])

    # Loop around dartboard in segments
    for score, intervals in DARTBOARD_CONSTANTS["SEGMENTS"].items():
        score_segment = np.zeros([pixels, pixels])
        for interval in intervals:
            low_angle, high_angle = interval[0] * np.pi, interval[1] * np.pi
            score_segment = np.logical_or(
                score_segment, ((theta < high_angle) & (theta >= low_angle)).astype(int)
            )

        score_arr += score * score_segment * multiplier_mask

    # Bullseyes
    score_arr += 25 * outer_bull + 50 * inner_bull

    return score_arr


def gaussian_filter(board: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """Generates a Gaussian filter with a specified mean and variance.

    Args:
        board (np.ndarray): The dartboard dsecribed by a numpy array.
        mu (np.ndarray): Mean vector. Should be a length 2 array.
        Sigma (np.ndarray): Variance matrix. Should be a 2x2 array.

    Returns:
        np.ndarray: Gaussian filter.
    """
    if len(mu) != 2:
        raise ValueError("mu should have length 2!")
    if Sigma.shape != (2, 2):
        raise ValueError("Sigma should be a 2x2 array!")
    if Sigma[0, 1] != Sigma[1, 0]:
        raise ValueError("Sigma should be a symmetric array!")
    if board.shape[0] != board.shape[1]:
        raise ValueError("Board should be a square array!")

    pixels = board.shape[0]
    radius = DARTBOARD_CONSTANTS["DARTBOARD_RADIUS_MM"]

    x, y = np.meshgrid(
        np.linspace(-radius, radius, pixels),
        np.linspace(-radius, radius, pixels),
    )

    det = Sigma[0, 0] * Sigma[1, 1] - Sigma[0, 1] ** 2
    exponent = (
        1.0
        / det
        * (
            Sigma[1, 1] * (x - mu[0]) ** 2
            - 2 * Sigma[0, 1] * (x - mu[0]) * (y - mu[1])
            + Sigma[0, 0] * (y - mu[1]) ** 2
        )
    )

    filter = np.exp(-exponent / 2.0)
    filter /= 2 * np.pi * np.sqrt(det)

    return filter


def expected_score(
    board: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    padding: int = None,
    score_function: Callable[[np.ndarray], np.ndarray] = None,
) -> np.ndarray:
    """Computes the expected score at all positions in the board, with the
    specified throwing distribution (defined by mu, Sigma).

    Args:
        board (np.ndarray): The dartboard described by a numpy array.
        mu (np.ndarray): Mean vector. Should be a length 2 array.
        Sigma (np.ndarray): Variance matrix. Should be a 2x2 array.
        padding (int, optional): Add zero padding to array before proceeding. Defaults to None.
        score_function (Callable[[np.ndarray], np.ndarray], optional): Function to apply to
        the board values before computing the expected values. Defaults to None.

    Returns:
        np.ndarray: Array of expected scores at each pixel.
    """
    # Add padding if provided
    if padding:
        board = np.pad(board, padding)

    filter = gaussian_filter(board, -mu, Sigma)
    # Why is this necessary?
    filter /= np.sum(filter)

    # Apply scoring function if provided
    score_array = score_function(board) if score_function else board

    # Fourier transform the board and filter
    board_ft = np.fft.fft2(score_array)
    filter_ft = np.fft.fft2(np.fft.ifftshift(filter))

    # Convolute by multiplying FTs
    # and inverting FT
    prod_ft = board_ft * filter_ft
    exp_map = np.real(np.fft.ifft2(prod_ft))

    # Remove padding
    if padding:
        exp_map = exp_map[padding:-padding, padding:-padding]

    return exp_map


def variance_score(
    board: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, padding: np.ndarray = False
) -> np.ndarray:
    """Computes the variance of the score at each point of the dartboard.

    Args:
        board (np.ndarray): The dartboard described by a numpy array.
        mu (np.ndarray): Mean vector. Should be a length 2 array.
        Sigma (np.ndarray): Variance matrix. Should be a 2x2 array.
        padding (int, optional): Add zero padding to array before proceeding. Defaults to None.

    Returns:
        np.ndarray: Array of score variances at each pixel.
    """
    expectation_X2 = expected_score(
        board, mu, Sigma, padding=padding, score_function=lambda x: x * x
    )
    expectation_X = expected_score(board, mu, Sigma, padding=padding)

    variance = expectation_X2 - expectation_X * expectation_X

    return variance


def std_score(
    board: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, padding: np.ndarray = False
) -> np.ndarray:
    """Computes the standard deviation of the score at each point of the dartboard.

    Args:
        board (np.ndarray): _description_
        mu (np.ndarray): _description_
        Sigma (np.ndarray): _description_
        padding (np.ndarray, optional): _description_. Defaults to False.

    Returns:
        np.ndarray: Array of score standard deviations at each pixel.
    """
    variance = variance_score(board, mu, Sigma, padding=padding)

    return np.sqrt(np.abs(variance))
