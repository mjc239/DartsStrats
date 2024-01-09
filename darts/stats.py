import numpy as np
from dartboards import DARTBOARD_CONSTANTS
from typing import Callable


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
