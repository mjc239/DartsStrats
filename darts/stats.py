import numpy as np
from darts.dartboards import DARTBOARD_CONSTANTS
from typing import Callable
from numba import njit


@njit
def gaussian_filter(board: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """Generates a Gaussian filter with a specified mean and variance.

    .. warning::
       ``mu`` and ``Sigma`` use opposite index orders, which matters as soon as
       the throw is not spherically symmetric:

       * ``mu`` is ``(row, column)`` -- ``mu[0]`` shifts down the array (the
         y direction), ``mu[1]`` shifts across it (the x direction).
       * ``Sigma`` is in ``(x, y)`` order -- ``Sigma[0, 0]`` is the variance
         across columns (horizontal), ``Sigma[1, 1]`` the variance down rows
         (vertical), and ``Sigma[0, 1]`` their covariance with the usual sign
         (positive tilts the cloud up and to the right on the board).

       So a covariance matrix fitted to throws in ordinary (horizontal,
       vertical) millimetre coordinates can be passed straight in, but an aim
       point cannot -- its coordinates must be swapped.

    Args:
        board (np.ndarray): The dartboard described by a numpy array.
        mu (np.ndarray): Mean vector, in (row, column) order. Length 2.
        Sigma (np.ndarray): Covariance matrix in (x, y) order. 2x2.

    Returns:
        np.ndarray: Gaussian filter.
    """
    # Check input dimensions
    if len(mu) != 2:
        raise ValueError("mu should have length 2!")
    if Sigma.shape != (2, 2):
        raise ValueError("Sigma should be a 2x2 array!")
    if Sigma[0, 1] != Sigma[1, 0]:
        raise ValueError("Sigma should be a symmetric array!")
    if board.shape[0] != board.shape[1]:
        raise ValueError("Board should be a square array!")

    pixels = board.shape[0]

    # Generate a grid of x and y values.
    # This must have unit spacing and put 0 exactly on pixel `pixels // 2`:
    # np.linspace(-n//2, n//2, n) has spacing n/(n-1) and straddles zero, which
    # displaces the throwing distribution from its intended aim point by up to
    # a pixel, with the error growing towards the edge of the board.
    xx = (np.arange(pixels) - pixels // 2).astype(np.float64)
    x = np.empty(shape=(xx.size, xx.size), dtype=xx.dtype)
    for j in range(xx.size):
        x[:, j] = xx[j]
    y = x.T

    # Compute the exponent of the Gaussian
    det = Sigma[0, 0] * Sigma[1, 1] - Sigma[0, 1] ** 2
    exponent = (
        1.0
        / det
        * (
            Sigma[1, 1] * (x - mu[1]) ** 2
            - 2 * Sigma[0, 1] * (x - mu[1]) * (y - mu[0])
            + Sigma[0, 0] * (y - mu[0]) ** 2
        )
    )

    # Compute the filter
    filter = np.exp(-exponent / 2.0)
    filter /= 2 * np.pi * np.sqrt(det)

    return filter


def expected_score(
    board: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    padding: int | None = None,
    score_function: Callable[[np.ndarray], np.ndarray] | None = None,
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
    board: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, padding: int | None = None
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

    # Compute the variance as E[X^2] - E[X]^2
    variance = expectation_X2 - expectation_X * expectation_X

    return variance


def std_score(
    board: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, padding: int | None = None
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
