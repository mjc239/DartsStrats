"""
Module for defining and solving a Markov Decision Process (MDP) for a single-player
darts game (e.g., 501) with continuous aiming points and Gaussian throw accuracy.
"""

import numpy as np
from darts.dartboards import generate_dartboard, DARTBOARD_CONSTANTS
from darts.stats import gaussian_filter
from itertools import product
from functools import cached_property
from numba import njit, types, prange
from numba.typed import Dict
from tqdm import tqdm


@njit
def compute_transition_probs_from_point_njit(
    board, point, Sigma, allowed_scores, checkouts
):
    """
    Computes the probability distribution over scores and checkout probabilities
    for aiming at a specific point on the board, given Gaussian throw accuracy.

    This is a Numba-jitted version for performance.

    Args:
        board (np.ndarray): Pixel representation of the dartboard scores.
        point (np.ndarray): The target [x, y] pixel coordinates to aim at.
        Sigma (np.ndarray): 2x2 covariance matrix for the Gaussian throw distribution.
        allowed_scores (np.ndarray): Array of unique possible scores on the board.
        checkouts (np.ndarray): Boolean mask indicating checkout segments (doubles/bull).

    Returns:
        tuple[Dict[int, float], Dict[int, float]]: A tuple containing:
            - probs: Dictionary mapping score to its probability.
            - checkout_probs: Dictionary mapping score to its checkout probability.
    """
    gf = gaussian_filter(board, point, Sigma)
    gf = gf / np.sum(gf)
    return prob_score(gf, board, allowed_scores, checkouts)


@njit
def prob_score(filt, board, allowed_scores, checkouts):
    """
    Calculates the probability of hitting each score based on a probability
    distribution filter applied over the board. Also calculates the probability
    of hitting checkout segments for each score.

    Args:
        filt (np.ndarray): A 2D probability distribution (filter) over the board pixels.
                           Must sum to 1.
        board (np.ndarray): Pixel representation of the dartboard scores.
        allowed_scores (np.ndarray): Array of unique possible scores on the board.
        checkouts (np.ndarray): Boolean mask indicating checkout segments (doubles/bull).

    Returns:
        tuple[Dict[int, float], Dict[int, float]]: A tuple containing:
            - probs: Dictionary mapping score to its probability.
            - checkout_probs: Dictionary mapping score to its checkout probability
                             (probability of hitting that score *and* it being a checkout).
    """
    probs = {np.int32(s): 0.0 for s in allowed_scores}
    checkout_probs = {np.int32(s): 0.0 for s in allowed_scores}
    for i in range(filt.shape[0]):
        for j in range(filt.shape[1]):
            score = np.int32(board[i, j])
            probability = filt[i, j]
            probs[score] += probability

            if checkouts[i, j]:
                checkout_probs[score] += probability

    return probs, checkout_probs


@njit
def _compute_state_value(state, values, points, probs, checkout_probs, threshold):
    """
    Performs value iteration for a single state to find its optimal expected value
    (minimum expected throws to finish).

    Uses the Bellman equation: V(s) = min_a Σ_{s'} P(s'|s, a) * (R(s, a, s') + γ * V(s'))
    Here, R = -1 (cost per throw), γ = 1 (no discounting, assumed).
    The equation simplifies to: V(s) = min_a [ Σ_{s'} P(s'|s, a) * (V(s') - 1) ]
    Or equivalently: V(s) = -1 + min_a [ Σ_{s'} P(s'|s, a) * V(s') ]

    Args:
        state (int): The current score (state) to evaluate.
        values (Dict[int, float]): Dictionary mapping state (score) to its current estimated value.
                                   This dictionary is updated in-place.
        points (np.ndarray): Array of possible aiming points [[x1, y1], [x2, y2], ...].
        probs (Dict[str, Dict[int, float]]): Dictionary mapping aiming point string ("x,y")
                                             to score probabilities Dict[score, prob].
        checkout_probs (Dict[str, Dict[int, float]]): Dictionary mapping aiming point string ("x,y")
                                                       to checkout probabilities Dict[score, prob].
        threshold (float): The convergence threshold for the value iteration delta.

    Returns:
        Dict[int, float]: The updated values dictionary.
    """
    if state == 0 or state == 1:
        values[state] = 0
        return values

    delta = 1e20
    while delta > threshold:
        max_q = -1e20

        for point in points:
            key = f"{point[0]},{point[1]}"
            p = probs[key]
            cp = checkout_probs[key]
            q = 0

            for score in p:
                # Valid throw
                if score <= state - 2:
                    q += p[score] * (values[state - score] - 1)

                # Checkout
                elif score == state:
                    q += cp[score] * (values[0] - 1)
                    q += (p[score] - cp[score]) * (values[state] - 1)

                # Bust
                else:
                    q += p[score] * (values[state] - 1)
            if q >= max_q:
                max_q = q

        delta = abs(max_q - values[state])
        values[state] = max_q

    return values


@njit
def _compute_actions(values, points, probs, checkout_probs):
    """
    Computes the optimal policy (best aiming point for each state/score)
    based on the converged state values.

    Args:
        values (Dict[int, float]): Dictionary mapping state (score) to its optimal value.
        points (np.ndarray): Array of possible aiming points [[x1, y1], [x2, y2], ...].
        probs (Dict[str, Dict[int, float]]): Dictionary mapping aiming point string ("x,y")
                                             to score probabilities Dict[score, prob].
        checkout_probs (Dict[str, Dict[int, float]]): Dictionary mapping aiming point string ("x,y")
                                                       to checkout probabilities Dict[score, prob].

    Returns:
        Dict[int, np.ndarray]: The optimal policy, mapping state (score) to the
                               best aiming point [x, y].
    """
    policy = Dict.empty(key_type=types.int32, value_type=types.int32[:])

    for state in values:
        if state == 0 or state == 1:
            policy[state] = points[0]
            continue

        max_q = -np.inf
        max_a = None

        for point in points:
            key = f"{point[0]},{point[1]}"
            p = probs[key]
            cp = checkout_probs[key]
            q = 0

            for score in p:
                # Valid throw
                if score <= state - 2:
                    q += p[score] * (values[state - score] - 1)

                # Checkout
                elif score == state:
                    q += cp[score] * (values[0] - 1)
                    q += (p[score] - cp[score]) * (values[state] - 1)

                # Bust
                else:
                    q += p[score] * (values[state] - 1)

            if q >= max_q:
                max_q = q
                max_a = point

        policy[state] = max_a

    return policy


class SinglePlayerContinuousMDP:
    """
    Represents and solves the MDP for a single-player darts game (like 501)
    aiming to minimize the expected number of throws to reach zero.

    Assumes a continuous state space for aiming points and models throw
    accuracy using a Gaussian distribution.

    Attributes:
        board_pixels (int): The width and height of the square pixel grid representing the board.
        Sigma (np.ndarray): 2x2 covariance matrix for the Gaussian throw distribution.
                            Represents player accuracy. Smaller values mean higher accuracy.
        margin (int): Extra pixel margin around the board's throwing area to consider
                      aiming points, accounting for inaccuracy.
        game_start (int): The starting score (e.g., 501).
        point_stride (int): The step size used when selecting aiming points from the grid.
                            A stride of 1 uses all pixels, higher strides use fewer points.
        quadro (bool | None): Whether to include a Quadro (Q) ring (specific board variant).
        board (np.ndarray): Pixel grid representing scores on the dartboard.
        checkouts (np.ndarray): Boolean mask indicating checkout segments (doubles/bull).
        allowed_scores (np.ndarray): Unique scores possible on the board.
        centre (np.ndarray): [x, y] coordinates of the board's center pixel.
        values (Dict[int, float]): Dictionary mapping score (state) to its calculated
                                   optimal expected value (min throws to win). Initialized to 0.
        policy (Dict[int, np.ndarray] | None): Dictionary mapping score (state) to the
                                               optimal aiming point [x, y]. Calculated by
                                               `compute_values`. Initially None.
    """

    def __init__(
        self, board_pixels, Sigma, margin, game_start, point_stride=1, quadro=None
    ):
        """
        Initializes the SinglePlayerContinuousMDP.

        Args:
            board_pixels (int): The width/height of the pixel board representation.
            Sigma (np.ndarray): 2x2 covariance matrix for Gaussian throw accuracy.
            margin (int): Pixel margin around the board for valid aim points.
            game_start (int): The starting score for the game (e.g., 501).
            point_stride (int, optional): Stride for selecting aim points. Defaults to 1.
            quadro (bool | None, optional): Include Quadro ring if True. Defaults to None.
        """
        self.board_pixels = board_pixels
        self.quadro = quadro

        self.board, self.checkouts = generate_dartboard(
            self.board_pixels, quadro=bool(self.quadro)
        )
        self.allowed_scores = np.unique(self.board)
        self.centre = np.array([int(self.board_pixels / 2), int(self.board_pixels / 2)])

        self.Sigma = Sigma
        self.margin = margin
        self.game_start = game_start
        self.values = {score: 0 for score in range(game_start + 1)}
        self.policy = None
        self.point_stride = point_stride

    @cached_property
    def points(self):
        """
        Calculates and caches the grid of possible aiming points within the board
        radius plus the margin, considering the point_stride.

        Returns:
            np.ndarray: An array of [x, y] coordinates for potential aiming points.
        """
        radius_pixels = int(
            DARTBOARD_CONSTANTS["DOUBLE_OUTER_RADIUS"]
            / DARTBOARD_CONSTANTS["DARTBOARD_RADIUS_MM"]
            * self.board_pixels
            / 2
        )
        return np.array(
            [
                [i, j]
                for i, j in product(
                    range(0, self.board_pixels, self.point_stride),
                    range(0, self.board_pixels, self.point_stride),
                )
                if np.linalg.norm(np.array([i, j]) - self.centre)
                < radius_pixels + self.margin
            ]
        )

    def compute_transition_probs_from_point(self, point):
        """
        Computes the probability distribution over scores and checkout probabilities
        for aiming at a specific point. This is the non-jitted version, potentially
        used if Numba is not available or for testing.

        Args:
            point (np.ndarray): The target [x, y] pixel coordinates relative to top-left.

        Returns:
            tuple[Dict[int, float], Dict[int, float]]: Probabilities and checkout probabilities.
        """
        gf = gaussian_filter(self.board, point, self.Sigma)
        gf = gf / np.sum(gf)
        return prob_score(gf, self.board, self.allowed_scores, self.checkouts)

    @cached_property
    def probs(self):
        """
        Calculates and caches the transition probabilities (score distributions and
        checkout probabilities) for *all* valid aiming points defined by `self.points`.

        Uses the Numba-jitted function for performance.

        Returns:
            Dict[str, Dict]: A dictionary containing two sub-dictionaries:
                - "probs": Maps tuple(point) -> {score: probability}
                - "checkout_probs": Maps tuple(point) -> {score: checkout_probability}
        """
        probs = {}
        checkout_probs = {}
        for start in tqdm(self.points):
            (
                probs[tuple(start)],
                checkout_probs[tuple(start)],
            ) = compute_transition_probs_from_point_njit(
                self.board,
                start - self.centre,
                self.Sigma,
                self.allowed_scores,
                self.checkouts,
            )
        return {"probs": probs, "checkout_probs": checkout_probs}

    def compute_values(self, threshold):
        """
        Computes the optimal state values (minimum expected throws) for all scores
        from game_start down to 0 using value iteration, and then determines the
        optimal policy (best aiming point for each score).

        Updates `self.values` and `self.policy`.

        Args:
            threshold (float, optional): The convergence threshold for value iteration.
                                         Defaults to 1e-6.
        """

        # Setup for numba function
        new_d_values = Dict.empty(key_type=types.int32, value_type=types.float64)
        for k, v in self.values.items():
            new_d_values[k] = v

        d_probs = Dict.empty(
            key_type=types.string, value_type=types.DictType(types.int32, types.float64)
        )
        for k, v in self.probs["probs"].items():
            d_probs[",".join([str(x) for x in k])] = v

        d_cprobs = Dict.empty(
            key_type=types.string, value_type=types.DictType(types.int32, types.float64)
        )
        for k, v in self.probs["checkout_probs"].items():
            d_cprobs[",".join([str(x) for x in k])] = v

        for state in tqdm(sorted(self.values)):
            d_values = Dict.empty(key_type=types.int32, value_type=types.float64)
            for k, v in new_d_values.items():
                d_values[k] = v

            if state >= 2:
                d_values[state] = d_values[state - 1]

            new_d_values = _compute_state_value(
                state, d_values, self.points, d_probs, d_cprobs, threshold
            )

        policy = _compute_actions(d_values, self.points, d_probs, d_cprobs)

        self.values = dict(d_values)
        self.policy = dict(policy)
