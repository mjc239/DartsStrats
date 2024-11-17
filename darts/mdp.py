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
    gf = gaussian_filter(board, point, Sigma)
    gf = gf / np.sum(gf)
    return prob_score(gf, board, allowed_scores, checkouts)


@njit
def prob_score(filt, board, allowed_scores, checkouts):
    probs = {np.int32(s): 0.0 for s in allowed_scores}
    checkout_probs = {np.int32(s): 0.0 for s in allowed_scores}
    for i in range(filt.shape[0]):
        for j in range(filt.shape[1]):
            probs[np.int32(board[i, j])] += filt[i, j]

            if checkouts[i, j]:
                checkout_probs[np.int32(board[i, j])] += filt[i, j]

    return probs, checkout_probs


@njit
def _compute_state_value(state, values, points, probs, checkout_probs, threshold):
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
    def __init__(
        self, board_pixels, Sigma, margin, game_start, point_stride=1, quadro=None
    ):
        self.board_pixels = board_pixels
        self.quadro = quadro

        self.board, self.checkouts = generate_dartboard(self.board_pixels, self.quadro)
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
        gf = gaussian_filter(self.board, point, self.Sigma)
        gf = gf / np.sum(gf)
        return prob_score(gf, self.board, self.allowed_scores, self.checkouts)

    @cached_property
    def probs(self):
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
