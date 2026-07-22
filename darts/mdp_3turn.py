"""
3-turn-aware MDP for single-player 501.

Extends the published single-player MDP by tracking which dart within a round
is being thrown. This correctly models bust recovery: a bust on dart 1 or 2
resets the score to the round's starting value and forfeits the remaining darts
in that round, just as in real 501.

State: (score, turn, round_start)
  score       -- remaining score (0 means game over)
  turn        -- dart number within the current round (1, 2, or 3)
  round_start -- score at the start of the current round (bust target)

Valid states satisfy: round_start >= score, and at turn=1, score == round_start
(since no dart has been thrown yet this round).

Value convention (matches darts/mdp.py): values are negative.
  values[score, turn-1, round_start] = -(expected darts to checkout)
Terminal state value: values[0, ...] = 0.
"""

import numpy as np
from numba import njit


@njit
def _compute_3turn_values(
    values,              # float64 (game_start+2, 3, game_start+2), modified in place
    probs_arr,           # float64 (n_points, n_scores)
    checkout_probs_arr,  # float64 (n_points, n_scores)
    allowed_scores,      # int32  (n_scores,)
    game_start,
    threshold,
):
    """
    Numba-jitted value iteration for the 3-turn MDP.

    Processes states in order of increasing round_start. For a fixed round_start s:
      - Turn-3 states (score, 3, s): valid throws reference V(ns, 1, ns) for ns < s,
        already finalized in earlier iterations.
      - Turn-2 states (score, 2, s): valid throws reference V(ns, 3, s), updated
        in the same sweep (Gauss-Seidel).
      - Turn-1 state (s, 1, s): valid throws reference V(ns, 2, s), updated
        in the same sweep.
    Busts always reference V(s, 1, s), creating a within-group fixed point that is
    resolved by iterating until group_delta < threshold.
    """
    n_points = probs_arr.shape[0]
    n_scores = probs_arr.shape[1]

    for start in range(2, game_start + 1):

        group_converged = False
        while not group_converged:
            group_delta = 0.0

            # ---- Turn 3: (score, 3, start) for score = 2..start ----
            # Valid throw -> new round V(ns, 1, ns) [already finalized for ns < start]
            # Bust / non-checkout same-score -> V(start, 1, start) [current group]
            for score in range(2, start + 1):
                old_val = values[score, 2, start]
                max_q = -1e20
                for i in range(n_points):
                    q = 0.0
                    for j in range(n_scores):
                        dart_score = allowed_scores[j]
                        p = probs_arr[i, j]
                        cp = checkout_probs_arr[i, j]
                        if dart_score <= score - 2:
                            ns = score - dart_score
                            q += p * (values[ns, 0, ns] - 1.0)
                        elif dart_score == score:
                            q += cp * (-1.0)                             # checkout: V(0) - 1 = -1
                            q += (p - cp) * (values[start, 0, start] - 1.0)  # non-double hit = bust
                        else:
                            q += p * (values[start, 0, start] - 1.0)   # bust
                    if q > max_q:
                        max_q = q
                values[score, 2, start] = max_q
                d = abs(max_q - old_val)
                if d > group_delta:
                    group_delta = d

            # ---- Turn 2: (score, 2, start) for score = 2..start ----
            # Valid throw -> turn 3: V(ns, 3, start) [just updated above]
            for score in range(2, start + 1):
                old_val = values[score, 1, start]
                max_q = -1e20
                for i in range(n_points):
                    q = 0.0
                    for j in range(n_scores):
                        dart_score = allowed_scores[j]
                        p = probs_arr[i, j]
                        cp = checkout_probs_arr[i, j]
                        if dart_score <= score - 2:
                            ns = score - dart_score
                            q += p * (values[ns, 2, start] - 1.0)
                        elif dart_score == score:
                            q += cp * (-1.0)
                            q += (p - cp) * (values[start, 0, start] - 1.0)
                        else:
                            q += p * (values[start, 0, start] - 1.0)
                    if q > max_q:
                        max_q = q
                values[score, 1, start] = max_q
                d = abs(max_q - old_val)
                if d > group_delta:
                    group_delta = d

            # ---- Turn 1: (start, 1, start) ----
            # Valid throw -> turn 2: V(ns, 2, start) [just updated above]
            # Bust self-loops back to V(start, 1, start) [this state]
            old_val = values[start, 0, start]
            max_q = -1e20
            for i in range(n_points):
                q = 0.0
                for j in range(n_scores):
                    dart_score = allowed_scores[j]
                    p = probs_arr[i, j]
                    cp = checkout_probs_arr[i, j]
                    if dart_score <= start - 2:
                        ns = start - dart_score
                        q += p * (values[ns, 1, start] - 1.0)
                    elif dart_score == start:
                        q += cp * (-1.0)
                        q += (p - cp) * (values[start, 0, start] - 1.0)  # bust self-loop
                    else:
                        q += p * (values[start, 0, start] - 1.0)         # bust self-loop
                if q > max_q:
                    max_q = q
            values[start, 0, start] = max_q
            d = abs(max_q - old_val)
            if d > group_delta:
                group_delta = d

            if group_delta < threshold:
                group_converged = True

    return values


def build_probs_arrays(mdp):
    """
    Convert mdp.probs (dict of Numba typed dicts) into plain numpy arrays
    suitable for the @njit inner loop.

    Returns:
        probs_arr:           float64 (n_points, n_scores)
        checkout_probs_arr:  float64 (n_points, n_scores)
        allowed_scores:      int32   (n_scores,)
    """
    points_list = [tuple(pt) for pt in mdp.points]
    n_points = len(points_list)

    sample_p = mdp.probs["probs"][points_list[0]]
    allowed_scores = np.array(sorted(int(k) for k in sample_p.keys()), dtype=np.int32)
    n_scores = len(allowed_scores)

    probs_arr = np.zeros((n_points, n_scores), dtype=np.float64)
    checkout_probs_arr = np.zeros((n_points, n_scores), dtype=np.float64)

    for k, point in enumerate(points_list):
        p = mdp.probs["probs"][point]
        cp = mdp.probs["checkout_probs"][point]
        for j, s in enumerate(allowed_scores):
            s32 = np.int32(s)
            probs_arr[k, j] = float(p[s32])
            checkout_probs_arr[k, j] = float(cp[s32])

    return probs_arr, checkout_probs_arr, allowed_scores


def compute_3turn_values(mdp, threshold=1e-4):
    """
    Compute the 3-turn MDP value function for the given MDP.

    Returns:
        values: float64 array of shape (game_start+2, 3, game_start+2).
            Access as values[score, turn-1, round_start].
            Values are negative: multiply by -1 to get expected darts to checkout.
    """
    probs_arr, checkout_probs_arr, allowed_scores = build_probs_arrays(mdp)
    game_start = mdp.game_start
    values = np.zeros((game_start + 2, 3, game_start + 2), dtype=np.float64)

    _compute_3turn_values(
        values, probs_arr, checkout_probs_arr, allowed_scores, game_start, threshold
    )
    return values
