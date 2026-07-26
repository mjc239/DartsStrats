"""
What is practice worth?

Two questions a player actually asks:

1. *If I get more accurate, how much better do I get?* -- differentiate the
   solved model with respect to sigma.
2. *Should I practise doubles or trebles?* -- which needs accuracy that depends
   on **where you are aiming**, not a single sigma for the whole board.

The model used here is the 3-dart single-player MDP with the **minimum-visits**
objective, converted into a leg win probability against a fixed opponent. That
is the compromise the question demands: a club match is a race in visits, so
visits is the right currency, and the single-player model is within ~1.4 points
of win probability of the full two-player game while solving in seconds rather
than a quarter of an hour.

Aim-dependent accuracy
----------------------
Sigma as a function of the aiming point cannot go through a single FFT, because
each aim point would need its own kernel. But it does not need to: partition the
aiming grid into a handful of classes (doubles, trebles, everything else), run
one FFT per class, and take each aim point's row from its own class. Three
passes instead of one, and exact.
"""

import numpy as np

from darts.dartboards import DARTBOARD_CONSTANTS, generate_dartboard
from darts.mdp_3turn import ThreeDartMDP
from darts.transitions import aim_points, transition_maps
from darts.utils import mm_per_pixel, region_label


def classify_points(points, board_pixels):
    """
    Label each aiming point by the kind of target it is: ``"double"``,
    ``"treble"`` or ``"other"``.
    """
    out = np.empty(len(points), dtype=object)
    for i, p in enumerate(points):
        lab = region_label(p, board_pixels)
        out[i] = ("double" if lab.startswith("D") or lab == "BULL"
                  else "treble" if lab.startswith("T")
                  else "other")
    return out


def transition_arrays_by_class(board_pixels, sigma_by_class, point_stride=2,
                               margin_mm=None, quadro=False):
    """
    Transition probabilities where the throw's spread depends on what kind of
    target is being aimed at.

    Args:
        board_pixels (int): board resolution.
        sigma_by_class (dict): ``{"double": mm, "treble": mm, "other": mm}``.
        point_stride (int): aiming grid stride.
        margin_mm (float): aiming margin beyond the double ring; defaults to a
            quarter of the largest sigma.
        quadro (bool): Quadro board.

    Returns:
        dict: same keys as :func:`darts.transitions.transition_arrays`.
    """
    board, checkouts = generate_dartboard(board_pixels, quadro=quadro)
    mmpp = mm_per_pixel(board_pixels)
    if margin_mm is None:
        margin_mm = 0.25 * max(sigma_by_class.values())
    points = aim_points(board_pixels, margin_mm / mmpp, point_stride)
    cls = classify_points(points, board_pixels)

    probs = checkout_probs = None
    scores = None
    for name, sigma in sigma_by_class.items():
        sel = np.flatnonzero(cls == name)
        if not len(sel):
            continue
        Sigma = (sigma / mmpp) ** 2 * np.eye(2)
        pm, cm, S = transition_maps(board, checkouts, Sigma)
        if probs is None:
            scores = S
            probs = np.zeros((len(points), len(S)))
            checkout_probs = np.zeros((len(points), len(S)))
        rows, cols = points[sel, 0], points[sel, 1]
        probs[sel] = pm[:, rows, cols].T
        checkout_probs[sel] = cm[:, rows, cols].T

    return {"probs": np.ascontiguousarray(probs),
            "checkout_probs": np.ascontiguousarray(checkout_probs),
            "allowed_scores": scores, "points": points,
            "board": board, "checkouts": checkouts, "mm_per_pixel": mmpp,
            "point_class": cls}


def leg_win_probability(my_visits, opponent_visits, throws_first=True,
                        n_terms=60):
    """
    Turn two expected-visit numbers into a leg win probability.

    Models each player's visit count as a geometric-like race: with mean
    ``m`` visits to finish, the chance of needing more than ``k`` visits is
    approximated by a geometric tail with the same mean. Crude compared with
    solving the two-player game, but it needs only the single-player model and
    reproduces the qualitative shape; use it for *differences* between nearby
    abilities, which is what a practice calculator is for.
    """
    pa = 1.0 / my_visits
    pb = 1.0 / opponent_visits
    # P(A finishes on visit k) = (1-pa)^(k-1) pa, similarly B; A throws first
    k = np.arange(1, n_terms + 1)
    fa = (1 - pa) ** (k - 1) * pa
    fb = (1 - pb) ** (k - 1) * pb
    surv_b = (1 - pb) ** (k - 1 if throws_first else k)
    return float((fa * surv_b).sum() / (1 - ((1 - pa) * (1 - pb)) ** n_terms))


def sigma_sensitivity(sigmas, values, at=None):
    """
    d(value)/d(sigma) by central differences on a solved sweep.

    Args:
        sigmas (array): the sigma grid, ascending.
        values (array): the solved value at each sigma (e.g. expected visits).
        at (float): a particular sigma to report; defaults to all of them.

    Returns:
        np.ndarray or float: the derivative.
    """
    sigmas = np.asarray(sigmas, float)
    values = np.asarray(values, float)
    d = np.gradient(values, sigmas)
    if at is None:
        return d
    return float(np.interp(at, sigmas, d))


def practice_split_value(base_sigma, improvement=0.2, board_pixels=None,
                         point_stride=None, game_start=501):
    """
    Compare spending practice time on doubles against trebles.

    Improves one class of target by ``improvement`` (a fraction of sigma) and
    re-solves, holding everything else fixed. The comparison is like for like:
    the same proportional gain in accuracy, applied to different targets.

    Args:
        base_sigma (float): the player's current sigma in mm.
        improvement (float): fractional reduction in sigma for the practised
            class, e.g. 0.2 for a 20% tighter group.

    Returns:
        list[dict]: one row per scenario with the resulting expected visits.
    """
    from darts import players
    board_pixels = board_pixels or players.BOARD_PIXELS
    point_stride = point_stride or players.POINT_STRIDE_SINGLE

    better = base_sigma * (1 - improvement)
    scenarios = {
        "no practice": {"double": base_sigma, "treble": base_sigma, "other": base_sigma},
        "doubles only": {"double": better, "treble": base_sigma, "other": base_sigma},
        "trebles only": {"double": base_sigma, "treble": better, "other": base_sigma},
        "everything": {"double": better, "treble": better, "other": better},
    }
    rows = []
    for name, smap in scenarios.items():
        tr = transition_arrays_by_class(board_pixels, smap, point_stride=point_stride)
        m = ThreeDartMDP(tr["probs"], tr["checkout_probs"], tr["allowed_scores"],
                         game_start, dart_cost=0.0, turn_cost=1.0).solve()
        rows.append({"scenario": name,
                     "sigma double": smap["double"], "sigma treble": smap["treble"],
                     "sigma other": smap["other"],
                     "expected visits": round(-m.V1[game_start], 4)})
    base = rows[0]["expected visits"]
    for r in rows:
        r["visits saved"] = round(base - r["expected visits"], 4)
    return rows
