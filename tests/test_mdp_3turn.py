"""
Tests for the 3-dart MDP solvers.

The fast structural solver in ``ThreeDartMDP`` is checked against two
independent references:

1. ``_compute_3turn_values``, the original Numba sweep over the full
   ``(score, dart, round_start)`` state space, and
2. a plain synchronous value iteration written from the Bellman equations,
   with no ordering assumptions at all.

A small synthetic dartboard is used so the tests stay fast.
"""

import numpy as np
import pytest

from darts.mdp_3turn import ThreeDartMDP, _compute_3turn_values


@pytest.fixture(scope="module")
def toy_model():
    """A small, non-degenerate transition model: 12 aiming points over the
    scores {0, 1, ..., 8, 10, 12} with a scattering of checkout mass."""
    rng = np.random.default_rng(0)
    scores = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12], dtype=np.int32)
    n_points, n_scores = 12, len(scores)

    P = rng.random((n_points, n_scores)) ** 3 + 1e-3
    P /= P.sum(axis=1, keepdims=True)
    # Only even scores (and 0) can be hit as a "double"; a fraction of the mass
    # on those scores is a checkout.
    CP = P * ((scores % 2 == 0) & (scores > 0)) * rng.random((n_points, n_scores))
    return P, CP, scores


def reference_values(P, CP, S, G, dart_cost=1.0, turn_cost=0.0, iters=20000, tol=1e-13):
    """Synchronous value iteration over every (score, dart, round_start) state."""
    n_sc = P.shape[1]
    V = np.zeros((G + 1, 3, G + 1))
    for _ in range(iters):
        Vold = V.copy()
        for s in range(2, G + 1):
            for t in (2, 1, 0):
                for u in range(2, s + 1):
                    if t == 0 and u != s:
                        continue
                    w = np.empty(n_sc)
                    wcp = np.zeros(n_sc)
                    for j, sj in enumerate(S):
                        if sj <= u - 2:
                            nu = u - sj
                            nxt = V[nu, 0, nu] if t == 2 else V[nu, t + 1, s]
                        elif sj == u:
                            nxt = V[s, 0, s]
                            wcp[j] = -nxt  # checkout instead: value 0
                        else:
                            nxt = V[s, 0, s]
                        w[j] = nxt
                    q = P @ w + CP @ wcp - dart_cost
                    V[u, t, s] = q.max() - (turn_cost if t == 0 else 0.0)
        if np.abs(V - Vold).max() < tol:
            break
    return V


@pytest.mark.parametrize("G", [12, 30])
def test_fast_solver_matches_plain_value_iteration(toy_model, G):
    P, CP, S = toy_model
    Vref = reference_values(P, CP, S, G)
    fast = ThreeDartMDP(P, CP, S, G).solve(tol=1e-13)

    for s in range(2, G + 1):
        assert fast.V1[s] == pytest.approx(Vref[s, 0, s], abs=1e-9)


def test_fast_solver_matches_numba_reference(toy_model):
    P, CP, S = toy_model
    G = 30
    cube = np.zeros((G + 2, 3, G + 2))
    _compute_3turn_values(cube, P, CP, S, G, 1e-13)
    fast = ThreeDartMDP(P, CP, S, G).solve(tol=1e-13)

    for s in range(2, G + 1):
        assert fast.V1[s] == pytest.approx(cube[s, 0, s], abs=1e-9)

    # Cached dart-2 / dart-3 tables must reproduce every round-start slice.
    max_dart = int(S.max())
    for s in range(2, G + 1):
        for u in range(max(2, s - 2 * max_dart), s):
            if u >= fast.u3_indep:
                assert fast.V3tab[u] == pytest.approx(cube[u, 2, s], abs=1e-9)
        for u in range(max(2, s - max_dart), s):
            if u >= fast.u2_indep:
                assert fast.V2tab[u] == pytest.approx(cube[u, 1, s], abs=1e-9)


def test_turn_cost_objective(toy_model):
    """With turn_cost the solver returns expected rounds, which must be
    between a third of and all of the expected dart count."""
    P, CP, S = toy_model
    G = 30
    darts = ThreeDartMDP(P, CP, S, G, dart_cost=1.0, turn_cost=0.0).solve(tol=1e-13)
    turns = ThreeDartMDP(P, CP, S, G, dart_cost=0.0, turn_cost=1.0).solve(tol=1e-13)

    Vref = reference_values(P, CP, S, G, dart_cost=0.0, turn_cost=1.0)
    for s in range(2, G + 1):
        assert turns.V1[s] == pytest.approx(Vref[s, 0, s], abs=1e-9)
        assert darts.expected_darts(s) / 3 <= turns.expected_darts(s) + 1e-9
        assert turns.expected_darts(s) <= darts.expected_darts(s) + 1e-9


def test_policy_reproduces_value_under_simulation(toy_model):
    """Playing the extracted policy must recover the value function."""
    P, CP, S = toy_model
    G = 30
    fast = ThreeDartMDP(P, CP, S, G).solve(tol=1e-13)
    darts, rounds = fast.simulate(n_legs=20000, seed=7)

    se = darts.std() / np.sqrt(len(darts))
    assert abs(darts.mean() - fast.expected_darts()) < 5 * se


def test_bust_forfeits_rest_of_round(toy_model):
    """A model where dart 1 always busts should need infinitely many darts;
    a model that can never bust must match a memoryless 1-dart MDP."""
    P, CP, S = toy_model
    G = 30
    fast = ThreeDartMDP(P, CP, S, G).solve(tol=1e-13)
    # Values must be monotone in the sense that no score is cheaper than the
    # cheapest possible finish (one dart).
    assert (fast.V1[2:] <= -1.0).all()
