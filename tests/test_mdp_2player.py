"""
Tests for the two-player zero-sum leg solvers.

Both solvers are checked against a brute-force value iteration written straight
from the game definition, with no ordering or caching assumptions.
"""

import numpy as np
import pytest

from darts.mdp_2player import OneDartLeg, ThreeDartLeg


@pytest.fixture(scope="module")
def toy_model():
    rng = np.random.default_rng(1)
    scores = np.array([0, 1, 2, 3, 4, 5, 6, 8, 10, 12], dtype=np.int32)
    P = rng.random((10, len(scores))) ** 2 + 1e-2
    P /= P.sum(axis=1, keepdims=True)
    CP = P * ((scores % 2 == 0) & (scores > 0)) * rng.random((10, len(scores)))
    return P, CP, scores


def brute_one_dart(P, CP, S, G, iters=100000, tol=1e-14):
    """W[u, v] by value iteration, one dart per turn. Updates in place
    (Gauss-Seidel), which converges in a handful of sweeps."""
    W = np.zeros((G + 1, G + 1))
    for _ in range(iters):
        old = W.copy()
        for u in range(2, G + 1):
            for v in range(2, G + 1):
                q = np.zeros(P.shape[0])
                for j, sj in enumerate(S):
                    if sj <= u - 2:
                        q += P[:, j] * (1.0 - W[v, u - sj])
                    elif sj == u:
                        q += CP[:, j] * 1.0
                        q += (P[:, j] - CP[:, j]) * (1.0 - W[v, u])
                    else:
                        q += P[:, j] * (1.0 - W[v, u])
                W[u, v] = q.max()
        if np.abs(W - old).max() < tol:
            break
    return W


def brute_three_dart(P, CP, S, G, iters=100000, tol=1e-14):
    """W[u, v] and the within-turn values, three darts per turn, updated in
    place. V is indexed [dart, score, turn start, opponent]."""
    n = P.shape[0]
    W = np.zeros((G + 1, G + 1))
    V = np.zeros((4, G + 1, G + 1, G + 1))
    for _ in range(iters):
        old = W.copy()
        for u in range(2, G + 1):
            for v in range(2, G + 1):
                for dart in (3, 2, 1):
                    for w in range(2, u + 1):
                        q = np.zeros(n)
                        for j, sj in enumerate(S):
                            if sj <= w - 2:
                                nw = w - sj
                                nxt = (
                                    1.0 - W[v, nw] if dart == 3 else V[dart + 1, nw, u, v]
                                )
                                q += P[:, j] * nxt
                            elif sj == w:
                                q += CP[:, j] * 1.0
                                q += (P[:, j] - CP[:, j]) * (1.0 - W[v, u])
                            else:
                                q += P[:, j] * (1.0 - W[v, u])
                        V[dart, w, u, v] = q.max()
                W[u, v] = V[1, u, u, v]
        if np.abs(W - old).max() < tol:
            break
    return W, V


@pytest.mark.parametrize("G", [12, 20])
def test_one_dart_leg_matches_brute_force(toy_model, G):
    P, CP, S = toy_model
    ref = brute_one_dart(P, CP, S, G)
    leg = OneDartLeg(P, CP, S, G).solve()
    for u in range(2, G + 1):
        for v in range(2, G + 1):
            assert leg.W[u, v] == pytest.approx(ref[u, v], abs=1e-9)


@pytest.mark.parametrize("G", [10, 16])
def test_three_dart_leg_matches_brute_force(toy_model, G):
    P, CP, S = toy_model
    ref, Vref = brute_three_dart(P, CP, S, G)
    leg = ThreeDartLeg(P, CP, S, G).solve()
    for u in range(2, G + 1):
        for v in range(2, G + 1):
            assert leg.W[u, v] == pytest.approx(ref[u, v], abs=1e-7)

    # Cached within-turn tables must match every turn-start slice they claim to
    # be independent of.
    for v in range(2, G + 1):
        for w in range(2, G + 1):
            if w >= leg.u3_indep:
                for u in range(w, G + 1):
                    assert leg.Y3[v, w] == pytest.approx(Vref[3, w, u, v], abs=1e-7)
            if w >= leg.u2_indep:
                for u in range(w, G + 1):
                    assert leg.Y2[v, w] == pytest.approx(Vref[2, w, u, v], abs=1e-7)


def test_win_probabilities_are_consistent(toy_model):
    P, CP, S = toy_model
    G = 24
    leg = ThreeDartLeg(P, CP, S, G).solve()
    sub = leg.W[2:, 2:]
    assert (sub >= 0).all() and (sub <= 1).all()

    # Throwing first is an advantage: from an identical position the player at
    # the oche has every option the other has, plus the chance to finish first.
    for u in range(2, G + 1):
        assert leg.W[u, u] > 0.5

    # Winning is at least as likely as checking out with the very next dart.
    for u in range(2, G + 1):
        immediate = CP[:, S == u].max() if (S == u).any() else 0.0
        assert leg.W[u, 2] >= immediate - 1e-9

    # Note: W is *not* monotone in either score. Odd scores cost an extra dart,
    # so being on 3 can be worse than being on 4.


def test_certain_checkout_wins_immediately():
    """If some aiming point always checks out, the player to throw always wins."""
    scores = np.array([0, 2, 4], dtype=np.int32)
    P = np.array([[0.5, 0.3, 0.2], [0.0, 1.0, 0.0]])
    CP = np.array([[0.0, 0.1, 0.1], [0.0, 1.0, 0.0]])
    for cls in (OneDartLeg, ThreeDartLeg):
        leg = cls(P, CP, scores, 2).solve()
        assert leg.W[2, 2] == pytest.approx(1.0)
        assert leg.policy[2, 2] == 1
