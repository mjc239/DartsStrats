"""
Legs between players of *different* abilities.

``darts/mdp_2player.py`` assumes both players throw the same way, which makes
the game symmetric: the value of the mirror position ``(v, u)`` is the same
function as the value of ``(u, v)``. That is the case worth optimising, because
it answers "how much is throwing first worth?", but it cannot answer "how much
better does the underdog have to be to give away the throw?".

Here each player has their own transition model, so two value functions are
carried:

    ``W_ab[u, v]`` -- A is on ``u`` and about to throw, B is on ``v``
    ``W_ba[v, u]`` -- B is on ``v`` and about to throw, A is on ``u``

If A's turn ends leaving A on ``u'``, B throws next, so A's chance is
``1 - W_ba[v, u']``. The diagonal ordering by ``u + v`` still works, and the
within-diagonal coupling is now index-aligned rather than reversed: A's pair
``k`` couples to B's pair ``k``, which is the same physical position with the
other player at the oche.

Only the one-dart-per-turn game is implemented. The three-dart version is the
same construction with the within-turn machinery of ``ThreeDartLeg`` doubled up,
but it costs ~15 minutes per *ordered pair* of abilities, so a grid of them is
an overnight job rather than something to solve on demand.
"""

import numpy as np

from darts.mdp_2player import _LegBase


class AsymmetricOneDartLeg:
    """
    Zero-sum leg between two different players, one dart per turn.

    Args:
        model_a: ``(probs, checkout_probs)`` for player A.
        model_b: ``(probs, checkout_probs)`` for player B. The two may have
            different numbers of aiming points but must share ``allowed_scores``.
        allowed_scores (np.ndarray): the board's distinct scores.
        game_start (int): starting score.

    After :meth:`solve`:
        W_ab, W_ba (np.ndarray): as described in the module docstring.
        p_first (float): P(A wins | both on game_start, A throws first).
        p_second (float): P(A wins | both on game_start, B throws first).
    """

    def __init__(self, model_a, model_b, allowed_scores, game_start):
        self.a = _LegBase(model_a[0], model_a[1], allowed_scores, game_start)
        self.b = _LegBase(model_b[0], model_b[1], allowed_scores, game_start)
        self.scores = self.a.scores
        self.game_start = int(game_start)

    def solve(self, tol=1e-13, max_iter=500, progress=False):
        G = self.game_start
        S = self.scores
        self.W_ab = np.zeros((G + 1, G + 1))
        self.W_ba = np.zeros((G + 1, G + 1))
        self.policy_ab = np.full((G + 1, G + 1), -1, dtype=np.int32)
        self.policy_ba = np.full((G + 1, G + 1), -1, dtype=np.int32)

        diagonals = list(self.a._diagonals())
        if progress:
            from tqdm import tqdm

            diagonals = tqdm(diagonals, desc="asymmetric leg")

        for us, vs in diagonals:
            # A to throw on us[k] against vs[k]; B to throw on vs[k] against us[k].
            A = self.a.PT.T @ self.a._handover(us, vs, 1.0 - self.W_ba) \
                + self.a.co[us].T
            B = self.b.PT.T @ self.b._handover(vs, us, 1.0 - self.W_ab) \
                + self.b.co[vs].T
            NA = self.a.bust[us].T + self.a.P0[:, None]
            NB = self.b.bust[vs].T + self.b.P0[:, None]

            n = len(us)
            xa, xb = np.zeros(n), np.zeros(n)
            for _ in range(max_iter):
                # A's wasted turn hands over to B in the same position, and
                # vice versa, so the two sides are solved together.
                na = (A + NA * (1.0 - xb)).max(axis=0)
                nb = (B + NB * (1.0 - xa)).max(axis=0)
                if max(np.abs(na - xa).max(), np.abs(nb - xb).max()) < tol:
                    xa, xb = na, nb
                    break
                xa, xb = na, nb

            qa = A + NA * (1.0 - xb)
            qb = B + NB * (1.0 - xa)
            self.W_ab[us, vs] = qa.max(axis=0)
            self.W_ba[vs, us] = qb.max(axis=0)
            self.policy_ab[us, vs] = qa.argmax(axis=0)
            self.policy_ba[vs, us] = qb.argmax(axis=0)

        self.p_first = float(self.W_ab[G, G])
        self.p_second = float(1.0 - self.W_ba[G, G])
        return self


def leg_probabilities(model_a, model_b, allowed_scores, game_start=501, **kw):
    """
    The two numbers a match model needs.

    Returns:
        tuple[float, float]: ``(p_first, p_second)`` -- A's probability of
        winning a leg when A throws first, and when B throws first.
    """
    leg = AsymmetricOneDartLeg(model_a, model_b, allowed_scores, game_start).solve(**kw)
    return leg.p_first, leg.p_second
