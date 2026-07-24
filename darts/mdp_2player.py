"""
Two-player (zero-sum) darts legs.

Both players want to check out first, so a leg is a turn-based zero-sum
stochastic game with perfect information. Only the player at the oche acts, so
the game has a value in pure strategies and can be solved by backward induction
rather than by solving matrix games at each state.

State and value
---------------
``W[u, v]`` is the probability that the player *about to throw*, on score ``u``,
wins the leg against an opponent on score ``v``. When my turn ends leaving me on
``u'``, my opponent throws next, so my chance from there is ``1 - W[v, u']``.
Checking out wins immediately.

Why this is tractable
---------------------
``W[u, v]`` depends on ``W[v, u']`` for ``u' <= u``, so ordering states by the
*total* ``u + v`` makes almost every dependency point strictly backwards. The
one exception is ``u' == u`` -- a turn that makes no progress -- which couples
``W[u, v]`` to ``W[v, u]``, its mirror image on the same diagonal. Each diagonal
is therefore a small fixed point among the pairs on it, contracting at the rate
of the "wasted turn" probability, which is tiny except at very low scores.

That ordering also makes each diagonal a single matrix multiply: every pair on
it applies the same ``(n_points x n_scores)`` transition matrix to its own
continuation vector.

Two games are provided:

``OneDartLeg``   -- one dart per turn, the two-player version of the memoryless
                    MDP in ``darts/mdp.py``.
``ThreeDartLeg`` -- three darts per turn with real bust rules, the two-player
                    version of ``darts/mdp_3turn.py``.

Cost
----
``OneDartLeg`` is one GEMM per diagonal and solves a full 501 leg in about a
minute. ``ThreeDartLeg`` is three GEMMs per diagonal for turns that cannot bust,
but every pair whose score is below ``3 * max_dart + 2`` needs its own within-turn
sweep, and there are ``O(180 * game_start)`` of those. Reducing the aiming grid
(see :func:`candidate_points`) is the practical way to make the full game
affordable.
"""

import numpy as np


class _LegBase:
    """Shared transition bookkeeping for the two-player solvers."""

    def __init__(self, probs, checkout_probs, allowed_scores, game_start):
        order = np.argsort(allowed_scores)
        self.scores = np.ascontiguousarray(allowed_scores[order]).astype(np.int64)
        self.P = np.ascontiguousarray(probs[:, order], dtype=np.float64)
        self.CP = np.ascontiguousarray(checkout_probs[:, order], dtype=np.float64)
        self.n_points, self.n_scores = self.P.shape
        self.game_start = int(game_start)
        self.max_dart = int(self.scores[-1])

        self.PT = np.ascontiguousarray(self.P.T)
        self.P0 = self.PT[0].copy() if self.scores[0] == 0 else np.zeros(self.n_points)

        G = self.game_start
        cum = np.cumsum(self.PT, axis=0)
        tail = np.vstack([np.ones((1, self.n_points)), 1.0 - cum])
        self.k_valid = np.array(
            [np.searchsorted(self.scores, u - 2, side="right") for u in range(G + 1)]
        )
        # co[u]   : probability of checking out from u with one dart
        # bust[u] : probability of busting from u with one dart
        self.co = np.zeros((G + 1, self.n_points))
        for j, s in enumerate(self.scores):
            if s <= G:
                self.co[s] = self.CP[:, j]
        self.bust = np.stack([tail[self.k_valid[u]] for u in range(G + 1)]) - self.co

    def _diagonals(self):
        """Yield (u_array, v_array) for each total score, in solvable order.

        Pair ``k`` on a diagonal is the mirror of pair ``-1-k``, which is what
        makes the within-diagonal coupling a simple reversal.
        """
        G = self.game_start
        for total in range(4, 2 * G + 1):
            lo, hi = max(2, total - G), min(G, total - 2)
            if lo > hi:
                continue
            u = np.arange(lo, hi + 1)
            yield u, total - u

    @staticmethod
    def _fixed_point(step, n_pair, tol, max_iter):
        """
        Solve ``x = step(x)`` on a diagonal.

        The map contracts at the rate of the "wasted turn" probability, which is
        negligible at high scores but close to 1 at low ones, so plain iteration
        is far too slow on the diagonals that contain a low score. Aitken
        extrapolation is applied componentwise and only kept where it does not
        move further from the fixed point than plain iteration would, which
        makes it safe even though the components are coupled.
        """
        x = np.zeros(n_pair)
        for _ in range(max_iter):
            fx = step(x)
            r = fx - x
            if np.abs(r).max() < tol:
                return fx
            f2x = step(fx)
            denom = (f2x - fx) - r
            with np.errstate(divide="ignore", invalid="ignore"):
                x_acc = np.where(np.abs(denom) > 1e-300, x - r * r / denom, f2x)
            x_acc = np.where(np.isfinite(x_acc), x_acc, f2x)
            keep = np.abs(step(x_acc) - x_acc) <= np.abs(f2x - fx)
            x = np.where(keep, x_acc, f2x)
        return x

    def _handover(self, us, vs, table):
        """
        ``(n_scores, n_pair)`` matrix of continuation values for a legal,
        score-reducing dart: entry ``[j, k]`` is the value of pair ``k`` after a
        dart scoring ``scores[j]``. ``table`` maps (opponent score, my new
        score) to a value.
        """
        S = self.scores
        out = np.zeros((self.n_scores, len(us)))
        for k, (u, v) in enumerate(zip(us, vs)):
            kk = self.k_valid[u]
            out[1:kk, k] = table[v, u - S[1:kk]]
        return out


class OneDartLeg(_LegBase):
    """
    Zero-sum leg with one dart per turn.

    After :meth:`solve`:
        W (np.ndarray): ``W[u, v]`` = probability the player on ``u``, about to
            throw, wins against an opponent on ``v``.
        policy (np.ndarray): index of the optimal aiming point for each state.
    """

    def solve(self, tol=1e-13, max_iter=500, progress=False):
        G = self.game_start
        self.W = np.zeros((G + 1, G + 1))
        self.policy = np.full((G + 1, G + 1), -1, dtype=np.int32)

        diagonals = list(self._diagonals())
        if progress:
            from tqdm import tqdm

            diagonals = tqdm(diagonals, desc="1-dart leg")

        for us, vs in diagonals:
            A = self.PT.T @ self._handover(us, vs, 1.0 - self.W) + self.co[us].T
            # A dart that scores zero wastes the turn exactly like a bust does,
            # because a turn is a single dart here.
            N = self.bust[us].T + self.P0[:, None]

            x = self._fixed_point(
                lambda z: (A + N * (1.0 - z[::-1])).max(axis=0), len(us), tol, max_iter
            )
            q = A + N * (1.0 - x[::-1])
            self.W[us, vs] = q.max(axis=0)
            self.policy[us, vs] = q.argmax(axis=0)
        return self


class ThreeDartLeg(_LegBase):
    """
    Zero-sum leg with three darts per turn and real bust rules.

    A bust returns the score to the value it had at the start of the turn *and*
    forfeits the remaining darts, so within a turn the value depends on the
    score the turn started from. As in the single-player model that dependence
    disappears once no bust is reachable, which is what ``Y3`` and ``Y2`` cache.

    After :meth:`solve`:
        W (np.ndarray): start-of-turn win probabilities.
        policy (np.ndarray): optimal first dart of the turn for each state.
        Y3, Y2 (np.ndarray): ``[opponent score, my score]`` values with one and
            two darts of the turn remaining, valid at or above ``u3_indep`` and
            ``u2_indep`` respectively.
    """

    def __init__(self, probs, checkout_probs, allowed_scores, game_start):
        super().__init__(probs, checkout_probs, allowed_scores, game_start)
        M = self.max_dart
        self.u3_indep = M + 2
        self.u2_indep = self.u3_indep + M
        self.s_indep = self.u2_indep + M

    def solve(self, tol=1e-13, max_iter=500, progress=False):
        G = self.game_start
        self.W = np.zeros((G + 1, G + 1))
        self.policy = np.full((G + 1, G + 1), -1, dtype=np.int32)
        self.Y3 = np.full((G + 1, G + 1), np.nan)
        self.Y2 = np.full((G + 1, G + 1), np.nan)

        diagonals = list(self._diagonals())
        if progress:
            from tqdm import tqdm

            diagonals = tqdm(diagonals, desc="3-dart leg")

        for us, vs in diagonals:
            co = self.co[us].T
            bust = self.bust[us].T
            # Dart 3 hands over to the opponent; darts 1 and 2 stay in the turn.
            A3 = self.PT.T @ self._handover(us, vs, 1.0 - self.W) + co
            A2 = self.PT.T @ self._handover(us, vs, self.Y3) + co
            A1 = self.PT.T @ self._handover(us, vs, self.Y2) + co
            low = np.nonzero(us < self.s_indep)[0]
            high = np.nonzero(us >= self.s_indep)[0]

            n_pair = len(us)
            y3 = np.zeros(n_pair)
            y2 = np.zeros(n_pair)
            q1 = np.zeros((self.n_points, n_pair))

            def sweep(x):
                # value of my turn ending back on the score it started from
                e = 1.0 - x[::-1]
                if len(high):
                    h = high
                    y3[h] = (A3[:, h] + (bust[:, h] + self.P0[:, None]) * e[h]).max(0)
                    y2[h] = (
                        A2[:, h] + self.P0[:, None] * y3[h] + bust[:, h] * e[h]
                    ).max(0)
                    q1[:, h] = (
                        A1[:, h] + self.P0[:, None] * y2[h] + bust[:, h] * e[h]
                    )
                for k in low:
                    y3[k], y2[k], q1[:, k] = self._low_turn(us[k], vs[k], e[k])
                return q1.max(axis=0)

            x = self._fixed_point(sweep, n_pair, tol, max_iter)
            x = sweep(x)

            self.W[us, vs] = x
            self.policy[us, vs] = q1.argmax(axis=0)
            self.Y3[vs, us] = y3
            self.Y2[vs, us] = y2
        return self

    def _low_turn(self, u, v, e):
        """
        Full within-turn sweep for a turn starting on ``u`` against an opponent
        on ``v``, given that ending the turn back on ``u`` is worth ``e``.

        Used for turns from which a bust is reachable, where the cached
        turn-start-independent tables do not apply. Both dart-3 and dart-2
        backups are done as a single matrix product over all the scores the turn
        could reach.

        Returns:
            tuple: dart-3 value at ``u``, dart-2 value at ``u``, and the
            per-aiming-point values of the turn's first dart.
        """
        S, M, PT = self.scores, self.max_dart, self.PT

        # ---- dart 3, for every score this turn could be on -----------------
        lo3 = max(2, u - 2 * M)
        v3 = np.empty(u - lo3 + 1)
        need = [w for w in range(lo3, u + 1) if w < self.u3_indep or w == u]
        for w in range(lo3, u):
            if w >= self.u3_indep:
                v3[w - lo3] = self.Y3[v, w]
        if need:
            Gm = np.zeros((self.n_scores, len(need)))
            for m, w in enumerate(need):
                kk = self.k_valid[w]
                Gm[1:kk, m] = 1.0 - self.W[v, w - S[1:kk]]
            Q = PT.T @ Gm + self.co[need].T + self.bust[need].T * e
            for m, w in enumerate(need):
                # A dart scoring zero also ends the turn, on w rather than on u.
                zero_val = e if w == u else 1.0 - self.W[v, w]
                v3[w - lo3] = (Q[:, m] + self.P0 * zero_val).max()

        # ---- dart 2 --------------------------------------------------------
        lo2 = max(2, u - M)
        v2 = np.empty(u - lo2 + 1)
        need2 = [w for w in range(lo2, u + 1) if w < self.u2_indep or w == u]
        for w in range(lo2, u):
            if w >= self.u2_indep:
                v2[w - lo2] = self.Y2[v, w]
        if need2:
            Gm = np.zeros((self.n_scores, len(need2)))
            for m, w in enumerate(need2):
                kk = self.k_valid[w]
                Gm[1:kk, m] = v3[w - S[1:kk] - lo3]
            Q = PT.T @ Gm + self.co[need2].T + self.bust[need2].T * e
            for m, w in enumerate(need2):
                v2[w - lo2] = (Q[:, m] + self.P0 * v3[w - lo3]).max()

        # ---- dart 1 --------------------------------------------------------
        kk = self.k_valid[u]
        g = np.zeros(self.n_scores)
        g[1:kk] = v2[u - S[1:kk] - lo2]
        q1 = PT.T @ g + self.co[u] + self.bust[u] * e + self.P0 * v2[u - lo2]
        return v3[u - lo3], v2[u - lo2], q1


def candidate_points(probs, checkout_probs, allowed_scores, sigmas_models=(),
                     game_start=170):
    """
    A reduced set of aiming points for the expensive two-player solves.

    Every Q-value in these games is a linear functional of a point's score
    distribution, so only points on the convex hull of those distributions can
    ever be optimal. Enumerating that hull exactly in 40-odd dimensions is
    expensive; this instead takes the union of the points that are actually
    optimal somewhere in a family of cheap single-player problems, which covers
    the same kinds of trade-off (scoring, setting up, going at a double,
    protecting a number).

    Args:
        probs, checkout_probs, allowed_scores: the full transition model.
        sigmas_models (iterable): solved ``ThreeDartMDP`` instances built on the
            same aiming grid, whose policies seed the candidate set.
        game_start (int): score to solve the seeding problems up to.

    Returns:
        np.ndarray: sorted indices of the retained aiming points.
    """
    from darts.mdp_3turn import ThreeDartMDP

    keep = set()
    models = list(sigmas_models)
    if not models:
        models = [
            ThreeDartMDP(probs, checkout_probs, allowed_scores, game_start,
                         dart_cost=dc, turn_cost=tc).solve()
            for dc, tc in [(1.0, 0.0), (0.0, 1.0)]
        ]
    for m in models:
        keep.update(int(i) for i in np.unique(m.pol1[2:]) if i >= 0)
        keep.update(int(i) for i in np.unique(m.pol2tab) if i >= 0)
        keep.update(int(i) for i in np.unique(m.pol3tab) if i >= 0)
        keep.update(int(i) for i in np.unique(m.pol2low) if i >= 0)
        keep.update(int(i) for i in np.unique(m.pol3low) if i >= 0)
    # Plus the best point for each individual outcome: maximum probability of
    # each score, and of each checkout.
    keep.update(int(i) for i in probs.argmax(axis=0))
    keep.update(int(i) for i in checkout_probs.argmax(axis=0))
    return np.array(sorted(keep))
