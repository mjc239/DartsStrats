"""
3-dart (per-turn) MDP for single-player 501.

Extends the published single-player MDP by tracking which dart within a round
is being thrown. This correctly models bust recovery: a bust on dart 1 or 2
resets the score to the round's starting value and forfeits the remaining darts
in that round, just as in real 501.

State: (score, dart, round_start)
  score       -- remaining score (0 means game over)
  dart        -- dart number within the current round (1, 2, or 3)
  round_start -- score at the start of the current round (the bust target)

Valid states satisfy round_start >= score, and at dart=1, score == round_start
(no dart has been thrown yet this round).

Value convention (matches darts/mdp.py): values are negative, e.g.
``V1[501] == -12.4`` means 12.4 expected darts to check out. Terminal V(0) = 0.

Objective
---------
The cost of a transition is ``dart_cost`` per dart thrown plus ``turn_cost`` per
round started:

* ``dart_cost=1, turn_cost=0`` (default) minimises the expected number of
  *darts thrown*. This matches darts/mdp.py. Note that a bust on dart 1 costs
  only the single dart that was thrown, because the rest of the round is
  forfeited and never thrown.
* ``dart_cost=0, turn_cost=1`` minimises the expected number of *rounds*, which
  is the quantity that matters in a race against an opponent -- an early bust
  then costs a whole visit, not one dart.

The two objectives give slightly different optimal policies around bust-risk
trade-offs, so both are supported.

Solvers
-------
``ThreeDartMDP``  -- fast structural solver (near-linear in ``game_start``),
                    returns values *and* the policy.
``_compute_3turn_values`` / ``compute_3turn_values`` -- the original
                    Numba reference implementation, kept for validation.

Structure exploited by the fast solver
--------------------------------------
Let ``M`` be the highest score a single dart can make (60 on a standard board).

* A dart thrown at a score ``u >= M + 2`` can never bust, so
  ``V(u, 3, s)`` does not depend on ``s`` at all -- dart 3 ends the round.
* Consequently ``V(u, 2, s)`` is independent of ``s`` once ``u >= 2M + 2``,
  and the whole round from ``s >= 3M + 2`` is independent of ``s`` except
  through the (tiny) probability of missing the board entirely.

So instead of solving ``O(game_start^2)`` states, the solver keeps
``s``-independent tables for dart-2 and dart-3 values and only carries a
per-round-start state set for the ~181 low scores where a bust is reachable.
Within a round the only circular dependency is the scalar ``V(s, 1, s)`` that
busts return to; that scalar fixed point is solved with Aitken-accelerated
iteration rather than by sweeping the whole state space.
"""

import numpy as np
from numba import njit


# --------------------------------------------------------------------------
# Fast structural solver
# --------------------------------------------------------------------------


class ThreeDartMDP:
    """
    Solver for the 3-dart single-player MDP.

    Args:
        probs (np.ndarray): (n_points, n_scores) probability of each board score
            for each aiming point.
        checkout_probs (np.ndarray): (n_points, n_scores) probability of hitting
            each score *in a checkout region* (double or inner bull).
        allowed_scores (np.ndarray): (n_scores,) sorted array of board scores.
        game_start (int): highest score to solve for (e.g. 501).
        dart_cost (float): cost charged per dart thrown.
        turn_cost (float): cost charged per round started.

    After calling :meth:`solve`, values and the optimal policy are available
    through :meth:`value` / :meth:`policy`, with ``V1`` holding the
    start-of-round values indexed by score.
    """

    def __init__(
        self,
        probs,
        checkout_probs,
        allowed_scores,
        game_start,
        dart_cost=1.0,
        turn_cost=0.0,
    ):
        order = np.argsort(allowed_scores)
        self.scores = np.ascontiguousarray(allowed_scores[order]).astype(np.int64)
        self.P = np.ascontiguousarray(probs[:, order], dtype=np.float64)
        self.CP = np.ascontiguousarray(checkout_probs[:, order], dtype=np.float64)

        self.n_points, self.n_scores = self.P.shape
        self.game_start = int(game_start)
        self.dart_cost = float(dart_cost)
        self.turn_cost = float(turn_cost)

        # (n_scores, n_points) layout: rows are contiguous per board score, and
        # the reductions below run over the (contiguous) aiming-point axis.
        self.PT = np.ascontiguousarray(self.P.T)
        self.CPT = np.ascontiguousarray(self.CP.T)
        self.P0 = self.PT[0].copy() if self.scores[0] == 0 else np.zeros(self.n_points)

        # tail[k] = sum of probabilities of board scores with index >= k
        cum = np.cumsum(self.PT, axis=0)
        self.tail = np.ascontiguousarray(
            np.vstack([np.ones((1, self.n_points)), 1.0 - cum])
        )

        self.max_dart = int(self.scores[-1])
        # Thresholds beyond which a state's value no longer depends on round_start.
        self.u3_indep = self.max_dart + 2
        self.u2_indep = self.u3_indep + self.max_dart
        self.s_indep = self.u2_indep + self.max_dart

        self._solved = False

    # -- small helpers ----------------------------------------------------

    def _n_valid(self, u):
        """Number of board scores that leave a legal (>= 2) remainder from u."""
        return int(np.searchsorted(self.scores, u - 2, side="right"))

    def _checkout_row(self, u):
        """Probability of checking out with a single dart from score u."""
        idx = np.searchsorted(self.scores, u)
        if idx < self.n_scores and self.scores[idx] == u:
            return self.CPT[idx]
        return None

    def _bust_row(self, u):
        """Per-aiming-point probability that a dart from score u busts."""
        b = self.tail[self._n_valid(u)].copy()
        cp = self._checkout_row(u)
        if cp is not None:
            b -= cp
        return b

    # -- main solve -------------------------------------------------------

    def solve(self, tol=1e-11, max_iter=500, progress=False):
        """
        Solve the MDP.

        Args:
            tol (float): convergence tolerance on the scalar round-start value.
            max_iter (int): safety cap on fixed-point iterations per round start.
            progress (bool): show a tqdm progress bar.

        Returns:
            ThreeDartMDP: self, so calls can be chained.
        """
        G = self.game_start
        M = self.max_dart
        n_pts = self.n_points
        S = self.scores
        PT = self.PT
        dc, tc = self.dart_cost, self.turn_cost

        # V1[s]  = V(s, 1, s), the start-of-round value.
        # V3tab / V2tab hold the round-start-independent dart-3 / dart-2 values.
        # For low scores the value does depend on round_start, so those are
        # recomputed inside each round-start group and never cached.
        self.V1 = np.zeros(G + 1)
        self.V3tab = np.full(G + 1, np.nan)
        self.V2tab = np.full(G + 1, np.nan)
        self.pol1 = np.full(G + 1, -1, dtype=np.int32)
        self.pol3tab = np.full(G + 1, -1, dtype=np.int32)
        self.pol2tab = np.full(G + 1, -1, dtype=np.int32)

        # Round-start-dependent policies, only defined for the low-score region.
        s_low = min(G, self.s_indep - 1)
        self.pol3low = np.full((s_low + 1, min(G, self.u3_indep - 1) + 1), -1, np.int32)
        self.pol2low = np.full((s_low + 1, min(G, self.u2_indep - 1) + 1), -1, np.int32)
        self.s_low = s_low

        # A3[u] = sum_j P[:, j] * V1[u - S[j]] over board scores that leave a
        # legal remainder, *including* the S[j] == 0 term. Only the low-score
        # rows have to be kept: above u3_indep the dart-3 value is a scalar.
        n_keep = min(G, self.u3_indep - 1) + 1
        A3 = np.zeros((n_keep, n_pts))
        bust_low = np.zeros((n_keep, n_pts))
        for u in range(2, n_keep):
            bust_low[u] = self._bust_row(u)

        self.n_iters = np.zeros(G + 1, dtype=np.int32)

        rng = range(2, G + 1)
        if progress:
            from tqdm import tqdm

            rng = tqdm(rng)

        for s in rng:
            k_s = self._n_valid(s)
            lo3 = max(2, s - 2 * M)
            lo2 = max(2, s - M)

            # ---- dart-3 pieces -------------------------------------------
            # Constant part: scores in [lo3, s-1] at or above u3_indep have a
            # round-start-independent value that was cached when they were the
            # round start themselves.
            v3 = np.empty(s - lo3 + 1)
            const3_lo = max(lo3, self.u3_indep)
            if const3_lo <= s - 1:
                v3[const3_lo - lo3 : s - lo3] = self.V3tab[const3_lo:s]

            # Variable part: scores below u3_indep, where a bust is reachable
            # and the value therefore depends on x = V(s, 1, s).
            hi3 = min(s - 1, self.u3_indep - 1)
            base3 = A3[lo3 : hi3 + 1]
            coef3 = bust_low[lo3 : hi3 + 1]
            n_var3 = max(0, hi3 - lo3 + 1)

            # State (s, 3, s): reached when darts 1 and 2 both missed the board.
            # A dart scoring 0 here leaves the score at s and ends the round, so
            # the miss probability joins the bust probability in the x term.
            base3_s = PT[1:k_s].T @ self.V1[s - S[1:k_s]]
            coef3_s = self._bust_row(s) + self.P0

            # ---- dart-2 pieces -------------------------------------------
            v2 = np.empty(s - lo2 + 1)
            const2_lo = max(lo2, self.u2_indep)
            if const2_lo <= s - 1:
                v2[const2_lo - lo2 : s - lo2] = self.V2tab[const2_lo:s]

            var2 = list(range(lo2, min(s - 1, self.u2_indep - 1) + 1)) + [s]
            var2_idx = np.array([u - lo2 for u in var2], dtype=np.int64)
            k2 = [self._n_valid(u) for u in var2]
            bust2 = np.stack([self._bust_row(u) for u in var2])
            W2 = np.zeros((self.n_scores, len(var2)))

            # ---- dart-1 pieces -------------------------------------------
            bust1 = self._bust_row(s)
            W1 = np.zeros(self.n_scores)

            def sweep(x, record=False):
                """One backup of the whole round, given bust value x; returns V(s,1,s)."""
                if n_var3:
                    q3 = base3 + x * coef3
                    v3[: n_var3] = q3.max(axis=1) - dc
                q3s = base3_s + x * coef3_s
                v3[s - lo3] = q3s.max() - dc

                W2[:] = 0.0
                for m, u in enumerate(var2):
                    kk = k2[m]
                    W2[:kk, m] = v3[u - S[:kk] - lo3]
                q2 = W2.T @ PT + x * bust2
                v2[var2_idx] = q2.max(axis=1) - dc

                W1[:] = 0.0
                W1[:k_s] = v2[s - S[:k_s] - lo2]
                q1 = W1 @ PT + x * bust1

                if record:
                    if n_var3:
                        a3 = q3.argmax(axis=1)
                        self.pol3low[s, lo3 : hi3 + 1] = a3
                    if s >= self.u3_indep:
                        self.V3tab[s] = v3[s - lo3]
                        self.pol3tab[s] = q3s.argmax()
                    else:
                        self.pol3low[s, s] = q3s.argmax()

                    a2 = q2.argmax(axis=1)
                    for m, u in enumerate(var2):
                        if u == s and s >= self.u2_indep:
                            self.V2tab[s] = v2[u - lo2]
                            self.pol2tab[s] = a2[m]
                        else:
                            self.pol2low[s, u] = a2[m]

                    self.pol1[s] = q1.argmax()

                return q1.max() - dc - tc

            # ---- scalar fixed point on x = V(s, 1, s) --------------------
            # Every state in the round is a monotone, piecewise-linear, convex
            # function of x with slope < 1, so plain iteration converges at the
            # rate of the round's bust probability. Aitken extrapolation makes
            # each step exact for a fixed policy (it is policy iteration on the
            # scalar), which matters for the low scores where busts are common.
            x = self.V1[s - 1] if s >= 3 else -1.0
            n_eval = 0
            for it in range(max_iter):
                fx = sweep(x)
                n_eval += 1
                r = fx - x
                if abs(r) < tol:
                    x = fx
                    break
                f2x = sweep(fx)
                n_eval += 1
                denom = (f2x - fx) - r
                if abs(denom) > 1e-300:
                    x_acc = x - r * r / denom
                    # Safeguard: only accept the extrapolation if it does not
                    # move further from the fixed point than plain iteration.
                    if np.isfinite(x_acc) and abs(sweep(x_acc) - x_acc) <= abs(f2x - fx):
                        n_eval += 1
                        x = x_acc
                        continue
                    n_eval += 1
                x = f2x
            self.n_iters[s] = n_eval

            # Final sweep at the fixed point, recording values and argmax actions.
            self.V1[s] = sweep(x, record=True)

            # Cache the dart-3 row for later round starts, now that V1[s] is known.
            if s < n_keep:
                A3[s] = base3_s + self.P0 * self.V1[s]

        self._solved = True
        return self

    # -- accessors --------------------------------------------------------

    def value(self, score, dart, round_start):
        """Value of state (score, dart, round_start); negative = expected cost."""
        if score == 0:
            return 0.0
        if dart == 1:
            return self.V1[score]
        if dart == 3 and score >= self.u3_indep:
            return self.V3tab[score]
        if dart == 2 and score >= self.u2_indep:
            return self.V2tab[score]
        raise KeyError(
            "low-score values depend on round_start and are not cached; "
            "re-solve with keep_low_values=True or use policy() instead"
        )

    def policy(self, score, dart, round_start):
        """Index of the optimal aiming point for state (score, dart, round_start)."""
        if dart == 1:
            return int(self.pol1[score])
        if dart == 3:
            if score >= self.u3_indep:
                return int(self.pol3tab[score])
            return int(self.pol3low[round_start, score])
        if dart == 2:
            if score >= self.u2_indep and score != round_start:
                return int(self.pol2tab[score])
            if score >= self.u2_indep and score == round_start:
                return int(self.pol2tab[score])
            return int(self.pol2low[round_start, score])
        raise ValueError("dart must be 1, 2 or 3")

    def expected_darts(self, score=None):
        """Expected darts (or rounds, if turn_cost is used) to check out."""
        score = self.game_start if score is None else score
        return -self.V1[score]

    def simulate(self, n_legs=10000, score=None, seed=0):
        """
        Play out legs under the optimal policy, sampling dart outcomes from the
        model's own transition probabilities.

        This is an end-to-end check: the mean number of darts should agree with
        ``expected_darts()`` up to Monte Carlo error.

        Args:
            n_legs (int): number of legs to simulate.
            score (int): starting score, defaults to ``game_start``.
            seed (int): RNG seed.

        Returns:
            tuple[np.ndarray, np.ndarray]: darts thrown and rounds used per leg.
        """
        rng = np.random.default_rng(seed)
        start_score = self.game_start if score is None else score
        S = self.scores
        cdf = np.cumsum(self.P, axis=1)
        # Conditional probability that a dart landing on score s_j was a checkout.
        with np.errstate(divide="ignore", invalid="ignore"):
            co_frac = np.where(self.P > 0, self.CP / np.maximum(self.P, 1e-300), 0.0)

        darts = np.zeros(n_legs, dtype=np.int64)
        rounds = np.zeros(n_legs, dtype=np.int64)
        for leg in range(n_legs):
            s = start_score
            u = start_score
            t = 1
            n_d = 0
            n_r = 1
            while True:
                i = self.policy(u, t, s)
                j = int(np.searchsorted(cdf[i], rng.random()))
                j = min(j, self.n_scores - 1)
                sj = int(S[j])
                n_d += 1
                if sj <= u - 2:
                    u -= sj
                    if t == 3:
                        s, t, n_r = u, 1, n_r + 1
                    else:
                        t += 1
                elif sj == u and rng.random() < co_frac[i, j]:
                    break
                else:  # bust: rest of the round is forfeited
                    u, t, n_r = s, 1, n_r + 1
            darts[leg] = n_d
            rounds[leg] = n_r
        return darts, rounds

    def to_dense_cube(self):
        """
        Expand into the ``(game_start+2, 3, game_start+2)`` cube used by the
        original implementation, indexed ``[score, dart-1, round_start]``.

        Only states that are reachable in play are filled; the rest are zero.
        Provided for comparison against ``compute_3turn_values``.
        """
        G, M = self.game_start, self.max_dart
        cube = np.zeros((G + 2, 3, G + 2))
        for s in range(2, G + 1):
            cube[s, 0, s] = self.V1[s]
            for u in range(max(2, s - M), s + 1):
                if u >= self.u2_indep and u != s:
                    cube[u, 1, s] = self.V2tab[u]
            for u in range(max(2, s - 2 * M), s + 1):
                if u >= self.u3_indep and u != s:
                    cube[u, 2, s] = self.V3tab[u]
            if s >= self.u3_indep:
                cube[s, 2, s] = self.V3tab[s]
            if s >= self.u2_indep:
                cube[s, 1, s] = self.V2tab[s]
        return cube


def solve_3dart(mdp, dart_cost=1.0, turn_cost=0.0, tol=1e-11, progress=False):
    """
    Convenience wrapper: build the transition arrays from a
    :class:`darts.mdp.SinglePlayerContinuousMDP` and solve the 3-dart MDP.

    Returns:
        ThreeDartMDP: the solved model.
    """
    probs_arr, checkout_probs_arr, allowed_scores = build_probs_arrays(mdp)
    solver = ThreeDartMDP(
        probs_arr,
        checkout_probs_arr,
        allowed_scores,
        mdp.game_start,
        dart_cost=dart_cost,
        turn_cost=turn_cost,
    )
    return solver.solve(tol=tol, progress=progress)


# --------------------------------------------------------------------------
# Reference implementation (original; kept for validation)
# --------------------------------------------------------------------------


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
    Numba-jitted value iteration for the 3-dart MDP (reference implementation).

    Processes states in order of increasing round_start. For a fixed round_start s:
      - Dart-3 states (score, 3, s): valid throws reference V(ns, 1, ns) for ns < s,
        already finalized in earlier iterations.
      - Dart-2 states (score, 2, s): valid throws reference V(ns, 3, s), updated
        in the same sweep (Gauss-Seidel).
      - Dart-1 state (s, 1, s): valid throws reference V(ns, 2, s), updated
        in the same sweep.
    Busts always reference V(s, 1, s), creating a within-group fixed point that is
    resolved by iterating until group_delta < threshold.

    This is O(game_start^2) in the number of states swept and is superseded by
    :class:`ThreeDartMDP`; it is retained as an independent check.
    """
    n_points = probs_arr.shape[0]
    n_scores = probs_arr.shape[1]

    for start in range(2, game_start + 1):

        group_converged = False
        while not group_converged:
            group_delta = 0.0

            # ---- Dart 3: (score, 3, start) for score = 2..start ----
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

            # ---- Dart 2: (score, 2, start) for score = 2..start ----
            # Valid throw -> dart 3: V(ns, 3, start) [just updated above]
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

            # ---- Dart 1: (start, 1, start) ----
            # Valid throw -> dart 2: V(ns, 2, start) [just updated above]
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
    suitable for the solvers.

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
    Compute the 3-dart MDP value function with the reference solver.

    Returns:
        values: float64 array of shape (game_start+2, 3, game_start+2).
            Access as values[score, dart-1, round_start].
            Values are negative: multiply by -1 to get expected darts to checkout.
    """
    probs_arr, checkout_probs_arr, allowed_scores = build_probs_arrays(mdp)
    game_start = mdp.game_start
    values = np.zeros((game_start + 2, 3, game_start + 2), dtype=np.float64)

    _compute_3turn_values(
        values, probs_arr, checkout_probs_arr, allowed_scores, game_start, threshold
    )
    return values
