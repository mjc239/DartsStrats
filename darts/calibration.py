"""
Calibrating the throw model against real competition scores.

Every experiment in this project so far has measured a *simulated* player, where
the aim point is known because we chose it. Real match data does not come with
an aim point: a scoresheet says a visit scored 140, not that the darts were
thrown at the treble 20 and two of them landed there.

There is one region of the game where the aim can be recovered from the score
alone, with no circularity and no assumption about what chart the player
follows. While the remaining score is high enough that no checkout is in reach,
the optimal aim is the **treble 20** for every ability this project models, and
every real player throws there too. In that region:

* all three darts of a visit are aimed at the same point, so they are i.i.d.;
* no dart can bust, since a dart thrown at 62 or more cannot;
* so the **visit total is the three-fold convolution** of one dart's score
  distribution, and its likelihood is exact and cheap.

That makes a per-visit scoresheet -- the granularity real data actually comes
in -- usable directly, without needing to know which dart scored what.

The region is not the whole scoring phase. The model prefers the treble 19 at a
handful of scores between 182 and 246, where three treble 20s would leave an
awkward number, and gives up 0.07-0.12 visits if forced to the 20 there. So the
convolution is valid only above a floor, and :func:`scoring_floor` derives that
floor from the solved policy rather than assuming it.

Two further things follow from the single target, and both are limitations
rather than conveniences.

**The aim and the bias cannot be separated.** From one target only the effective
centre ``mu = t + b`` is identifiable -- this is notebook 09's confounding, and
it is why the fit below reports ``mu`` rather than a bias. It also means the fit
cannot tell "aims at the middle of the treble and pulls low" from "aims low and
throws true", which for calibration purposes is fine and for coaching is not.

**The treble 20 is close to the worst place on the board to measure anybody**
(notebook 09): about 3x the variance of the best target for a league player and
5.5x for an elite one. Competition data is therefore the worst measurement
design there is -- but it arrives in enormous quantity, and volume wins. Roughly
9,000 scoring darts pin an elite player's sigma to +/-0.1mm, which is a season
of televised darts.
"""

import numpy as np

from darts.dartboards import generate_dartboard
from darts.fitting import ScoreLikelihood
from darts.utils import mm_per_pixel

#: Centre of the treble 20 bed, in millimetres from the bull.
T20_MM = np.array([0.0, 103.0])

#: Remaining score above which the optimal aim is the treble 20 for every dart
#: and every ability in the pro-to-league range. Derived by
#: :func:`scoring_floor`; see its docstring for how, and why it is not 182.
SCORING_FLOOR = 250

MAX_DART = 60
MAX_VISIT = 3 * MAX_DART


def scoring_floor(sigmas=(6.5, 8.0, 10.0, 13.0), board_pixels=512,
                  point_stride=4, game_start=501, margin=4):
    """
    The lowest score above which the treble 20 is optimal for every dart.

    Solves the leg for each ability and finds the highest score at which any
    dart of a visit prefers something else. Above that score the three darts of
    a visit are aimed identically, which is what makes the convolution in
    :class:`ScoringVisits` exact.

    Args:
        sigmas (sequence): abilities to require the answer to hold for.
        margin (int): added to the answer, since the search is over a discrete
            grid and near-ties should not be trusted to the point.

    Returns:
        int: the floor, to be compared against the score *after* the visit.
    """
    from darts.mdp_3turn import ThreeDartMDP
    from darts.transitions import transition_arrays
    from darts.utils import aim_description

    highest = 0
    for sigma in sigmas:
        tr = transition_arrays(board_pixels, float(sigma), margin_mm=10.0,
                               point_stride=point_stride)
        mdp = ThreeDartMDP(tr["probs"], tr["checkout_probs"], tr["allowed_scores"],
                           game_start, dart_cost=0.0, turn_cost=1.0).solve()
        labels = np.array([aim_description(p, board_pixels) for p in tr["points"]])
        for dart in (1, 2, 3):
            for score in range(182, game_start + 1):
                a = int(np.argmax(mdp.q_values(score, dart, max(score, 182))))
                if labels[a] != "T20":
                    highest = max(highest, score)
    return int(highest + margin)


class ScoringVisits:
    """
    The distribution of a visit total in the pure scoring phase.

    Args:
        board_pixels (int): board resolution. 512 or finer -- see notebook 02.
        board (np.ndarray): a prebuilt board, to avoid rebuilding it.
        floor (int): the score below which the single-target assumption stops
            holding. A visit is usable when the score *after* it is at least
            this, which guarantees every dart in it was thrown from at least
            this score too.
    """

    def __init__(self, board_pixels=512, board=None, floor=SCORING_FLOOR):
        self.like = ScoreLikelihood(board_pixels=board_pixels, board=board)
        self.floor = int(floor)
        self.pixels = self.like.pixels

    # -- distributions ------------------------------------------------------
    def dart_pmf(self, mu, Sigma):
        """P(a single dart scores v), as an array indexed by ``v`` up to 60."""
        probs = self.like.score_probabilities(np.asarray(mu, float),
                                              np.asarray(Sigma, float))
        pmf = np.zeros(MAX_DART + 1)
        for value, p in probs.items():
            if 0 <= value <= MAX_DART:
                pmf[value] += p
        total = pmf.sum()
        return pmf / total if total > 0 else pmf

    def visit_pmf(self, mu, Sigma):
        """
        P(a three-dart visit totals t), indexed by ``t`` up to 180.

        Exact, because in this region the three darts are i.i.d.: the visit
        total is the three-fold convolution of :meth:`dart_pmf`.
        """
        p = self.dart_pmf(mu, Sigma)
        return np.convolve(np.convolve(p, p), p)

    # -- likelihood ---------------------------------------------------------
    def log_likelihood(self, mu, Sigma, score_before, visit_score):
        """
        Log-likelihood of observed visit totals, truncated by the selection.

        A visit is only included when the score after it is at least ``floor``,
        which is a condition on the total: ``visit_score <= score_before -
        floor``. Selecting on the observation would bias the fit, so the
        likelihood is conditioned on the same event -- each visit contributes
        ``p(t) / P(total <= score_before - floor)``.
        """
        score_before = np.asarray(score_before, dtype=np.int64)
        visit_score = np.asarray(visit_score, dtype=np.int64)
        pmf = self.visit_pmf(mu, Sigma)
        cdf = np.cumsum(pmf)

        cut = np.clip(score_before - self.floor, 0, MAX_VISIT)
        if np.any(visit_score > cut):
            raise ValueError("a visit scored more than its selection allows; "
                             "filter with usable() first")
        num = np.log(np.maximum(pmf[visit_score], 1e-300))
        den = np.log(np.maximum(cdf[cut], 1e-300))
        return float((num - den).sum())

    def usable(self, score_before, visit_score, darts_used=None):
        """
        Mask of the visits this model may be fitted to.

        Keeps visits that stayed in the pure scoring region -- the score after
        the visit is at least ``floor`` -- and that used all three darts, since
        a visit cut short by a checkout is not three throws at the same target.
        """
        score_before = np.asarray(score_before, dtype=np.int64)
        visit_score = np.asarray(visit_score, dtype=np.int64)
        ok = (score_before - visit_score) >= self.floor
        if darts_used is not None:
            ok &= np.asarray(darts_used, dtype=np.int64) == 3
        return ok

    # -- what a scoresheet would show ---------------------------------------
    def statistics(self, mu, Sigma):
        """
        The published match statistics this throw would produce, in the
        scoring phase: the three-dart average and the scoring-band rates.

        These are the quantities available when only aggregates are published,
        and :func:`fit_from_aggregates` inverts them.
        """
        pmf = self.visit_pmf(mu, Sigma)
        t = np.arange(len(pmf))
        return {
            "three_dart_average": float((pmf * t).sum()),
            "p_180": float(pmf[180]),
            "p_140_plus": float(pmf[140:].sum()),
            "p_100_plus": float(pmf[100:].sum()),
            "p_60_plus": float(pmf[60:].sum()),
        }


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------

def _unpack(theta, model):
    """(parameter vector, model) -> (mu, Sigma)."""
    if model == "isotropic":
        mu = np.array([theta[0], theta[1]])
        s = np.exp(theta[2])
        return mu, s ** 2 * np.eye(2)
    mu = np.array([theta[0], theta[1]])
    # log-Cholesky, which keeps Sigma positive definite without a constraint
    L = np.array([[np.exp(theta[2]), 0.0], [theta[4], np.exp(theta[3])]])
    return mu, L @ L.T


def _pack(mu, sigma, model):
    if model == "isotropic":
        return np.array([mu[0], mu[1], np.log(sigma)])
    return np.array([mu[0], mu[1], np.log(sigma), np.log(sigma), 0.0])


def fit_from_visits(score_before, visit_score, darts_used=None, model="isotropic",
                    board_pixels=512, board=None, floor=SCORING_FLOOR,
                    mu_init=None, sigma_init=12.0, fix_mu=False):
    """
    Fit a throw from per-visit totals in the pure scoring phase.

    Args:
        score_before (sequence[int]): remaining score at the start of each visit.
        visit_score (sequence[int]): what the visit scored.
        darts_used (sequence[int]): darts thrown, to drop checkout visits.
        model (str): ``"isotropic"`` for a single sigma, ``"full"`` for a
            covariance. From one target the full model is weakly identified --
            see notebook 09 -- so the isotropic fit is the honest default.
        fix_mu (bool): hold the aim at the treble 20 centre instead of fitting
            it. Only sensible if you trust the player to aim at the bed centre;
            ``mu`` otherwise absorbs their bias, which cannot be separated from
            the aim at a single target.

    Returns:
        dict: ``mu``, ``Sigma``, ``sigma_mm``, ``n_visits``, ``log_likelihood``,
        ``converged``, and the fitted statistics.
    """
    from scipy.optimize import minimize

    sv = ScoringVisits(board_pixels=board_pixels, board=board, floor=floor)
    sb = np.asarray(score_before, dtype=np.int64)
    vs = np.asarray(visit_score, dtype=np.int64)
    keep = sv.usable(sb, vs, darts_used)
    sb, vs = sb[keep], vs[keep]
    if len(vs) == 0:
        raise ValueError("no visits survive the scoring-phase filter")

    mu0 = T20_MM if mu_init is None else np.asarray(mu_init, float)
    theta0 = _pack(mu0, sigma_init, model)
    free = slice(2, None) if fix_mu else slice(None)

    def nll(free_theta):
        theta = theta0.copy()
        theta[free] = free_theta
        mu, Sigma = _unpack(theta, model)
        return -sv.log_likelihood(mu, Sigma, sb, vs)

    res = minimize(nll, theta0[free], method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 4000})
    theta = theta0.copy()
    theta[free] = res.x
    mu, Sigma = _unpack(theta, model)
    return {"mu": mu, "Sigma": Sigma,
            "sigma_mm": float(np.sqrt(np.trace(Sigma) / 2)),
            "n_visits": int(len(vs)), "n_darts": int(3 * len(vs)),
            "log_likelihood": float(-res.fun), "converged": bool(res.success),
            "statistics": sv.statistics(mu, Sigma)}


def fit_from_aggregates(three_dart_average=None, p_180=None, p_140_plus=None,
                        p_100_plus=None, board_pixels=512, board=None,
                        sigma_init=12.0, weights=None):
    """
    Fit a single sigma to published match aggregates.

    The only statistics most sources publish are the three-dart average and the
    counts of big visits. Each is a scalar function of sigma, so any one of them
    determines it -- which is exactly why fitting *several* is the interesting
    thing to do. If one sigma cannot reproduce the average and the 180 rate at
    once, the isotropic Gaussian is wrong, and that is a result rather than a
    nuisance. See :func:`aggregate_consistency`.

    Args:
        weights (dict): per-statistic weights for the least-squares objective.
            Defaults to equal weight on each supplied statistic, after scaling
            each by its own value so they are comparable.
    """
    from scipy.optimize import minimize_scalar

    sv = ScoringVisits(board_pixels=board_pixels, board=board)
    observed = {k: v for k, v in
                (("three_dart_average", three_dart_average), ("p_180", p_180),
                 ("p_140_plus", p_140_plus), ("p_100_plus", p_100_plus))
                if v is not None}
    if not observed:
        raise ValueError("supply at least one statistic")
    w = weights or {k: 1.0 for k in observed}

    def cost(log_sigma):
        pred = sv.statistics(T20_MM, np.exp(log_sigma) ** 2 * np.eye(2))
        return sum(w[k] * ((pred[k] - v) / max(abs(v), 1e-9)) ** 2
                   for k, v in observed.items())

    res = minimize_scalar(cost, bounds=(np.log(2.0), np.log(60.0)),
                          method="bounded", options={"xatol": 1e-8})
    sigma = float(np.exp(res.x))
    return {"sigma_mm": sigma, "cost": float(res.fun), "observed": observed,
            "predicted": sv.statistics(T20_MM, sigma ** 2 * np.eye(2))}


def aggregate_consistency(statistics, board_pixels=512, board=None):
    """
    The sigma implied by each published statistic *separately*.

    If the isotropic Gaussian describes a real player, every statistic they
    generate implies the same sigma. If their 180 rate implies 7mm and their
    three-dart average implies 9mm, one model cannot produce both, and the
    disagreement is evidence about *how* it fails rather than merely that it
    does.

    Returns:
        dict: statistic name -> implied sigma in mm (nan where the observed
        value is outside the range the model can produce at any sigma).
    """
    out = {}
    for name, value in statistics.items():
        try:
            fit = fit_from_aggregates(board_pixels=board_pixels, board=board,
                                      **{name: value})
            pred = fit["predicted"][name]
            rel = abs(pred - value) / max(abs(value), 1e-9)
            out[name] = fit["sigma_mm"] if rel < 0.02 else float("nan")
        except (TypeError, ValueError):
            out[name] = float("nan")
    return out


# --------------------------------------------------------------------------
# The other half of the board: doubles
# --------------------------------------------------------------------------

class DoubleAttempts:
    """
    Hit probability for a dart aimed at a double, as a function of the throw.

    The scoring statistics measure a player at the treble 20; the checkout
    statistics measure them at a double, 60mm further out and in a bed of a
    different shape. Under one throw model both must be explained by the same
    ``Sigma``. Whether they are is the question notebook 18 asks.
    """

    def __init__(self, board_pixels=512, board=None, checkouts=None):
        if board is None:
            board, checkouts = generate_dartboard(board_pixels)
        elif checkouts is None:
            raise ValueError("pass checkouts along with a prebuilt board")
        self.board, self.checkouts = board, checkouts
        self.pixels = board.shape[0]
        self.mm_per_pixel = mm_per_pixel(self.pixels)
        offs = (np.arange(self.pixels) - self.pixels // 2) * self.mm_per_pixel
        x, y = np.meshgrid(offs, offs)
        self.coords = np.stack([x.ravel(), y.ravel()], axis=1)
        self._flat = board.ravel()
        self._co = checkouts.ravel().astype(bool)

    def bed_centre(self, number):
        """Millimetre centre of the double bed for ``number`` (25 for the bull)."""
        mask = (self._flat == (50 if number == 25 else 2 * number)) & self._co
        if not mask.any():
            raise ValueError(f"no double bed found for {number}")
        return self.coords[mask].mean(axis=0)

    def hit_probability(self, Sigma, number, mu=None):
        """P(a dart aimed at the bed centre lands in that double)."""
        mu = self.bed_centre(number) if mu is None else np.asarray(mu, float)
        mask = (self._flat == (50 if number == 25 else 2 * number)) & self._co
        d = self.coords - mu
        Sigma = np.asarray(Sigma, float)
        det = Sigma[0, 0] * Sigma[1, 1] - Sigma[0, 1] * Sigma[1, 0]
        inv = np.array([[Sigma[1, 1], -Sigma[0, 1]],
                        [-Sigma[1, 0], Sigma[0, 0]]]) / det
        q = np.einsum("ij,jk,ik->i", d, inv, d)
        pdf = np.exp(-0.5 * q) / (2 * np.pi * np.sqrt(det))
        return float(pdf[mask].sum() * self.mm_per_pixel ** 2)

    def fit_sigma(self, attempts, hits, number, sigma_init=12.0):
        """
        Maximum-likelihood sigma from Binomial double attempts.

        Args:
            attempts, hits (int or sequence): darts thrown at that double and
                darts that hit it. Sequences are summed, so several matches can
                be pooled.
            number (int): which double, e.g. 16 for D16.
        """
        from scipy.optimize import minimize_scalar
        n = int(np.sum(attempts))
        k = int(np.sum(hits))
        if not 0 <= k <= n or n == 0:
            raise ValueError("need 0 <= hits <= attempts and attempts > 0")

        def nll(log_sigma):
            p = self.hit_probability(np.exp(log_sigma) ** 2 * np.eye(2), number)
            p = min(max(p, 1e-12), 1 - 1e-12)
            return -(k * np.log(p) + (n - k) * np.log1p(-p))

        res = minimize_scalar(nll, bounds=(np.log(2.0), np.log(60.0)),
                              method="bounded", options={"xatol": 1e-8})
        sigma = float(np.exp(res.x))
        return {"sigma_mm": sigma, "attempts": n, "hits": k,
                "observed_rate": k / n,
                "predicted_rate": self.hit_probability(
                    sigma ** 2 * np.eye(2), number),
                "log_likelihood": float(-res.fun)}
