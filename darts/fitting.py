"""
Fitting a player's throwing distribution.

Two situations:

**You know where the darts landed.** If you can photograph the board and read
off coordinates, the maximum likelihood estimate is just the sample mean and
covariance. :func:`fit_from_positions`.

**You only know what they scored.** This is the realistic case -- you can write
down "T20, 5, T20, 1, 20, ..." from memory or a scoresheet, but not millimetres.
The landing position is then a latent variable and the fit needs EM.
:func:`fit_from_scores`.

Relation to the earlier implementation
--------------------------------------
``darts/darts.py`` already had a Monte Carlo EM for the second case
(``ThrowingDistribution.estimate_distribution``). Its derivation is right --
the M step is the standard Gaussian one, and the E step is self-normalised
importance sampling of ``E[Z | X]`` -- but it has some problems:

* the E step sampled pixels **uniformly at random** from each scoring region and
  reweighted. Since the board is already a finite pixel grid, that expectation
  is a plain weighted sum over the region and can be computed **exactly**. That
  removes the Monte Carlo error entirely, makes each iteration deterministic,
  and is faster;
* convergence was tested on ``Q``, the EM surrogate, and compared across
  iterations that used different Monte Carlo sample sizes, so the criterion was
  not measuring what it appeared to. The observed-data log-likelihood is
  available in closed form here and is used instead, which also lets the
  monotonicity of EM be asserted as a test;
* ``np.mean(likelihoods[-5:])`` on the first iteration is ``np.mean([])``, i.e.
  a NaN and a warning;
* the loop exited when the sample size hit its cap whether or not anything had
  converged;
* it was built on the second, older board implementation in ``darts/darts.py``,
  whose coordinate conventions differ from the rest of the package.

Identifiability
---------------
Scores alone constrain ``(mu, Sigma)`` weakly. A tight group in the 20 bed and
a loose group centred slightly low can produce similar score histograms, so
with few throws the likelihood is flat along a ridge. :func:`fit_from_scores`
reports a bootstrap standard error so this is visible rather than implied, and
:func:`effective_sample_size` warns when a fit is being read too confidently.

Fitting a Student-t
-------------------
Notebook 21 measured a dart's landing point on real competition darts and found
it is a **Student-t**, not a Gaussian. Every function here takes ``nu``;
``nu=None`` is the Gaussian and runs exactly the code it always did.

The t costs almost nothing to add, because of what a t *is*: a Gaussian whose
width is redrawn for every dart. Write ``Z | W ~ N(mu, Sigma / W)`` with
``W ~ Gamma(nu/2, nu/2)`` and the whole distribution is recovered. So there are
now two latent variables per dart -- where it landed and how wide that dart was
-- and the second one turns out to slot into the machinery the first one already
needed. The E step gains one weight,

    u(z) = E[W | Z = z] = (nu + 2) / (nu + q(z)),   q = (z-mu)' Sigma^-1 (z-mu)

applied pixel by pixel inside the same sum, and the M step is the *same*
weighted Gaussian one. A dart that would have to have landed a long way out gets
downweighted rather than being allowed to drag the fitted spread, which is the
whole behavioural difference. At ``nu = inf`` the weight is 1 everywhere and the
two agree exactly, which is how the t path is tested.

``nu`` itself is not estimated inside the EM. It is **profiled** on a grid by
:func:`profile_nu`: notebook 21 found it sharply identified but strongly
correlated with the core scale, and a profile shows that ridge instead of
hiding it inside a point estimate.

One warning about reading the output. When ``nu`` is set, ``Sigma`` is the
**scale** matrix and ``sigma_mm`` is a **core scale**, not a standard deviation
-- a t with ``nu`` near 2 has a variance that barely exists. Comparing that
number with a Gaussian fit's ``sigma_mm`` compares two different quantities; the
project's convention for putting them on the same footing is
:func:`darts.transitions.matched_scale`.
"""

import numpy as np

from darts.dartboards import generate_dartboard
from darts.utils import mm_per_pixel


def fit_from_positions(positions_mm):
    """
    Maximum likelihood fit when the landing positions are known.

    Args:
        positions_mm (array-like): (n, 2) array of (x, y) landing positions in
            millimetres from the centre of the board.

    Returns:
        dict: ``mu`` (2,), ``Sigma`` (2, 2), ``sigma_mm`` (the isotropic
        equivalent, sqrt of the mean eigenvalue), and ``n``.
    """
    z = np.asarray(positions_mm, dtype=float)
    if z.ndim != 2 or z.shape[1] != 2:
        raise ValueError("positions_mm must be (n, 2)")
    mu = z.mean(axis=0)
    Sigma = np.cov(z.T, bias=True)
    return {"mu": mu, "Sigma": Sigma,
            "sigma_mm": float(np.sqrt(np.trace(Sigma) / 2)), "n": len(z)}


class ScoreLikelihood:
    """
    The board, precomputed into the pieces the EM needs.

    Holds the millimetre coordinates of every pixel, grouped by the score in
    that pixel, so that ``E[Z | X = s]`` is an exact weighted sum rather than a
    Monte Carlo estimate.
    """

    def __init__(self, board_pixels=256, board=None, quadro=False, nu=None):
        if board is None:
            board, _ = generate_dartboard(board_pixels, quadro=quadro)
        if nu is not None and nu <= 0:
            raise ValueError("nu must be positive")
        self.nu = nu
        self.board = board
        self.pixels = board.shape[0]
        self.mm_per_pixel = mm_per_pixel(self.pixels)
        self.pixel_area = self.mm_per_pixel ** 2

        offs = (np.arange(self.pixels) - self.pixels // 2) * self.mm_per_pixel
        x, y = np.meshgrid(offs, offs)          # x varies along columns
        self.coords = np.stack([x.ravel(), y.ravel()], axis=1)
        flat = board.ravel().astype(np.int64)
        self.scores = np.unique(flat)
        self.index = {int(s): np.flatnonzero(flat == s) for s in self.scores}

    def _quadratic(self, mu, Sigma):
        """``q(z) = (z-mu)' Sigma^-1 (z-mu)`` at every pixel, and ``det Sigma``."""
        d = self.coords - mu
        det = Sigma[0, 0] * Sigma[1, 1] - Sigma[0, 1] * Sigma[1, 0]
        inv = np.array([[Sigma[1, 1], -Sigma[0, 1]], [-Sigma[1, 0], Sigma[0, 0]]]) / det
        return np.einsum("ij,jk,ik->i", d, inv, d), det

    def _pdf(self, mu, Sigma):
        q, det = self._quadratic(mu, Sigma)
        if self.nu is None:
            return np.exp(-0.5 * q) / (2 * np.pi * np.sqrt(det))
        if np.isinf(self.nu):
            log_profile = -0.5 * q
        else:
            log_profile = -0.5 * (self.nu + 2.0) * np.log1p(q / self.nu)
        return np.exp(log_profile) / (2 * np.pi * np.sqrt(det))

    def mixture_weight(self, mu, Sigma):
        """
        ``E[W | Z = z]`` at every pixel, where ``W`` is the dart's own precision
        multiplier in the scale mixture.

        This is the only thing that distinguishes a Student-t fit from a
        Gaussian one. It is ``1`` everywhere for a Gaussian -- every dart is the
        same width -- and ``(nu+2)/(nu+q)`` for a t, which is near ``(nu+2)/nu``
        in the middle of the group and near zero for a dart that would have had
        to travel. The M step is then the Gaussian one with these weights, so a
        far dart is treated as evidence of a *wide dart* rather than evidence of
        a wide player.
        """
        if self.nu is None or np.isinf(self.nu):
            return np.ones(len(self.coords))
        q, _ = self._quadratic(mu, Sigma)
        return (self.nu + 2.0) / (self.nu + q)

    def score_probabilities(self, mu, Sigma):
        """
        P(score = s) for every board score, as a dict.

        A Student-t puts real mass beyond the board array -- 1e-3 of it at
        ``nu = 2.25`` -- and a dart that lands there scores nothing. That mass is
        added to score 0 rather than dropped, which is what keeps the
        probabilities a distribution. For a Gaussian the deficit is round-off, so
        that path is left alone entirely.
        """
        w = self._pdf(mu, Sigma) * self.pixel_area
        out = {int(s): float(w[idx].sum()) for s, idx in self.index.items()}
        if self.nu is not None and 0 in out:
            out[0] += max(1.0 - float(w.sum()), 0.0)
        return out

    def log_likelihood(self, mu, Sigma, counts):
        """Observed-data log-likelihood of a bag of scores."""
        p = self.score_probabilities(mu, Sigma)
        total = 0.0
        for s, n in counts.items():
            total += n * np.log(max(p.get(int(s), 0.0), 1e-300))
        return total

    def conditional_moments_all(self, mu, Sigma, scores):
        """
        Exact ``E[Z | X = s]`` and ``E[Z Z^T | X = s]`` for several scores at
        once.

        The density is evaluated over the whole board once and then split by
        score, rather than once per score, which is the difference between a
        fit taking seconds and taking minutes.
        """
        pdf = self._pdf(mu, Sigma)
        out = {}
        for s in scores:
            idx = self.index[int(s)]
            w = pdf[idx]
            tot = w.sum()
            if tot <= 0:
                w = np.ones(len(idx))
                tot = float(len(idx))
            z = self.coords[idx]
            ez = (w @ z) / tot
            out[int(s)] = (ez, (z * w[:, None]).T @ z / tot)
        return out

    def mixture_moments_all(self, mu, Sigma, scores):
        """
        The E step of the Student-t fit: ``E[W | X=s]``, ``E[W Z | X=s]`` and
        ``E[W Z Z' | X=s]``, exactly, for several scores at once.

        Same sum over the same pixels as :meth:`conditional_moments_all`, with
        the density weighted by :meth:`mixture_weight` in the numerators and
        left alone in the denominator -- because the denominator is
        ``P(X = s)``, which is a statement about the throw and not about the
        dart's width.

        The pixel grid stops at the edge of the board array, so the tail beyond
        it is missing from these sums. It is negligible *because* of the weight:
        out there ``u`` is of order ``(nu+2)/q``, some 2e-3 at the array's edge
        for a realistic scale, multiplying a mass of about 1e-3. Downweighting
        the far darts is exactly what makes truncating them harmless, which is
        not true of the Gaussian moments above.

        Returns:
            dict: ``score -> (a, b, C)`` with ``a`` scalar, ``b`` (2,), ``C``
            (2, 2).
        """
        pdf = self._pdf(mu, Sigma)
        weight = self.mixture_weight(mu, Sigma)
        out = {}
        for s in scores:
            idx = self.index[int(s)]
            p = pdf[idx]
            tot = p.sum()
            if tot <= 0:
                # numerically impossible under these parameters; fall back to the
                # unweighted region, as the Gaussian E step does, so EM can move
                p = np.ones(len(idx))
                tot = float(len(idx))
            w = p * weight[idx]
            z = self.coords[idx]
            out[int(s)] = (float(w.sum()) / tot,
                           (w @ z) / tot,
                           (z * w[:, None]).T @ z / tot)
        return out

    def conditional_moments(self, mu, Sigma, score):
        """
        Exact ``E[Z | X = score]`` and ``E[Z Z^T | X = score]`` under the
        current parameters -- a weighted sum over the pixels of that score.
        """
        idx = self.index[int(score)]
        w = self._pdf(mu, Sigma)[idx]
        tot = w.sum()
        if tot <= 0:
            # numerically impossible score under these parameters; fall back to
            # the unweighted centroid of the region so EM can still move
            w = np.ones(len(idx))
            tot = float(len(idx))
        z = self.coords[idx]
        ez = (w @ z) / tot
        ezz = (z * w[:, None]).T @ z / tot
        return ez, ezz


def _project_covariance(Sigma):
    """Nearest symmetric positive definite matrix, against round-off."""
    Sigma = 0.5 * (Sigma + Sigma.T)
    w, V = np.linalg.eigh(Sigma)
    return V @ np.diag(np.maximum(w, 1e-6)) @ V.T


def _pack(means, Sigma):
    return np.concatenate([np.asarray(means, float).ravel(),
                           [Sigma[0, 0], Sigma[0, 1], Sigma[1, 1]]])


def _unpack(theta):
    means = theta[:-3].reshape(-1, 2)
    Sigma = _project_covariance(np.array([[theta[-3], theta[-2]],
                                          [theta[-2], theta[-1]]]))
    return means, Sigma


def _squarem(em_step, log_lik, theta, tol, max_iter, accelerate=True,
             verbose=False, max_backtracks=8):
    """
    Run an EM iteration to convergence, optionally with SQUAREM acceleration.

    Plain EM here is linearly convergent and *slow* -- several hundred
    iterations to settle sigma to five figures, because the score of a dart
    says so little about where it landed that each step moves the parameters
    only slightly. SQUAREM takes two ordinary EM steps, reads off the implied
    direction and step length, and jumps:

        r = F(t) - t,   v = F(F(t)) - F(t) - r,   alpha = -|r| / |v|
        t' = t - 2 alpha r + alpha^2 v

    The jump is then backtracked toward ``alpha = -1``, at which point it
    degenerates to two plain EM steps, until it does at least as well as those
    two steps would have. That safeguard means the accelerated iteration still
    increases the observed-data log-likelihood monotonically, which is the
    property worth protecting -- it is the check that the exact E step is
    right.

    The backtrack halves the distance from ``alpha = -1``, so an extrapolation
    that starts a long way out and never succeeds costs one EM step per halving
    and gets nowhere. That is rare when the throw is Gaussian and routine when it
    is a Student-t, whose EM path is more curved: unbounded, it was spending
    forty-odd steps an iteration to arrive back at the two plain ones. Capping
    the backtracks costs nothing, because the fallback *is* those two steps.

    Args:
        em_step (callable): one EM step, mapping a packed parameter vector to
            the next.
        log_lik (callable): observed-data log-likelihood of a packed vector.
        theta (np.ndarray): starting parameters, packed.
        tol (float): relative log-likelihood change to stop at.
        max_iter (int): cap on outer iterations.
        accelerate (bool): set ``False`` for plain EM.
        verbose (bool): print progress.
        max_backtracks (int): step halvings to try before giving up on the jump
            and taking the two plain EM steps instead.

    Returns:
        tuple: ``(theta, history, converged, n_em_steps)``.
    """
    history = [log_lik(theta)]
    converged = False
    n_steps = 0

    for it in range(max_iter):
        t1 = em_step(theta)
        n_steps += 1
        best = t1

        if accelerate:
            t2 = em_step(t1)
            n_steps += 1
            ll2 = log_lik(t2)
            best = t2
            r = t1 - theta
            v = t2 - t1 - r
            nv = float(np.linalg.norm(v))
            if nv > 1e-300:
                alpha = min(-float(np.linalg.norm(r)) / nv, -1.0)
                tries = 0
                while alpha < -1.0 - 1e-12 and tries < max_backtracks:
                    cand = em_step(theta - 2 * alpha * r + alpha ** 2 * v)
                    n_steps += 1
                    tries += 1
                    llc = log_lik(cand)
                    if np.isfinite(llc) and llc >= ll2:
                        best = cand
                        break
                    alpha = min(-1.0, (alpha - 1.0) / 2.0)

        ll = log_lik(best)
        if ll < history[-1]:
            # the safeguard: fall back on the plain step, which cannot decrease
            best = t1
            ll = log_lik(t1)
        theta = best
        history.append(ll)
        if verbose:
            means, Sigma = _unpack(theta)
            print(f"  iter {it:>3}: loglik {ll:.6f}  "
                  f"sigma {np.sqrt(np.trace(Sigma) / 2):.3f}mm")
        if abs(ll - history[-2]) <= tol * max(1.0, abs(history[-2])):
            converged = True
            break

    return theta, history, converged, n_steps


def fit_from_scores(scores, board_pixels=256, mu_init=None, Sigma_init=None,
                    tol=1e-10, max_iter=500, board=None, verbose=False,
                    accelerate=True, nu=None):
    """
    Fit ``(mu, Sigma)`` from observed scores alone, by exact EM.

    Every throw is assumed aimed at the same (unknown) point ``mu`` -- i.e. this
    is for a practice session at one target, not a bag of match darts. See
    :func:`fit_multi_target` for a session split across several targets, which
    measures the spread considerably more precisely for the same darts.

    Args:
        scores (sequence[int]): the observed dart scores.
        board_pixels (int): resolution of the board used for the likelihood.
            The E step is a sum over pixels, so this is the accuracy knob.
        mu_init, Sigma_init: starting values; defaults to the centre of the
            board and a 30mm isotropic spread.
        tol (float): relative change in log-likelihood to stop at. The default
            is tight because EM converges slowly here: at ``1e-8`` the fit can
            stop with sigma still drifting in the third significant figure.
        max_iter (int): cap on iterations.
        board (np.ndarray): a prebuilt board array, to avoid rebuilding it.
        verbose (bool): print the likelihood each iteration.
        accelerate (bool): use SQUAREM (see :func:`_squarem`).
        nu (float): fit a Student-t of this many degrees of freedom instead of a
            Gaussian. ``None`` is the Gaussian. ``Sigma`` and ``sigma_mm`` are
            then a **scale**, not a variance -- see the module docstring.

    Returns:
        dict: ``mu``, ``Sigma``, ``sigma_mm``, ``nu``, ``log_likelihood``,
        ``n_iter``, ``n_em_steps``, ``converged``, ``history``.
    """
    scores = np.asarray(scores, dtype=np.int64)
    like = ScoreLikelihood(board_pixels, board=board, nu=nu)
    unknown = set(int(s) for s in np.unique(scores)) - set(int(s) for s in like.scores)
    if unknown:
        raise ValueError(f"scores not achievable on this board: {sorted(unknown)}")

    counts = {int(s): int((scores == s).sum()) for s in np.unique(scores)}
    n = len(scores)

    def em_step(theta):
        (mu,), Sigma = _unpack(theta)
        if nu is None:
            # E step: exact conditional moments, one density evaluation for all
            # distinct scores
            moments = like.conditional_moments_all(mu, Sigma, counts)
            # M step: the Gaussian MLE using those moments
            ez = sum(counts[s] * moments[s][0] for s in counts) / n
            ezz = sum(counts[s] * moments[s][1] for s in counts) / n
            return _pack([ez], ezz - np.outer(ez, ez))
        # E step: the same sums, weighted by each dart's own width
        moments = like.mixture_moments_all(mu, Sigma, counts)
        A = sum(counts[s] * moments[s][0] for s in counts)
        B = sum(counts[s] * moments[s][1] for s in counts)
        C = sum(counts[s] * moments[s][2] for s in counts)
        # M step: the weighted Gaussian MLE. The mean divides by the weight, the
        # scale by the count -- that asymmetry is the t update, and it is what
        # keeps a handful of wide darts from inflating the fitted core.
        mu_new = B / A
        return _pack([mu_new], (C - A * np.outer(mu_new, mu_new)) / n)

    def log_lik(theta):
        (mu,), Sigma = _unpack(theta)
        return like.log_likelihood(mu, Sigma, counts)

    mu = np.zeros(2) if mu_init is None else np.asarray(mu_init, float)
    Sigma = 30.0 ** 2 * np.eye(2) if Sigma_init is None else np.asarray(Sigma_init, float)

    theta, history, converged, n_steps = _squarem(
        em_step, log_lik, _pack([mu], Sigma), tol, max_iter,
        accelerate=accelerate, verbose=verbose)
    (mu,), Sigma = _unpack(theta)

    return {"mu": mu, "Sigma": Sigma,
            "sigma_mm": float(np.sqrt(np.trace(Sigma) / 2)), "nu": nu,
            "log_likelihood": history[-1], "n_iter": len(history) - 1,
            "n_em_steps": n_steps, "converged": converged, "history": history}


def fit_multi_target(sessions, board_pixels=256, b_init=None, Sigma_init=None,
                     tol=1e-10, max_iter=500, board=None, verbose=False,
                     shared_bias=True, accelerate=True, nu=None):
    """
    Fit one throwing distribution from throws aimed at *several* targets.

    The session model is: a dart aimed at target ``t`` lands at
    ``N(t + b, Sigma)``, with the bias ``b`` and the spread ``Sigma`` shared
    across targets. Only the scores are observed.

    The parameter count is five -- ``(b_x, b_y, Sigma_xx, Sigma_xy,
    Sigma_yy)`` -- **whatever the number of targets**, which is what makes
    "200 darts at one target" and "100 at each of two" a like-for-like
    comparison. Splitting the darts buys information without spending any of it
    on extra parameters.

    Args:
        sessions (sequence): pairs ``(target_mm, scores)``, where ``target_mm``
            is the (x, y) point in millimetres from the centre that the player
            was asked to aim at, and ``scores`` the darts they scored there.
        board_pixels (int): resolution used for the likelihood.
        b_init, Sigma_init: starting values.
        tol (float): relative log-likelihood change to stop at.
        max_iter (int): iteration cap.
        board (np.ndarray): prebuilt board array.
        verbose (bool): print the likelihood each iteration.
        shared_bias (bool): if ``False``, give every target its own free mean,
            so the fit assumes nothing about the bias being the same at each.
            Costs ``2k`` parameters instead of 2 and is only worth it when you
            suspect the aim error depends on the target.
        nu (float): fit a Student-t of this many degrees of freedom instead of a
            Gaussian. ``None`` is the Gaussian.

    Returns:
        dict: ``b`` (or ``mu_by_target``), ``Sigma``, ``sigma_mm``, ``nu``,
        ``log_likelihood``, ``n_iter``, ``converged``, ``history``, ``n``.
    """
    like = ScoreLikelihood(board_pixels, board=board, nu=nu)
    known = set(int(s) for s in like.scores)

    targets, counts = [], []
    for target_mm, scores in sessions:
        s = np.asarray(scores, dtype=np.int64)
        unknown = set(int(v) for v in np.unique(s)) - known
        if unknown:
            raise ValueError(f"scores not achievable on this board: {sorted(unknown)}")
        targets.append(np.asarray(target_mm, dtype=float))
        counts.append({int(v): int((s == v).sum()) for v in np.unique(s)})
    if not targets:
        raise ValueError("no sessions given")
    n = sum(sum(c.values()) for c in counts)

    def em_step(theta):
        offsets, Sigma = _unpack(theta)
        sum_a = 0.0
        sum_r = np.zeros(2)
        sum_rr = np.zeros((2, 2))
        per_target = []
        # E step: exact conditional moments of the residual R = Z - t at each
        # target, under that target's current mean. For a Student-t every moment
        # additionally carries that dart's own width, W; a is the total weight,
        # and reduces to the dart count when the throw is Gaussian.
        for i, (t, cnt) in enumerate(zip(targets, counts)):
            mu_i = t + offsets[0 if shared_bias else i]
            mom = (like.conditional_moments_all(mu_i, Sigma, cnt) if nu is None
                   else like.mixture_moments_all(mu_i, Sigma, cnt))
            a_i = 0.0
            r_i = np.zeros(2)
            rr_i = np.zeros((2, 2))
            for s, k in cnt.items():
                a, ez, ezz = (1.0, *mom[s]) if nu is None else mom[s]
                # E[W R R^T] = E[W Z Z^T] - t E[W Z]^T - E[W Z] t^T + E[W] t t^T
                rr = ezz - np.outer(t, ez) - np.outer(ez, t) + a * np.outer(t, t)
                a_i += k * a
                r_i += k * (ez - a * t)
                rr_i += k * rr
            per_target.append((a_i, r_i))
            sum_a += a_i
            sum_r += r_i
            sum_rr += rr_i

        # M step: the weighted Gaussian MLE from those moments. With a shared
        # bias the whole session contributes to one mean; with free means each
        # target gets its own, and the spread is measured about each. The mean
        # divides by the total weight and the spread by the dart count -- for a
        # Gaussian those are the same number and this is the update it always
        # was.
        if shared_bias:
            new = [sum_r / sum_a]
            correction = sum_a * np.outer(new[0], new[0])
        else:
            new = [r_i / a_i for a_i, r_i in per_target]
            correction = sum(a_i * np.outer(o, o)
                             for o, (a_i, _) in zip(new, per_target))
        return _pack(new, (sum_rr - correction) / n)

    def log_lik(theta):
        offsets, Sigma = _unpack(theta)
        return sum(like.log_likelihood(t + offsets[0 if shared_bias else i],
                                       Sigma, cnt)
                   for i, (t, cnt) in enumerate(zip(targets, counts)))

    b = np.zeros(2) if b_init is None else np.asarray(b_init, float)
    Sigma = (30.0 ** 2 * np.eye(2) if Sigma_init is None
             else np.asarray(Sigma_init, float))
    start = _pack([b] if shared_bias else [b] * len(targets), Sigma)

    theta, history, converged, n_steps = _squarem(
        em_step, log_lik, start, tol, max_iter, accelerate=accelerate,
        verbose=verbose)
    offsets, Sigma = _unpack(theta)

    out = {"Sigma": Sigma, "sigma_mm": float(np.sqrt(np.trace(Sigma) / 2)),
           "nu": nu,
           "log_likelihood": history[-1], "n_iter": len(history) - 1,
           "n_em_steps": n_steps, "converged": converged, "history": history,
           "n": n, "targets": np.array(targets)}
    if shared_bias:
        out["b"] = offsets[0]
    else:
        out["mu_by_target"] = offsets
    return out


#: Degrees of freedom to profile over. Spaced geometrically in ``nu - 2``,
#: because that is what the likelihood is smooth in, and bracketing the 2.05-12
#: range notebook 21 fitted across seventeen professionals. ``inf`` is the
#: Gaussian and is included so the profile has its null in it.
NU_GRID = (2.05, 2.1, 2.25, 2.5, 3.0, 4.0, 6.0, 10.0, 20.0, 50.0, np.inf)


def profile_nu(fitter, *args, nu_grid=NU_GRID, **kwargs):
    """
    Fit at each ``nu`` in turn and return the profile likelihood.

    ``nu`` is not estimated inside the EM. It could be -- the scale mixture has
    a closed-form update for it too -- but a point estimate would hide the thing
    worth seeing. Notebook 21 found ``nu`` and the core scale strongly
    correlated (r = +0.62 across players), so they trace out a ridge rather than
    a peak, and a profile shows the ridge.

    Args:
        fitter (callable): :func:`fit_from_scores` or :func:`fit_multi_target`.
        *args: passed through to ``fitter`` (the scores, or the sessions).
        nu_grid (sequence): degrees of freedom to try. ``np.inf`` is the
            Gaussian, and is what every other row is compared against.
        **kwargs: passed through to ``fitter``.

    Returns:
        dict: ``best`` (the winning fit dict), ``gaussian`` (the ``nu = inf``
        fit), ``profile`` (a list of one row per ``nu``, each with ``nu``,
        ``log_likelihood``, ``sigma_mm`` and ``vs_gaussian``), ``best_nu``,
        ``best_vs_gaussian``, and ``identified``.

    ``identified`` says only that the winner is interior to the grid, not that
    the data pinned it down: a profile can peak in the middle by a fraction of a
    log-unit, which is noise. Read it together with ``best_vs_gaussian``, and
    with how flat the profile is around the peak. Scores are much less
    informative about ``nu`` than about the scale -- a few hundred darts will
    say confidently that there *is* a tail and only loosely how heavy.
    """
    fits = [fitter(*args, nu=float(nu), **kwargs) for nu in nu_grid]
    ll = [f["log_likelihood"] for f in fits]
    best = int(np.argmax(ll))
    gauss = next(i for i, nu in enumerate(nu_grid) if np.isinf(nu))
    profile = [{"nu": float(nu), "log_likelihood": f["log_likelihood"],
                "sigma_mm": f["sigma_mm"],
                "vs_gaussian": f["log_likelihood"] - ll[gauss]}
               for nu, f in zip(nu_grid, fits)]
    return {"best": fits[best], "gaussian": fits[gauss], "profile": profile,
            "best_nu": float(nu_grid[best]),
            "best_vs_gaussian": ll[best] - ll[gauss],
            "identified": 0 < best < len(nu_grid) - 1}


def simulate_session(design_mm, n_per_target, b, Sigma, board_pixels=256,
                     seed=0, board=None, nu=None):
    """
    Simulate a measurement session at several targets.

    Args:
        design_mm (sequence): (x, y) targets in millimetres.
        n_per_target (int or sequence): darts thrown at each target.
        b (array-like): the player's systematic bias in mm.
        Sigma (array-like): the player's covariance in mm^2, or scale if ``nu``
            is given.
        nu (float): throw a Student-t instead of a Gaussian.

    Returns:
        list: ``(target_mm, scores)`` pairs, ready for :func:`fit_multi_target`.
    """
    design_mm = [np.asarray(t, float) for t in design_mm]
    if np.isscalar(n_per_target):
        n_per_target = [int(n_per_target)] * len(design_mm)
    if board is None:
        board, _ = generate_dartboard(board_pixels)
    b = np.asarray(b, float)

    rng = np.random.default_rng(seed)
    out = []
    for t, n in zip(design_mm, n_per_target):
        scores = simulate_scores(n, t + b, Sigma, board=board, nu=nu,
                                 seed=int(rng.integers(1 << 31)))
        out.append((t, scores))
    return out


def bootstrap_uncertainty(scores, n_boot=40, seed=0, **kwargs):
    """
    Bootstrap standard errors for a score-only fit.

    Scores constrain the throwing distribution weakly, so this is not optional
    decoration -- it is how you find out whether a fit means anything.

    Returns:
        dict: ``sigma_mm`` and ``mu`` point estimates, their standard errors,
        and the raw bootstrap draws.
    """
    scores = np.asarray(scores)
    rng = np.random.default_rng(seed)
    board, _ = generate_dartboard(kwargs.get("board_pixels", 256))
    base = fit_from_scores(scores, board=board, **kwargs)

    draws = []
    for _ in range(n_boot):
        sample = rng.choice(scores, size=len(scores), replace=True)
        try:
            f = fit_from_scores(sample, board=board, **kwargs)
        except ValueError:
            continue
        draws.append([f["sigma_mm"], f["mu"][0], f["mu"][1]])
    draws = np.array(draws)
    return {"sigma_mm": base["sigma_mm"], "mu": base["mu"],
            "sigma_mm_se": float(draws[:, 0].std(ddof=1)),
            "mu_se": draws[:, 1:].std(axis=0, ddof=1),
            "draws": draws}


def simulate_scores(n, mu, Sigma, board_pixels=256, seed=0, board=None, nu=None):
    """
    Draw ``n`` scores from a known throwing distribution, for validating a fit.

    With ``nu`` set the throw is a Student-t, drawn the way the model describes
    it: a Gaussian dart whose width is redrawn each throw. A dart that lands
    beyond the board array scores 0, same as one that lands on the wire.
    """
    rng = np.random.default_rng(seed)
    if board is None:
        board, _ = generate_dartboard(board_pixels)
    pixels = board.shape[0]
    scale = mm_per_pixel(pixels)
    z = rng.multivariate_normal(np.zeros(2), np.asarray(Sigma, float), n)
    if nu is not None and not np.isinf(nu):
        z /= np.sqrt(rng.chisquare(nu, n) / nu)[:, None]
    z += np.asarray(mu, float)
    col = np.rint(z[:, 0] / scale).astype(int) + pixels // 2
    row = np.rint(z[:, 1] / scale).astype(int) + pixels // 2
    inside = (col >= 0) & (col < pixels) & (row >= 0) & (row < pixels)
    out = np.zeros(n, dtype=np.int64)
    out[inside] = board[row[inside], col[inside]].astype(np.int64)
    return out


def effective_sample_size(scores):
    """
    A crude warning signal: how many *distinct* scores were observed, and the
    entropy of the score histogram. A fit from throws that nearly all scored
    the same thing has very little to say about the spread.
    """
    scores = np.asarray(scores)
    _, counts = np.unique(scores, return_counts=True)
    p = counts / counts.sum()
    return {"n": len(scores), "distinct scores": len(counts),
            "entropy (nats)": float(-(p * np.log(p)).sum())}
