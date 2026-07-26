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

    def __init__(self, board_pixels=256, board=None, quadro=False):
        if board is None:
            board, _ = generate_dartboard(board_pixels, quadro=quadro)
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

    def _pdf(self, mu, Sigma):
        d = self.coords - mu
        det = Sigma[0, 0] * Sigma[1, 1] - Sigma[0, 1] * Sigma[1, 0]
        inv = np.array([[Sigma[1, 1], -Sigma[0, 1]], [-Sigma[1, 0], Sigma[0, 0]]]) / det
        q = np.einsum("ij,jk,ik->i", d, inv, d)
        return np.exp(-0.5 * q) / (2 * np.pi * np.sqrt(det))

    def score_probabilities(self, mu, Sigma):
        """P(score = s) for every board score, as a dict."""
        w = self._pdf(mu, Sigma) * self.pixel_area
        return {int(s): float(w[idx].sum()) for s, idx in self.index.items()}

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
             verbose=False):
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

    Args:
        em_step (callable): one EM step, mapping a packed parameter vector to
            the next.
        log_lik (callable): observed-data log-likelihood of a packed vector.
        theta (np.ndarray): starting parameters, packed.
        tol (float): relative log-likelihood change to stop at.
        max_iter (int): cap on outer iterations.
        accelerate (bool): set ``False`` for plain EM.
        verbose (bool): print progress.

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
                while alpha < -1.0 - 1e-12:
                    cand = em_step(theta - 2 * alpha * r + alpha ** 2 * v)
                    n_steps += 1
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
                    accelerate=True):
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

    Returns:
        dict: ``mu``, ``Sigma``, ``sigma_mm``, ``log_likelihood``,
        ``n_iter``, ``n_em_steps``, ``converged``, ``history``.
    """
    scores = np.asarray(scores, dtype=np.int64)
    like = ScoreLikelihood(board_pixels, board=board)
    unknown = set(int(s) for s in np.unique(scores)) - set(int(s) for s in like.scores)
    if unknown:
        raise ValueError(f"scores not achievable on this board: {sorted(unknown)}")

    counts = {int(s): int((scores == s).sum()) for s in np.unique(scores)}
    n = len(scores)

    def em_step(theta):
        (mu,), Sigma = _unpack(theta)
        # E step: exact conditional moments, one density evaluation for all
        # distinct scores
        moments = like.conditional_moments_all(mu, Sigma, counts)
        # M step: the Gaussian MLE using those moments
        ez = sum(counts[s] * moments[s][0] for s in counts) / n
        ezz = sum(counts[s] * moments[s][1] for s in counts) / n
        return _pack([ez], ezz - np.outer(ez, ez))

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
            "sigma_mm": float(np.sqrt(np.trace(Sigma) / 2)),
            "log_likelihood": history[-1], "n_iter": len(history) - 1,
            "n_em_steps": n_steps, "converged": converged, "history": history}


def fit_multi_target(sessions, board_pixels=256, b_init=None, Sigma_init=None,
                     tol=1e-10, max_iter=500, board=None, verbose=False,
                     shared_bias=True, accelerate=True):
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

    Returns:
        dict: ``b`` (or ``mu_by_target``), ``Sigma``, ``sigma_mm``,
        ``log_likelihood``, ``n_iter``, ``converged``, ``history``, ``n``.
    """
    like = ScoreLikelihood(board_pixels, board=board)
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
        sum_r = np.zeros(2)
        sum_rr = np.zeros((2, 2))
        per_target = []
        # E step: exact conditional moments of the residual R = Z - t at each
        # target, under that target's current mean.
        for i, (t, cnt) in enumerate(zip(targets, counts)):
            mu_i = t + offsets[0 if shared_bias else i]
            mom = like.conditional_moments_all(mu_i, Sigma, cnt)
            r_i = np.zeros(2)
            rr_i = np.zeros((2, 2))
            for s, k in cnt.items():
                ez, ezz = mom[s]
                # E[R R^T] = E[Z Z^T] - t E[Z]^T - E[Z] t^T + t t^T
                rr = ezz - np.outer(t, ez) - np.outer(ez, t) + np.outer(t, t)
                r_i += k * (ez - t)
                rr_i += k * rr
            per_target.append((r_i, sum(cnt.values())))
            sum_r += r_i
            sum_rr += rr_i

        # M step: the Gaussian MLE from those moments. With a shared bias the
        # whole session contributes to one mean; with free means each target
        # gets its own, and the spread is measured about each.
        if shared_bias:
            new = [sum_r / n]
            correction = np.outer(new[0], new[0])
        else:
            new = [r_i / k for r_i, k in per_target]
            correction = sum(k * np.outer(o, o)
                             for o, (_, k) in zip(new, per_target)) / n
        return _pack(new, sum_rr / n - correction)

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
           "log_likelihood": history[-1], "n_iter": len(history) - 1,
           "n_em_steps": n_steps, "converged": converged, "history": history,
           "n": n, "targets": np.array(targets)}
    if shared_bias:
        out["b"] = offsets[0]
    else:
        out["mu_by_target"] = offsets
    return out


def simulate_session(design_mm, n_per_target, b, Sigma, board_pixels=256,
                     seed=0, board=None):
    """
    Simulate a measurement session at several targets.

    Args:
        design_mm (sequence): (x, y) targets in millimetres.
        n_per_target (int or sequence): darts thrown at each target.
        b (array-like): the player's systematic bias in mm.
        Sigma (array-like): the player's covariance in mm^2.

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
        scores = simulate_scores(n, t + b, Sigma, board=board,
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


def simulate_scores(n, mu, Sigma, board_pixels=256, seed=0, board=None):
    """
    Draw ``n`` scores from a known throwing distribution, for validating a fit.
    """
    rng = np.random.default_rng(seed)
    if board is None:
        board, _ = generate_dartboard(board_pixels)
    pixels = board.shape[0]
    scale = mm_per_pixel(pixels)
    z = rng.multivariate_normal(np.asarray(mu, float), np.asarray(Sigma, float), n)
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
