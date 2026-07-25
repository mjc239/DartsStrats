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


def fit_from_scores(scores, board_pixels=256, mu_init=None, Sigma_init=None,
                    tol=1e-8, max_iter=500, board=None, verbose=False):
    """
    Fit ``(mu, Sigma)`` from observed scores alone, by exact EM.

    Every throw is assumed aimed at the same (unknown) point ``mu`` -- i.e. this
    is for a practice session at one target, not a bag of match darts.

    Args:
        scores (sequence[int]): the observed dart scores.
        board_pixels (int): resolution of the board used for the likelihood.
            The E step is a sum over pixels, so this is the accuracy knob.
        mu_init, Sigma_init: starting values; defaults to the centre of the
            board and a 30mm isotropic spread.
        tol (float): relative change in log-likelihood to stop at.
        max_iter (int): iteration cap.
        board (np.ndarray): a prebuilt board array, to avoid rebuilding it.
        verbose (bool): print the likelihood each iteration.

    Returns:
        dict: ``mu``, ``Sigma``, ``sigma_mm``, ``log_likelihood``,
        ``n_iter``, ``converged``, ``history`` (the log-likelihood trace).
    """
    scores = np.asarray(scores, dtype=np.int64)
    like = ScoreLikelihood(board_pixels, board=board)
    unknown = set(int(s) for s in np.unique(scores)) - set(int(s) for s in like.scores)
    if unknown:
        raise ValueError(f"scores not achievable on this board: {sorted(unknown)}")

    counts = {int(s): int((scores == s).sum()) for s in np.unique(scores)}
    n = len(scores)

    mu = np.zeros(2) if mu_init is None else np.asarray(mu_init, float)
    Sigma = 30.0 ** 2 * np.eye(2) if Sigma_init is None else np.asarray(Sigma_init, float)

    history = []
    converged = False
    for it in range(max_iter):
        # E step: exact conditional moments, cached per distinct score
        ez_cache, ezz_cache = {}, {}
        for s in counts:
            ez_cache[s], ezz_cache[s] = like.conditional_moments(mu, Sigma, s)

        # M step: the Gaussian MLE using those moments
        ez = sum(counts[s] * ez_cache[s] for s in counts) / n
        ezz = sum(counts[s] * ezz_cache[s] for s in counts) / n
        mu = ez
        Sigma = ezz - np.outer(ez, ez)
        # keep it positive definite against round-off
        Sigma = 0.5 * (Sigma + Sigma.T)
        w, V = np.linalg.eigh(Sigma)
        Sigma = V @ np.diag(np.maximum(w, 1e-6)) @ V.T

        ll = like.log_likelihood(mu, Sigma, counts)
        history.append(ll)
        if verbose:
            print(f"  iter {it:>3}: loglik {ll:.6f}  mu {mu.round(2)}  "
                  f"sigma {np.sqrt(np.trace(Sigma) / 2):.2f}mm")
        if it > 0 and abs(ll - history[-2]) <= tol * max(1.0, abs(history[-2])):
            converged = True
            break

    return {"mu": mu, "Sigma": Sigma,
            "sigma_mm": float(np.sqrt(np.trace(Sigma) / 2)),
            "log_likelihood": history[-1], "n_iter": len(history),
            "converged": converged, "history": history}


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
