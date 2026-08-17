"""
A throwing model with shape and aim error: ``Z ~ N(t + b, Sigma)``.

:mod:`darts.bayes` carries a posterior over a single isotropic ``sigma``, with
the bias assumed zero. Both are simplifications that the notebooks flagged as
the next things to relax, and they relax very differently:

**Anisotropy** (``Sigma`` not a multiple of the identity) changes the *board*.
A player whose scatter is tall and narrow misses differently around a treble
than around a side double, so the value function itself changes and every
candidate ``Sigma`` needs its own MDP solve. This is the expensive direction.

**Bias** (``b`` not zero) does not change the board at all, only where you
stand relative to it. Aiming at ``a`` under bias ``b`` lands you at ``a + b``,
which is exactly the unbiased outcome of aiming at ``a + b``. So the biased
game is the unbiased game with the action set translated -- same values, same
shots, relabelled. A *known* bias costs nothing but aiming off; the whole cost
of a bias is the cost of not knowing it. That makes this the cheap direction,
and it is why combining the two is no more expensive in MDP solves than
anisotropy alone.

:func:`shift_indices` implements the relabelling, and the notebooks check the
claim numerically rather than taking it on trust -- the translated action set
is not *identical* to the original (the grid is finite and points near the rim
shift off it), so the theorem holds up to discretisation and boundary, and how
much is a question with a number.
"""

import numpy as np

from darts.dartboards import generate_dartboard
from darts.mdp_3turn import ThreeDartMDP
from darts.transitions import transition_arrays
from darts.utils import mm_per_pixel


# --------------------------------------------------------------------------
# Building the parameter grids
# --------------------------------------------------------------------------

def sigma_matrices(sx_values, sy_values, rho_values=(0.0,)):
    """
    A grid of covariance matrices from marginal spreads and correlations.

    Args:
        sx_values, sy_values (sequence): standard deviations in mm along the
            board's x (horizontal) and y (vertical) axes.
        rho_values (sequence): correlations, for a tilted ellipse.

    Returns:
        tuple: ``(Sigmas, labels)`` with ``Sigmas`` of shape (K, 2, 2) and
        ``labels`` a (K, 3) array of ``(sigma_x, sigma_y, rho)``.
    """
    mats, labels = [], []
    for sx in sx_values:
        for sy in sy_values:
            for r in rho_values:
                c = float(r) * float(sx) * float(sy)
                mats.append([[float(sx) ** 2, c], [c, float(sy) ** 2]])
                labels.append([float(sx), float(sy), float(r)])
    return np.array(mats), np.array(labels)


def bias_grid(x_values, y_values):
    """A grid of aim biases in mm, as an (M, 2) array of (b_x, b_y)."""
    xs, ys = np.meshgrid(np.asarray(x_values, float),
                         np.asarray(y_values, float), indexing="ij")
    return np.stack([xs.ravel(), ys.ravel()], axis=1)


def isotropic_equivalent(Sigma):
    """``sqrt(tr(Sigma) / 2)``, the single sigma this project quotes."""
    Sigma = np.asarray(Sigma, float)
    return float(np.sqrt(np.trace(Sigma) / 2))


def shape_prior(Sigmas, band, spread=1.4, ratio_spread=1.35):
    """
    Prior over covariance matrices from a named ability band.

    Factorises into *size* and *shape*: a lognormal on the isotropic-equivalent
    sigma centred on the band (as in :func:`darts.bayes.band_prior`), times a
    lognormal on the axis ratio ``sigma_y / sigma_x`` centred on 1. The shape
    prior is deliberately wide -- claiming to be a club player says something
    about how big your group is and nothing about whether it is round.
    """
    from darts import players
    centre = players.ABILITY_BANDS[band] if isinstance(band, str) else float(band)
    Sigmas = np.asarray(Sigmas, float)
    size = np.array([isotropic_equivalent(S) for S in Sigmas])
    ratio = np.sqrt(Sigmas[:, 1, 1] / Sigmas[:, 0, 0])
    z_size = (np.log(size) - np.log(centre)) / np.log(spread)
    z_ratio = np.log(ratio) / np.log(ratio_spread)
    p = np.exp(-0.5 * (z_size ** 2 + z_ratio ** 2))
    return p / p.sum()


def bias_prior(biases, sd_mm=8.0):
    """
    Prior over aim bias: centred on zero, since a player is trying to hit what
    they aim at, with ``sd_mm`` of slack -- about half a treble bed's length,
    which is a generous allowance for a systematic pull.
    """
    biases = np.asarray(biases, float)
    p = np.exp(-0.5 * (biases ** 2).sum(axis=1) / sd_mm ** 2)
    return p / p.sum()


# --------------------------------------------------------------------------
# The posterior
# --------------------------------------------------------------------------

class _BoardPixels:
    """
    The board, precomputed into what a per-dart likelihood needs: the
    millimetre coordinates of every pixel, indexed by the score in it.

    Shared by the grid posterior and the particle filter, which differ only in
    how they carry the belief, not in how a dart is scored.
    """

    def __init__(self, board_pixels=256, board=None, checkouts=None):
        if board is None:
            board, checkouts = generate_dartboard(board_pixels)
        elif checkouts is None:
            raise ValueError("pass checkouts along with a prebuilt board")
        self.board, self.checkouts = board, checkouts
        self.pixels = board.shape[0]
        self.mm_per_pixel = mm_per_pixel(self.pixels)

        offs = (np.arange(self.pixels) - self.pixels // 2) * self.mm_per_pixel
        x, y = np.meshgrid(offs, offs)          # x along columns, y along rows
        self._coords = np.stack([x.ravel(), y.ravel()], axis=1)
        flat = board.ravel().astype(np.int64)
        co = checkouts.ravel().astype(bool)
        self._index = {int(s): np.flatnonzero(flat == s) for s in np.unique(flat)}
        self._co_index = {int(s): np.flatnonzero((flat == s) & co)
                          for s in np.unique(flat[co])}

    def point_to_mm(self, point):
        """(row, col) pixel -> (x, y) mm from the centre."""
        c = self.pixels // 2
        return np.array([(point[1] - c) * self.mm_per_pixel,
                         (point[0] - c) * self.mm_per_pixel])

    def _pixels_for(self, score, checkout):
        idx = (self._co_index if checkout else self._index).get(int(score))
        if idx is None or len(idx) == 0:
            raise ValueError(f"score {score} (checkout={checkout}) unreachable")
        return idx


class ThrowPosterior(_BoardPixels):
    """
    A discrete posterior over ``(Sigma, b)``, updated one dart at a time.

    The grid is the outer product of a covariance grid and a bias grid, so the
    log posterior is a (K, M) array and the marginals are sums along an axis.

    The per-dart likelihood is exact on the pixel grid: for observed score
    value ``s`` thrown at ``t``,

        p(s | t, Sigma, b) = sum over pixels scoring s of N(t + b, Sigma)

    Exact, and priced accordingly: the cost is the number of grid points times
    the pixels of the observed score, and a product grid over ``d`` parameters
    has ``n**d`` points. Past about four parameters use
    :class:`ParticleThrowPosterior` instead.
    """

    def __init__(self, Sigmas, biases=None, prior=None, board_pixels=256,
                 board=None, checkouts=None):
        super().__init__(board_pixels, board, checkouts)
        self.Sigmas = np.asarray(Sigmas, float).reshape(-1, 2, 2)
        self.biases = (np.zeros((1, 2)) if biases is None
                       else np.asarray(biases, float).reshape(-1, 2))

        # per-Sigma inverse entries and normalisation, precomputed once
        det = (self.Sigmas[:, 0, 0] * self.Sigmas[:, 1, 1]
               - self.Sigmas[:, 0, 1] * self.Sigmas[:, 1, 0])
        if (det <= 0).any():
            raise ValueError("every Sigma must be positive definite")
        self._a00 = self.Sigmas[:, 1, 1] / det
        self._a01 = -self.Sigmas[:, 0, 1] / det
        self._a11 = self.Sigmas[:, 0, 0] / det
        self._lognorm = -np.log(2 * np.pi * np.sqrt(det))
        self._logarea = 2 * np.log(self.mm_per_pixel)

        K, M = len(self.Sigmas), len(self.biases)
        if prior is None:
            lp = np.zeros((K, M))
        else:
            lp = np.log(np.asarray(prior, float).reshape(K, M))
        self.log_prior = lp - lp.max()
        self.log_post = self.log_prior.copy()
        self.n_updates = 0

    # -- likelihood ---------------------------------------------------------
    def _log_likelihood(self, aim_mm, score, checkout=False, max_bytes=64 << 20):
        """
        (K, M) log likelihood of one dart.

        Chunked over the covariance axis: a joint grid of a few hundred
        covariances by a hundred biases, against a few thousand pixels of a
        score, is a gigabyte if formed at once.
        """
        idx = self._pixels_for(score, checkout)
        # d[m, n] = pixel_n - (aim + bias_m)
        centres = np.asarray(aim_mm, float)[None, :] + self.biases      # (M, 2)
        dx = self._coords[idx, 0][None, :] - centres[:, 0][:, None]     # (M, N)
        dy = self._coords[idx, 1][None, :] - centres[:, 1][:, None]
        dxx, dxy, dyy = dx ** 2, dx * dy, dy ** 2

        K = len(self.Sigmas)
        per_k = max(dx.size * 8, 1)
        step = max(1, min(K, int(max_bytes // per_k)))
        out = np.empty((K, len(self.biases)))
        for lo in range(0, K, step):
            hi = min(lo + step, K)
            q = (self._a00[lo:hi, None, None] * dxx[None]
                 + 2 * self._a01[lo:hi, None, None] * dxy[None]
                 + self._a11[lo:hi, None, None] * dyy[None])
            m = -0.5 * q + self._lognorm[lo:hi, None, None] + self._logarea
            top = m.max(axis=2, keepdims=True)
            out[lo:hi] = top[:, :, 0] + np.log(np.exp(m - top).sum(axis=2))
        return out

    def update(self, aim, score, checkout=False, pixel=True):
        """Fold in one dart. ``aim`` is (row, col) pixels, or (x, y) mm."""
        aim_mm = self.point_to_mm(aim) if pixel else np.asarray(aim, float)
        self.log_post = self.log_post + self._log_likelihood(aim_mm, score, checkout)
        self.log_post -= self.log_post.max()
        self.n_updates += 1
        return self

    # -- summaries ----------------------------------------------------------
    @property
    def probs(self):
        p = np.exp(self.log_post - self.log_post.max())
        return p / p.sum()

    def marginal_sigma(self):
        """Posterior over the covariance grid, marginalising out the bias."""
        return self.probs.sum(axis=1)

    def marginal_bias(self):
        """Posterior over the bias grid, marginalising out the covariance."""
        return self.probs.sum(axis=0)

    def mean_sigma_xy(self):
        """Posterior mean of ``(sigma_x, sigma_y)``."""
        w = self.marginal_sigma()
        return (float(w @ np.sqrt(self.Sigmas[:, 0, 0])),
                float(w @ np.sqrt(self.Sigmas[:, 1, 1])))

    def mean_ratio(self):
        """Posterior mean of the axis ratio ``sigma_y / sigma_x``."""
        w = self.marginal_sigma()
        return float(w @ np.sqrt(self.Sigmas[:, 1, 1] / self.Sigmas[:, 0, 0]))

    def mean_isotropic(self):
        w = self.marginal_sigma()
        return float(w @ np.array([isotropic_equivalent(S) for S in self.Sigmas]))

    def mean_bias(self):
        return self.marginal_bias() @ self.biases

    def sd_bias(self):
        w = self.marginal_bias()
        m = w @ self.biases
        return np.sqrt(w @ (self.biases - m) ** 2)

    def sd_ratio(self):
        w = self.marginal_sigma()
        r = np.sqrt(self.Sigmas[:, 1, 1] / self.Sigmas[:, 0, 0])
        m = w @ r
        return float(np.sqrt(w @ (r - m) ** 2))

    def map_estimate(self):
        k, m = np.unravel_index(int(np.argmax(self.log_post)), self.log_post.shape)
        return self.Sigmas[k], self.biases[m]


# --------------------------------------------------------------------------
# Acting on it
# --------------------------------------------------------------------------

class ParticleThrowPosterior(_BoardPixels):
    """
    The same belief over ``(Sigma, b)``, carried by particles instead of a grid.

    :class:`ThrowPosterior` is exact and costs ``n**d`` grid points, which is
    fine for an evening's analysis and runs out of room at about four
    parameters -- a second a dart, with no headroom and a fifth parameter
    costing another factor of five. This is the version for advising a player
    between darts.

    It fixes *both* halves of that second. The belief update is over
    ``n_particles`` points rather than the whole grid; and because the
    recommender averages Q-values over whatever support the belief offers, the
    particles group into a few dozen distinct (covariance, shift) pairs and
    shrink that sum too. Replacing the grid with a small support set is the
    fix; a Laplace approximation would have addressed only the first half.

    **Static parameters need rejuvenation.** A throw's shape does not evolve
    between darts, so plain sequential importance resampling degenerates: the
    particles collapse onto a handful of duplicated values and the posterior
    reports false certainty. This uses the Liu-West filter -- after resampling,
    each particle is shrunk toward the weighted mean and jittered, in a way
    that preserves the first two moments of the belief. ``delta`` sets the
    trade-off: near 1 is faithful and slow to explore, lower diversifies harder.

    Parameters live in a transformed space -- ``log sigma_x``, ``log sigma_y``,
    ``atanh rho``, ``b_x``, ``b_y`` -- so that jitter can never produce a
    negative spread or a correlation outside (-1, 1).

    ``drift`` is the hook the notebooks keep asking for: a per-dart random walk
    on the parameters, which turns the filter from an estimator of a fixed
    player into a tracker of a changing one. Left at zero it is an estimator.
    """

    N_PARAM = 5          # log sx, log sy, atanh rho, bx, by

    def __init__(self, n_particles=500, band="league", tilt=False,
                 board_pixels=256, board=None, checkouts=None, rng=None,
                 delta=0.98, ess_fraction=0.5, drift=0.0, particles=None,
                 sigma_spread=1.4, ratio_spread=1.35, bias_sd=8.0):
        super().__init__(board_pixels, board, checkouts)
        self.rng = np.random.default_rng() if rng is None else rng
        self.delta = float(delta)
        self.ess_fraction = float(ess_fraction)
        self.drift = np.zeros(self.N_PARAM) + np.asarray(drift, float)
        self.tilt = bool(tilt)
        self.n_updates = 0
        self.n_resamples = 0

        if particles is not None:
            self._theta = np.asarray(particles, float).reshape(-1, self.N_PARAM)
        else:
            self._theta = self._sample_prior(int(n_particles), band,
                                             sigma_spread, ratio_spread, bias_sd)
        n = len(self._theta)
        self.log_w = np.full(n, -np.log(n))

    # -- prior --------------------------------------------------------------
    def _sample_prior(self, n, band, sigma_spread, ratio_spread, bias_sd):
        """
        Draw from the same prior the grid uses: lognormal on the overall size
        centred on the named band, lognormal on the axis ratio centred on
        round, and a Gaussian on the bias centred on no pull.
        """
        from darts import players
        centre = (players.ABILITY_BANDS[band] if isinstance(band, str)
                  else float(band))
        size = np.log(centre) + np.log(sigma_spread) * self.rng.standard_normal(n)
        lr = np.log(ratio_spread) * self.rng.standard_normal(n)
        # size is the isotropic equivalent; split it into the two axes by the
        # ratio, holding sqrt(tr/2) at the drawn size
        r = np.exp(lr)
        scale = np.sqrt(2.0 / (1.0 + r ** 2))
        log_sx = size + np.log(scale)
        log_sy = log_sx + lr
        rho = (np.arctanh(np.clip(0.3 * self.rng.standard_normal(n), -0.9, 0.9))
               if self.tilt else np.zeros(n))
        b = bias_sd * self.rng.standard_normal((n, 2))
        return np.column_stack([log_sx, log_sy, rho, b])

    # -- the belief, in useful coordinates ----------------------------------
    @property
    def weights(self):
        w = np.exp(self.log_w - self.log_w.max())
        return w / w.sum()

    @property
    def sigma_xy(self):
        return np.exp(self._theta[:, :2])

    @property
    def rho(self):
        return np.tanh(self._theta[:, 2])

    @property
    def biases(self):
        return self._theta[:, 3:5]

    @property
    def Sigmas(self):
        sx, sy = self.sigma_xy[:, 0], self.sigma_xy[:, 1]
        c = self.rho * sx * sy
        out = np.empty((len(sx), 2, 2))
        out[:, 0, 0] = sx ** 2
        out[:, 1, 1] = sy ** 2
        out[:, 0, 1] = out[:, 1, 0] = c
        return out

    def ess(self):
        """Effective sample size: how many particles are really contributing."""
        w = self.weights
        return float(1.0 / (w ** 2).sum())

    # -- likelihood and update ----------------------------------------------
    def _log_likelihood(self, aim_mm, score, checkout=False):
        """(n_particles,) log likelihood of one dart, each at its own theta."""
        idx = self._pixels_for(score, checkout)
        z = self._coords[idx]                                   # (N, 2)
        centres = np.asarray(aim_mm, float)[None, :] + self.biases   # (P, 2)
        dx = z[None, :, 0] - centres[:, 0][:, None]             # (P, N)
        dy = z[None, :, 1] - centres[:, 1][:, None]

        sx, sy = self.sigma_xy[:, 0], self.sigma_xy[:, 1]
        r = self.rho
        det = (sx * sy) ** 2 * (1.0 - r ** 2)
        a00 = (sy ** 2) / det
        a11 = (sx ** 2) / det
        a01 = -(r * sx * sy) / det
        q = (a00[:, None] * dx ** 2 + 2 * a01[:, None] * dx * dy
             + a11[:, None] * dy ** 2)
        m = (-0.5 * q - np.log(2 * np.pi * np.sqrt(det))[:, None]
             + 2 * np.log(self.mm_per_pixel))
        top = m.max(axis=1, keepdims=True)
        return top[:, 0] + np.log(np.exp(m - top).sum(axis=1))

    def update(self, aim, score, checkout=False, pixel=True):
        """Fold in one dart, resampling and rejuvenating when needed."""
        aim_mm = self.point_to_mm(aim) if pixel else np.asarray(aim, float)
        if self.drift.any():
            self._theta = self._theta + self.drift * self.rng.standard_normal(
                self._theta.shape)
            if not self.tilt:
                self._theta[:, 2] = 0.0
        self.log_w = self.log_w + self._log_likelihood(aim_mm, score, checkout)
        self.log_w -= self.log_w.max()
        self.n_updates += 1
        if self.ess() < self.ess_fraction * len(self._theta):
            self._resample()
        return self

    def _resample(self):
        """Systematic resampling, then a Liu-West shrink-and-jitter."""
        w = self.weights
        n = len(w)
        # systematic resampling: lower variance than multinomial
        u = (self.rng.random() + np.arange(n)) / n
        idx = np.searchsorted(np.cumsum(w), u)
        idx = np.clip(idx, 0, n - 1)

        mean = w @ self._theta
        centred = self._theta - mean
        cov = (centred * w[:, None]).T @ centred
        a = (3 * self.delta - 1) / (2 * self.delta)
        h2 = 1.0 - a ** 2

        shrunk = a * self._theta[idx] + (1 - a) * mean
        try:
            L = np.linalg.cholesky(h2 * cov + 1e-12 * np.eye(self.N_PARAM))
        except np.linalg.LinAlgError:
            L = np.diag(np.sqrt(np.maximum(h2 * np.diag(cov), 1e-12)))
        self._theta = shrunk + self.rng.standard_normal((n, self.N_PARAM)) @ L.T
        if not self.tilt:
            self._theta[:, 2] = 0.0
        self.log_w = np.full(n, -np.log(n))
        self.n_resamples += 1

    # -- summaries, matching ThrowPosterior ---------------------------------
    def mean_sigma_xy(self):
        w = self.weights
        s = self.sigma_xy
        return float(w @ s[:, 0]), float(w @ s[:, 1])

    def mean_ratio(self):
        w = self.weights
        s = self.sigma_xy
        return float(w @ (s[:, 1] / s[:, 0]))

    def sd_ratio(self):
        w = self.weights
        r = self.sigma_xy[:, 1] / self.sigma_xy[:, 0]
        m = w @ r
        return float(np.sqrt(w @ (r - m) ** 2))

    def mean_isotropic(self):
        w = self.weights
        s = self.sigma_xy
        return float(w @ np.sqrt((s[:, 0] ** 2 + s[:, 1] ** 2) / 2))

    def mean_bias(self):
        return self.weights @ self.biases

    def sd_bias(self):
        w = self.weights
        m = w @ self.biases
        return np.sqrt(w @ (self.biases - m) ** 2)

    def mean_rho(self):
        return float(self.weights @ self.rho)

    def map_estimate(self):
        i = int(np.argmax(self.log_w))
        return self.Sigmas[i], self.biases[i]

    # -- what the recommender needs -----------------------------------------
    def support_groups(self, recommender, bias_step=None):
        """
        Collapse the particles to ``(k, bias, weight)`` triples.

        Two quantisations, both of which the recommender imposes anyway: each
        particle's covariance is mapped to the nearest *solved* one, and its
        bias is snapped to the aiming grid, since an aim point can only be
        displaced by whole grid steps. Particles agreeing on both are merged,
        which is what makes the Q-average cheap.
        """
        if bias_step is None:
            pts = recommender.points
            rows = np.unique(pts[:, 0])
            step_px = float(np.min(np.diff(rows))) if len(rows) > 1 else 1.0
            bias_step = step_px * recommender.mm_per_pixel

        d = ((self.Sigmas[:, None] - recommender.Sigmas[None]) ** 2).sum(axis=(2, 3))
        k = np.argmin(d, axis=1)
        snapped = np.rint(self.biases / bias_step).astype(int)
        w = self.weights

        groups = {}
        for i in range(len(w)):
            key = (int(k[i]), int(snapped[i, 0]), int(snapped[i, 1]))
            groups[key] = groups.get(key, 0.0) + w[i]
        return [(kk, np.array([bx, by]) * bias_step, wt)
                for (kk, bx, by), wt in groups.items() if wt >= 1e-12]


def shift_indices(points, shift_px, pixels):
    """
    Relabel actions under a bias: for each candidate point ``p``, the index of
    the candidate nearest ``p + shift``.

    This is the shift theorem made concrete. Points whose shifted position
    leaves the candidate set are mapped to the nearest surviving one, which is
    the boundary effect that stops the theorem being exact.

    Args:
        points (np.ndarray): (n, 2) candidate points, in (row, col) pixels.
        shift_px (array-like): (row, col) shift in pixels.
        pixels (int): board resolution, for bounds.

    Returns:
        tuple: ``(index, exact)`` -- the relabelling, and a boolean array
        marking the points whose shifted position landed on a real candidate
        rather than being snapped in from outside.
    """
    from scipy.spatial import cKDTree
    pts = np.asarray(points, float)
    tree = cKDTree(pts)
    target = pts + np.asarray(shift_px, float)[None, :]
    dist, idx = tree.query(target, k=1)
    # a shifted point is "exact" if it landed within half a grid step
    step = np.min(np.diff(np.unique(pts[:, 0]))) if len(np.unique(pts[:, 0])) > 1 else 1.0
    return idx.astype(int), dist <= step / 2 + 1e-9


class ShapeRecommender:
    """
    Recommendations under uncertainty about both the shape of the throw and
    the aim bias.

    One MDP is solved per covariance on the grid, all sharing a candidate-point
    set. Bias enters by relabelling actions rather than by re-solving, so the
    cost is set entirely by the covariance grid: adding bias to an anisotropy
    study is close to free.
    """

    def __init__(self, Sigmas, board_pixels=256, point_stride=2, margin_mm=10.0,
                 game_start=501, progress=False):
        self.Sigmas = np.asarray(Sigmas, float).reshape(-1, 2, 2)
        self.game_start = game_start
        self.board_pixels = board_pixels
        self.models, self.tr = [], None
        for S in self.Sigmas:
            tr = transition_arrays(board_pixels, 0.0, margin_mm=margin_mm,
                                   point_stride=point_stride, Sigma_mm=S)
            if self.tr is None:
                self.tr = tr
            elif not np.array_equal(tr["points"], self.tr["points"]):
                raise AssertionError("candidate grids differ between covariances")
            self.models.append(
                ThreeDartMDP(tr["probs"], tr["checkout_probs"],
                             tr["allowed_scores"], game_start,
                             dart_cost=0.0, turn_cost=1.0).solve())
            if progress:
                print(f"  solved sigma_x={np.sqrt(S[0,0]):.1f} "
                      f"sigma_y={np.sqrt(S[1,1]):.1f}", flush=True)
        self.points = self.tr["points"]
        self.board, self.checkouts = self.tr["board"], self.tr["checkouts"]
        self.mm_per_pixel = self.tr["mm_per_pixel"]
        self._shift_cache = {}
        self._assign_cache = {}

    # -- bias relabelling ---------------------------------------------------
    def shift_for(self, bias_mm):
        """Cached action relabelling for a bias, in millimetres."""
        key = (round(float(bias_mm[0]), 6), round(float(bias_mm[1]), 6))
        if key not in self._shift_cache:
            # bias (x, y) mm -> (row, col) pixels: row is y, col is x
            shift_px = np.array([key[1], key[0]]) / self.mm_per_pixel
            self._shift_cache[key] = shift_indices(self.points, shift_px,
                                                   self.board_pixels)
        return self._shift_cache[key]

    def nearest_sigma(self, Sigma):
        """Index of the solved covariance closest to ``Sigma`` (Frobenius)."""
        d = ((self.Sigmas - np.asarray(Sigma, float)[None]) ** 2).sum(axis=(1, 2))
        return int(np.argmin(d))

    def q_biased(self, k, bias_mm, score, dart, round_start=None):
        """
        Q-values over *actions* for covariance ``k`` under bias ``bias_mm``.

        Aiming at ``a`` lands where aiming at ``a + b`` would land with no
        bias, so the Q-value of action ``a`` is the unbiased Q at ``a + b``.
        """
        q = self.models[k].q_values(score, dart, round_start)
        if abs(bias_mm[0]) < 1e-12 and abs(bias_mm[1]) < 1e-12:
            return q
        idx, _ = self.shift_for(bias_mm)
        return q[idx]

    def weights_from(self, posterior):
        """
        Aggregate a posterior onto the solved covariance grid.

        The posterior may live on a finer grid than the MDPs were solved on --
        learning the shape of a throw is cheap, solving for it is not -- so
        mass is collected onto the nearest solved covariance. The bias axis
        passes through untouched, since bias is handled exactly by relabelling
        rather than by solving.
        """
        p = posterior.probs
        if len(posterior.Sigmas) == len(self.Sigmas) and np.allclose(
                posterior.Sigmas, self.Sigmas):
            return p
        key = id(posterior.Sigmas)
        assign = self._assign_cache.get(key)
        if assign is None:
            d = ((posterior.Sigmas[:, None] - self.Sigmas[None]) ** 2).sum(
                axis=(2, 3))
            assign = np.argmin(d, axis=1)
            self._assign_cache[key] = assign
        out = np.zeros((len(self.Sigmas), p.shape[1]))
        np.add.at(out, assign, p)
        return out

    def support(self, posterior, weights=None):
        """
        The belief, reduced to what averaging Q-values actually needs:
        ``(k, bias, weight)`` triples over solved covariances.

        Both posteriors answer this, which is what lets the same recommender
        drive either. The grid enumerates its cells; the particle filter groups
        its particles, and that grouping is why the filter speeds up the
        *recommendation* as well as the belief update -- a few hundred
        particles collapse to a few dozen distinct (covariance, shift) pairs.
        """
        if hasattr(posterior, "support_groups"):
            return posterior.support_groups(self)
        w = self.weights_from(posterior) if weights is None else weights
        out = []
        for k in range(len(self.Sigmas)):
            if w[k].sum() < 1e-12:
                continue
            for m, wm in enumerate(w[k]):
                if wm >= 1e-12:
                    out.append((k, posterior.biases[m], float(wm)))
        return out

    def qbar(self, posterior, score, dart, round_start=None, weights=None):
        """Posterior-weighted Q over actions, averaging over Sigma and bias."""
        groups = self.support(posterior, weights)
        out, base, base_k = None, None, -1
        for k, b, wm in sorted(groups, key=lambda g: g[0]):
            if k != base_k:
                base = self.models[k].q_values(score, dart, round_start)
                base_k = k
            if abs(b[0]) < 1e-12 and abs(b[1]) < 1e-12:
                q = base
            else:
                idx, _ = self.shift_for(b)
                q = base[idx]
            out = wm * q if out is None else out + wm * q
        return out

    def recommend(self, posterior, score, dart, round_start=None):
        return int(np.argmax(self.qbar(posterior, score, dart, round_start)))

    def oracle_action(self, Sigma, bias_mm, score, dart, round_start=None):
        k = self.nearest_sigma(Sigma)
        return int(np.argmax(self.q_biased(k, bias_mm, score, dart, round_start)))

    def value_loss(self, Sigma, bias_mm, action, score, dart, round_start=None):
        """Expected visits lost by ``action`` against the best action for the
        true ``(Sigma, bias)``."""
        k = self.nearest_sigma(Sigma)
        q = self.q_biased(k, bias_mm, score, dart, round_start)
        return float(q.max() - q[action])


def play_leg(recommender, posterior, true_Sigma, true_bias, rng,
             game_start=None, policy="bayes", record=None):
    """
    Simulate a leg under a general ``N(t + b, Sigma)`` throw.

    Args:
        policy: ``"bayes"`` (posterior-weighted over Sigma and bias),
            ``"oracle"`` (the true parameters), ``"isotropic"`` (the
            isotropic-equivalent sigma, no bias -- what the earlier model
            would do), or ``"nobias"`` (the true Sigma but bias assumed zero).
    """
    board, checkouts = recommender.board, recommender.checkouts
    px = board.shape[0]
    mmpp = recommender.mm_per_pixel
    game_start = game_start or recommender.game_start
    true_Sigma = np.asarray(true_Sigma, float)
    true_bias = np.asarray(true_bias, float)
    L = np.linalg.cholesky(true_Sigma)

    if isinstance(policy, tuple):
        # an explicit belief: play as if the throw were (Sigma, bias)
        S_belief, b_belief = policy
        k_fixed = recommender.nearest_sigma(np.asarray(S_belief, float))
        b_fixed = np.asarray(b_belief, float)
    elif policy == "isotropic":
        s = isotropic_equivalent(true_Sigma)
        k_fixed = recommender.nearest_sigma(s ** 2 * np.eye(2))
        b_fixed = np.zeros(2)
    elif policy == "nobias":
        k_fixed = recommender.nearest_sigma(true_Sigma)
        b_fixed = np.zeros(2)
    elif policy == "oracle":
        k_fixed = recommender.nearest_sigma(true_Sigma)
        b_fixed = true_bias
    else:
        k_fixed = None

    score, darts = game_start, 0
    while score > 0:
        round_start = score
        for dart in (1, 2, 3):
            if k_fixed is None:
                a = recommender.recommend(posterior, score, dart, round_start)
            else:
                a = int(np.argmax(recommender.q_biased(
                    k_fixed, b_fixed, score, dart, round_start)))
            aim = recommender.points[a]

            land = (posterior.point_to_mm(aim) + true_bias
                    + L @ rng.standard_normal(2))
            col = int(round(land[0] / mmpp)) + px // 2
            row = int(round(land[1] / mmpp)) + px // 2
            if 0 <= row < px and 0 <= col < px:
                v, on_double = int(board[row, col]), bool(checkouts[row, col])
            else:
                v, on_double = 0, False
            darts += 1

            finished = (score - v == 0) and on_double
            bust = (not finished) and (score - v <= 1)
            posterior.update(aim, v, checkout=finished)
            if record is not None:
                record.append({
                    "dart": darts, "state_score": score, "state_dart": dart,
                    "aim_row": int(aim[0]), "aim_col": int(aim[1]), "scored": v,
                    "sigma_x": posterior.mean_sigma_xy()[0],
                    "sigma_y": posterior.mean_sigma_xy()[1],
                    "ratio": posterior.mean_ratio(),
                    "bias_x": posterior.mean_bias()[0],
                    "bias_y": posterior.mean_bias()[1],
                    "value_loss": recommender.value_loss(
                        true_Sigma, true_bias, a, score, dart, round_start),
                })
            if finished:
                return darts
            if bust:
                score = round_start
                break
            score -= v
    return darts
