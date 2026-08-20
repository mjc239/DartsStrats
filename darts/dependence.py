"""
Breaking the independence assumption: models for what couples the darts of a visit.

Every solver in this project treats a visit as three i.i.d. draws around a fixed
aim point. Notebook 19 measured that assumption failing on real competition
darts, and notebook 20 asks what to replace it with. This module holds the model
family and the machinery to fit it to *observed beds* -- which is what real data
carries, since a scoresheet records where a dart landed but never where it was
aimed.

Two quite different things can couple the darts of a visit, and they need to be
separated before either can be measured.

**The aim rule.** A player who misses the treble 20 often moves to the treble 19
for the rest of the visit. That is a decision, not a throw: it changes where the
dart is aimed, and it is invisible in a per-dart score. It manufactures exactly
the correlation notebook 19 measured -- "missed, so the next dart was not even
thrown at the treble 20" -- without any statement about the throw itself. The
MDP has no state for it, because the aim point in this project depends on the
score and the dart index and nothing else.

**The throw coupling.** Whatever remains once the aim is accounted for: the
stance, the grip, the rhythm a player has for those three darts. Two shapes of
it are worth telling apart, and they predict different things:

* a shared *location* offset -- the darts of a visit scatter around a point that
  is itself drawn per visit, so they cluster in the same **direction**;
* a shared *scale* -- the player is tight or loose for a whole visit, so the
  darts agree in **magnitude** but not in direction.

The models below are the cross of an aim rule with a throw coupling, so that
"how much dependence is left once the aim rule is in" is a comparison between two
fitted models rather than an argument.

Everything is fitted to bed sequences by exact maximum likelihood. The latent
per-visit variable is integrated out by Gauss-Hermite quadrature and the latent
target sequence is summed out exactly, which is cheap because there are only two
targets and the visit is only three darts long.
"""

import numpy as np
from scipy import optimize
from scipy.special import logsumexp

from darts.dartboards import DARTBOARD_CONSTANTS
from darts.utils import mm_per_pixel

#: Segment numbers clockwise from the 20 at the top.
BOARD_ORDER = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]

#: The scoring targets a player uses, in the order they step down through them.
#:
#: Not one target, and not two. In the pure scoring phase professionals start at
#: the treble 20 (97.6% of first darts) and work *down* the board after a miss:
#: by the third dart 24.5% are at the 19, 5.4% at the 18 and 2.0% at the 17. The
#: moves are almost all one step -- from the 20 a miss goes to the 19 (28.3%),
#: from the 19 it goes to the 18 (36.1%) -- which is why one chain with one pair
#: of probabilities describes the whole thing.
TARGETS = (20, 19, 18, 17)


def treble_centre_mm(number):
    """Centre of a number's treble bed, in mm from the bull, as ``(x, y)``."""
    c = DARTBOARD_CONSTANTS
    r = 0.5 * (c["TRIPLE_INNER_RADIUS"] + c["TRIPLE_OUTER_RADIUS"])
    angle = np.deg2rad(90 - 18 * BOARD_ORDER.index(number))
    return np.array([r * np.cos(angle), r * np.sin(angle)])


def bed_labels(pixels=512):
    """
    Integer bed code at every pixel, and the list of bed names it indexes.

    A *bed* is finer than a score: the treble 20 and the single 20 are different
    beds that a score of 60 and 20 happen to distinguish, but the single 5 and
    the treble 15 both score 15 and are opposite sides of the board. Real data
    records the bed, so the likelihood is built on beds.

    Returns:
        tuple: ``(codes, names)`` -- an ``(pixels, pixels)`` int array and a list
        of names such as ``"T20"``, ``"S20"``, ``"D20"``, ``"25"``, ``"BULL"``,
        ``"MISS"``.
    """
    c = DARTBOARD_CONSTANTS
    radius = c["DARTBOARD_RADIUS_MM"]
    coords = (np.arange(pixels) - pixels // 2) * (2 * radius / pixels)
    x, y = np.meshgrid(coords, coords)
    r = np.hypot(x, y)
    theta = np.mod(np.arctan2(y, x) + np.pi, 2 * np.pi) - np.pi

    names = ["MISS", "BULL", "25"]
    for n in BOARD_ORDER:
        names += [f"S{n}", f"D{n}", f"T{n}"]
    index = {name: i for i, name in enumerate(names)}

    codes = np.zeros((pixels, pixels), dtype=np.int32)      # MISS
    codes[r < c["OUTER_BULLSEYE_RADIUS_MM"]] = index["25"]
    codes[r < c["INNER_BULLSEYE_RADIUS_MM"]] = index["BULL"]

    ring = np.zeros((pixels, pixels), dtype=np.int32)       # 1 single, 2 double, 3 treble
    ring[(r >= c["OUTER_BULLSEYE_RADIUS_MM"]) & (r < c["TRIPLE_INNER_RADIUS"])] = 1
    ring[(r >= c["TRIPLE_OUTER_RADIUS"]) & (r < c["DOUBLE_INNER_RADIUS"])] = 1
    ring[(r >= c["TRIPLE_INNER_RADIUS"]) & (r < c["TRIPLE_OUTER_RADIUS"])] = 3
    ring[(r >= c["DOUBLE_INNER_RADIUS"]) & (r < c["DOUBLE_OUTER_RADIUS"])] = 2

    for number, intervals in c["SEGMENTS"].items():
        seg = np.zeros((pixels, pixels), dtype=bool)
        for lo, hi in intervals:
            seg |= (theta >= lo * np.pi) & (theta < hi * np.pi)
        for mult, prefix in ((1, "S"), (2, "D"), (3, "T")):
            codes[seg & (ring == mult)] = index[f"{prefix}{number}"]

    return codes, names


class BedGrid:
    """
    Bed probabilities for a dart aimed anywhere near one of the scoring targets.

    Holds the bed-code board and a window around each target, so that the
    probability of every bed given an aim point and a covariance is a weighted
    count over a few thousand pixels rather than over the whole board.

    Args:
        pixels (int): board resolution. 512 or finer -- see notebook 02.
        half_window_mm (float): half-width of the window kept around each
            target. Must comfortably exceed four standard deviations of the
            widest throw to be fitted, or probability leaks out of the window.
    """

    def __init__(self, pixels=512, half_window_mm=70.0):
        self.pixels = int(pixels)
        self.scale = mm_per_pixel(self.pixels)
        self.codes, self.names = bed_labels(self.pixels)
        self.n_beds = len(self.names)
        self.half = int(np.ceil(half_window_mm / self.scale))

        coords = (np.arange(self.pixels) - self.pixels // 2) * self.scale
        self._gx, self._gy = np.meshgrid(coords, coords)
        self._wide_cache = {}

        self._windows = {}
        for number in TARGETS:
            centre = treble_centre_mm(number)
            row = int(round(self.pixels // 2 + centre[1] / self.scale))
            col = int(round(self.pixels // 2 + centre[0] / self.scale))
            rows = np.arange(row - self.half, row + self.half + 1)
            cols = np.arange(col - self.half, col + self.half + 1)
            sub = self.codes[np.ix_(rows, cols)]
            # pixel centres in mm, relative to the exact bed centre
            dy = (rows - self.pixels // 2) * self.scale - centre[1]
            dx = (cols - self.pixels // 2) * self.scale - centre[0]
            self._windows[number] = (sub.ravel(), dx, dy)

        # Beds no throw at either target can reach -- the far doubles, the bull
        # -- are pooled with MISS into one catch-all bucket, and observations are
        # mapped the same way by encode_visits. Without it those beds would carry
        # probability exactly zero and a handful of real darts would dominate the
        # log-likelihood with a constant of -690 each, which is not a fit but a
        # broken convergence test. Reachability is judged once at a deliberately
        # wide throw so the outcome space is identical for every model compared.
        wide = sum(self.bed_pmf(n, np.zeros(2), 25.0) for n in TARGETS)
        self.reachable = wide > 0
        self.collapse = np.where(self.reachable, np.arange(self.n_beds), 0)

    def wide_pmf(self, number, sigma):
        """
        ``P(bed)`` over the *whole* board for a throw far too wide for the window.

        The contaminating component of a real throw -- the dart that gets away --
        is spread over tens of millimetres, so it reaches the double ring and the
        bull, which the window around a target deliberately does not cover. It is
        also so wide that a few millimetres of aim offset changes nothing, so this
        ignores the offset and is computed once per likelihood evaluation rather
        than once per quadrature node.
        """
        key = (number, round(float(sigma), 4))
        if key in self._wide_cache:
            return self._wide_cache[key]
        centre = treble_centre_mm(number)
        w = np.exp(-0.5 * (((self._gx - centre[0]) / sigma) ** 2
                           + ((self._gy - centre[1]) / sigma) ** 2))
        pmf = np.bincount(self.codes.ravel(), weights=w.ravel(),
                          minlength=self.n_beds)
        pmf = pmf / pmf.sum()
        if len(self._wide_cache) > 512:
            self._wide_cache.clear()
        self._wide_cache[key] = pmf
        return pmf

    def bed_pmf(self, number, offset, sigma):
        """
        ``P(bed)`` for an isotropic throw aimed at a target's bed centre plus
        ``offset`` millimetres, with standard deviation ``sigma``.

        Returns:
            np.ndarray: length ``n_beds``, summing to 1 up to what falls outside
            the window (which is folded into ``MISS``).
        """
        flat, dx, dy = self._windows[number]
        fx = np.exp(-0.5 * ((dx - offset[0]) / sigma) ** 2)
        fy = np.exp(-0.5 * ((dy - offset[1]) / sigma) ** 2)
        w = np.outer(fy, fx).ravel()
        pmf = np.bincount(flat, weights=w, minlength=self.n_beds)
        total = pmf.sum()
        # everything beyond the window is off the scoring region we model
        norm = 2.0 * np.pi * (sigma / self.scale) ** 2
        outside = max(norm - total, 0.0)
        pmf[0] += outside
        return pmf / (total + outside)


def _gauss_hermite_2d(n):
    """Nodes and weights integrating ``f(u)`` against a standard 2-D normal."""
    x, w = np.polynomial.hermite_e.hermegauss(n)
    w = w / w.sum()
    ux, uy = np.meshgrid(x, x, indexing="ij")
    return np.stack([ux.ravel(), uy.ravel()], axis=1), np.outer(w, w).ravel()


def _gauss_hermite_1d(n):
    x, w = np.polynomial.hermite_e.hermegauss(n)
    return x, w / w.sum()


class VisitModel:
    """
    A model for the three beds of a scoring visit.

    The throw is isotropic with a per-dart standard deviation that splits into a
    part shared by the visit and a part private to the dart, and the aim may move
    between the treble 20 and the treble 19 depending on how the previous dart
    went.

    Args:
        grid (BedGrid): the board machinery.
        shared_offset (bool): give the visit a location offset drawn once and
            used by all three darts (``tau``). This is the "same stance for
            three darts" model, and it couples the darts in *direction*.
        shared_scale (bool): give the visit a multiplicative scale drawn once
            (``nu``, the log standard deviation). This is the "tight visit or
            loose visit" model, and it couples them in *magnitude* only.
        switching (bool): let the aim move to the treble 19 after a dart, with
            separate probabilities after a hit and after a miss. With this off,
            every dart is aimed at the treble 20.
        n_quad (int): quadrature nodes per dimension.

    The parameter vector, in order, is
    ``[log sigma, bias_x, bias_y] (+ [log tau]) (+ [log nu]) (+ [logit s_hit,
    logit s_miss])``.
    """

    def __init__(self, grid, shared_offset=False, shared_scale=False,
                 switching=True, contamination=False,
                 radial_bias=False, n_quad=9):
        self.grid = grid
        self.shared_offset = bool(shared_offset)
        self.shared_scale = bool(shared_scale)
        self.switching = bool(switching)
        self.contamination = bool(contamination)
        self.radial_bias = bool(radial_bias)
        self.n_quad = int(n_quad)
        self._u, self._wu = _gauss_hermite_2d(self.n_quad)
        self._s, self._ws = _gauss_hermite_1d(self.n_quad)

    # -- parameters ---------------------------------------------------------
    @property
    def names(self):
        out = ["log_sigma", "bias_x"]
        if self.radial_bias:
            out.append("bias_y")
        if self.contamination:
            out += ["logit_eps", "log_kappa"]
        if self.shared_offset:
            out.append("log_tau")
        if self.shared_scale:
            out.append("log_nu")
        if self.switching:
            out += ["logit_s_hit", "logit_s_miss"]
        return out

    @property
    def n_params(self):
        return len(self.names)

    def unpack(self, theta):
        """
        Parameter vector to a dict on the natural scale.

        ``bias_y`` -- the radial component, "throwing low" -- is off by default,
        because bed data barely identifies it. Both single 20 beds carry the same
        label, so pushing the aim a millimetre above the treble and a millimetre
        below it produce almost the same distribution of beds: the sideways
        component carries about 32x the Fisher information, and even the *sign*
        of the radial one is close to unidentified. Leaving it at zero lets
        ``sigma`` absorb a real radial pull, which is a stated approximation
        rather than a hidden one, and it biases every model in this family the
        same way, so model comparison is unaffected.
        """
        theta = np.asarray(theta, float)
        i = 1
        bias = np.array([theta[i], theta[i + 1] if self.radial_bias else 0.0])
        i += 1 + self.radial_bias
        out = {"sigma": np.exp(theta[0]), "bias": bias}
        if self.contamination:
            out["eps"] = 1.0 / (1.0 + np.exp(-theta[i]))
            out["kappa"] = 1.0 + np.exp(theta[i + 1])
            i += 2
        else:
            out["eps"], out["kappa"] = 0.0, 1.0
        out["tau"] = np.exp(theta[i]) if self.shared_offset else 0.0
        i += self.shared_offset
        out["nu"] = np.exp(theta[i]) if self.shared_scale else 0.0
        i += self.shared_scale
        if self.switching:
            out["s_hit"] = 1.0 / (1.0 + np.exp(-theta[i]))
            out["s_miss"] = 1.0 / (1.0 + np.exp(-theta[i + 1]))
        else:
            out["s_hit"] = out["s_miss"] = 0.0
        return out

    def pack(self, params):
        """
        Inverse of :meth:`unpack`: a parameter dict back to a vector.

        Lets a fit stored as named columns -- which is how ``results/dependence``
        keeps them -- be turned back into a working model without refitting.
        """
        def logit(p):
            p = min(max(float(p), 1e-12), 1 - 1e-12)
            return float(np.log(p / (1 - p)))

        theta = [float(np.log(params["sigma"])), float(np.asarray(params["bias"])[0])]
        if self.radial_bias:
            theta.append(float(np.asarray(params["bias"])[1]))
        if self.contamination:
            theta += [logit(params["eps"]),
                      float(np.log(max(params["kappa"] - 1.0, 1e-12)))]
        if self.shared_offset:
            theta.append(float(np.log(max(params["tau"], 1e-12))))
        if self.shared_scale:
            theta.append(float(np.log(max(params["nu"], 1e-12))))
        if self.switching:
            theta += [logit(params["s_hit"]), logit(params["s_miss"])]
        return np.array(theta)

    def start(self):
        """A sane starting point for the optimiser."""
        theta = [np.log(7.5), 0.0]
        if self.radial_bias:
            theta.append(0.0)
        if self.contamination:
            theta += [-3.0, np.log(3.0)]
        if self.shared_offset:
            theta.append(np.log(4.0))
        if self.shared_scale:
            theta.append(np.log(0.3))
        if self.switching:
            theta += [-3.0, -1.0]
        return np.array(theta)

    # -- the per-node bed distributions -------------------------------------
    def node_pmfs(self, params):
        """
        ``P(bed | target, node)`` for every quadrature node, and the node weights.

        The nodes are the values of whatever the visit shares: a location offset,
        a scale, or the product grid of both. Conditional on a node the three
        darts are independent, which is what makes the visit likelihood a
        product.

        Returns:
            tuple: ``(pmfs, weights)`` with ``pmfs`` of shape
            ``(n_nodes, n_targets, n_beds)``.
        """
        sigma, bias, tau, nu = (params["sigma"], params["bias"],
                                params["tau"], params["nu"])
        offsets = self._u * tau if self.shared_offset else np.zeros((1, 2))
        w_off = self._wu if self.shared_offset else np.array([1.0])
        scales = np.exp(self._s * nu - 0.5 * nu ** 2) if self.shared_scale else np.array([1.0])
        w_sca = self._ws if self.shared_scale else np.array([1.0])

        pmfs = np.empty((len(offsets) * len(scales), len(TARGETS), self.grid.n_beds))
        weights = np.empty(len(offsets) * len(scales))
        k = 0
        for off, wo in zip(offsets, w_off):
            for sc, ws in zip(scales, w_sca):
                for t, number in enumerate(TARGETS):
                    core = self.grid.bed_pmf(number, bias + off, sigma * sc)
                    if params["eps"] > 0:
                        wide = self.grid.wide_pmf(number, sigma * sc * params["kappa"])
                        core = (1.0 - params["eps"]) * core + params["eps"] * wide
                    pmfs[k, t] = core
                weights[k] = wo * ws
                k += 1
        return pmfs, weights

    # -- likelihood ---------------------------------------------------------
    def log_likelihood(self, theta, beds, hit):
        """
        Total log-likelihood of a set of visits.

        Args:
            theta (array): parameter vector.
            beds (np.ndarray): ``(n_visits, 3)`` integer bed codes.
            hit (np.ndarray): ``(n_visits, 3)`` boolean, whether each dart hit
                the treble of *whichever* target it was aimed at. Only the first
                two columns are used, and only to drive the aim rule.

        The target sequence is summed out exactly. Dart 1 is aimed at the treble
        20; each later dart either stays where it was or steps one place down
        :data:`TARGETS`, with a probability that depends only on whether the
        previous dart hit its treble. That is an *observed* quantity, so this is
        a valid conditional likelihood -- every model in the family predicts the
        same thing, each dart's bed given the beds before it, and their
        likelihoods are directly comparable.
        """
        params = self.unpack(theta)
        if not np.isfinite(list(params["bias"])).all() or params["sigma"] <= 0:
            return -np.inf
        pmfs, weights = self.node_pmfs(params)
        log_p = np.log(np.maximum(pmfs, 1e-300))

        n = len(beds)
        total = np.zeros(n)
        for k, w in enumerate(weights):
            lp = log_p[k]                                   # (n_targets, n_beds)
            # forward pass over the step-down chain: stay, or move one place
            # down TARGETS. The last target is absorbing -- there is nowhere
            # further down that professionals go in the scoring phase.
            last = len(TARGETS) - 1
            alpha = np.full((n, len(TARGETS)), -np.inf)
            alpha[:, 0] = lp[0, beds[:, 0]]                 # dart 1 is at the 20
            for d in (1, 2):
                s = np.where(hit[:, d - 1], params["s_hit"], params["s_miss"])
                stay = np.log(np.maximum(1.0 - s, 1e-300))
                move = np.log(np.maximum(s, 1e-300))
                nxt = np.empty_like(alpha)
                nxt[:, 0] = alpha[:, 0] + stay
                for t in range(1, last):
                    nxt[:, t] = np.logaddexp(alpha[:, t] + stay,
                                             alpha[:, t - 1] + move)
                nxt[:, last] = np.logaddexp(alpha[:, last],
                                            alpha[:, last - 1] + move)
                alpha = nxt + lp[:, beds[:, d]].T
            total = total + w * np.exp(logsumexp(alpha, axis=1))
        return float(np.sum(np.log(np.maximum(total, 1e-300))))

    def fit(self, beds, hit, theta0=None, maxiter=2000):
        """Maximum likelihood by Nelder-Mead, which needs no derivatives."""
        theta0 = self.start() if theta0 is None else np.asarray(theta0, float)
        res = optimize.minimize(
            lambda th: -self.log_likelihood(th, beds, hit), theta0,
            method="Nelder-Mead",
            options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-3},
        )
        return res

    # -- simulation ---------------------------------------------------------
    def simulate(self, theta, n_visits, rng=None, return_targets=False):
        """
        Draw visits from the model, returning ``(beds, hit)`` in the same
        encoding :meth:`log_likelihood` consumes.

        Used to validate the fitter, and to make the posterior-predictive checks
        that decide between models.

        Args:
            return_targets (bool): also return the ``(n_visits, 3)`` index into
                :data:`TARGETS` each dart was actually aimed at. Real data never
                carries this, but simulated data does, and some checks need the
                true aim rather than the one inferred from where the dart landed
                -- a dart thrown at the 20 can easily finish nearer the 18.
        """
        rng = np.random.default_rng() if rng is None else rng
        params = self.unpack(theta)
        grid = self.grid
        treble = [grid.names.index(f"T{n}") for n in TARGETS]

        beds = np.empty((n_visits, 3), dtype=np.int64)
        hit = np.zeros((n_visits, 3), dtype=bool)
        aimed = np.zeros((n_visits, 3), dtype=np.int64)
        for v in range(n_visits):
            off = (rng.normal(size=2) * params["tau"] if self.shared_offset
                   else np.zeros(2))
            sc = (np.exp(rng.normal() * params["nu"] - 0.5 * params["nu"] ** 2)
                  if self.shared_scale else 1.0)
            pmf = []
            for n in TARGETS:
                core = grid.bed_pmf(n, params["bias"] + off, params["sigma"] * sc)
                if params["eps"] > 0:
                    wide = grid.wide_pmf(n, params["sigma"] * sc * params["kappa"])
                    core = (1.0 - params["eps"]) * core + params["eps"] * wide
                pmf.append(core)
            t = 0
            for d in range(3):
                b = rng.choice(grid.n_beds, p=pmf[t] / pmf[t].sum())
                beds[v, d] = b
                aimed[v, d] = t
                hit[v, d] = b == treble[t]
                s = params["s_hit"] if hit[v, d] else params["s_miss"]
                if rng.random() < s:
                    t = min(t + 1, len(TARGETS) - 1)
        return (beds, hit, aimed) if return_targets else (beds, hit)


def bed_geometry(grid):
    """
    For every bed, its signed angular offset from each target and its ring.

    The angular offset is in *segments*: 0 is the target's own number, +1 the
    segment clockwise of it, -1 anticlockwise, and so on, wrapping at +/-10.
    ``np.nan`` marks a bed with no angular position at all (the bull, the 25, and
    anything off the board).

    This is what makes the direction/magnitude distinction testable. A shared
    location offset moves a whole visit the same way round the board, so the
    *signed* offsets of its darts correlate. A shared scale makes them all land
    far out without any preferred side, so only the *absolute* offsets do.
    """
    n_beds = grid.n_beds
    angle = {t: np.full(n_beds, np.nan) for t in TARGETS}
    ring = np.full(n_beds, np.nan)
    for i, name in enumerate(grid.names):
        if name in ("MISS", "BULL", "25"):
            continue
        ring[i] = {"S": 1, "D": 2, "T": 3}[name[0]]
        number = int(name[1:])
        for t in TARGETS:
            d = BOARD_ORDER.index(number) - BOARD_ORDER.index(t)
            angle[t][i] = (d + 10) % 20 - 10
    return angle, ring


def signatures(beds, hit, grid, targets=None):
    """
    The pre-specified statistics that tell the models apart.

    Applied identically to real visits and to simulated ones, so a model is
    judged by whether its own data reproduces what the real data shows.

    Args:
        beds (np.ndarray): ``(n_visits, 3)`` bed codes.
        hit (np.ndarray): ``(n_visits, 3)`` treble-hit flags.
        targets (np.ndarray): ``(n_visits, 3)`` target index per dart, if known.
            Real data does not carry it; when omitted each dart is assigned to
            whichever target its bed is angularly nearer, which is unambiguous
            for every bed that has an angle at all, since the two targets are
            eleven segments apart.

    Returns:
        dict: ``t20_lift_12``, ``t20_lift_23``, ``t20_lift_13`` (percentage
        points), ``treble_lift_12`` (target-invariant), ``dir_corr`` (signed
        angular offsets), ``mag_corr`` (absolute ones), ``p_k`` (the
        distribution of trebles per visit) and ``switch_rate``.
    """
    angle, _ = bed_geometry(grid)
    stacked = np.stack([angle[t] for t in TARGETS])            # (2, n_beds)
    if targets is None:
        # beds with no angle at all (bull, 25, off the board) go to target 0;
        # every other bed is unambiguous, the targets being eleven segments apart
        filled = np.where(np.isnan(stacked), np.inf, np.abs(stacked))
        nearer = np.argmin(filled, axis=0)
        targets = nearer[beds]
    ang = stacked[targets, beds]

    treble_names = [grid.names.index(f"T{t}") for t in TARGETS]
    t20 = (beds == grid.names.index("T20")).astype(float)
    any_treble = np.isin(beds, treble_names).astype(float)

    def lift(a, b):
        if a.sum() < 10 or (1 - a).sum() < 10:
            return np.nan
        return 100.0 * (b[a == 1].mean() - b[a == 0].mean())

    def paired_corr(values, absolute):
        pairs = [(0, 1), (1, 2), (0, 2)]
        xs, ys = [], []
        for i, j in pairs:
            v = np.abs(values) if absolute else values
            ok = np.isfinite(v[:, i]) & np.isfinite(v[:, j])
            xs.append(v[ok, i])
            ys.append(v[ok, j])
        x, y = np.concatenate(xs), np.concatenate(ys)
        if len(x) < 10 or x.std() == 0 or y.std() == 0:
            return np.nan
        return float(np.corrcoef(x, y)[0, 1])

    k = any_treble.sum(axis=1).astype(int)
    return {
        "t20_lift_12": lift(t20[:, 0], t20[:, 1]),
        "t20_lift_23": lift(t20[:, 1], t20[:, 2]),
        "t20_lift_13": lift(t20[:, 0], t20[:, 2]),
        "treble_lift_12": lift(any_treble[:, 0], any_treble[:, 1]),
        "dir_corr": paired_corr(ang, absolute=False),
        "mag_corr": paired_corr(ang, absolute=True),
        "p_k": np.array([(k == i).mean() for i in range(4)]),
        "switch_rate": float((targets[:, 1:] != targets[:, :-1]).mean()),
        "p_t20": float(t20.mean()),
    }


def encode_visits(bed_sequences, grid):
    """
    Turn visits of bed names into the ``(beds, hit)`` arrays the model consumes.

    ``hit`` is "this dart hit the treble of the target it was aimed at", which is
    not observed directly. It does not need to be: a treble can only have been
    aimed at its own number, since no two of :data:`TARGETS` are close enough for
    one to be hit while aiming at another. So the outcome that drives the aim
    rule is observed exactly, even though the aim itself never is.
    """
    index = {name: i for i, name in enumerate(grid.names)}
    beds = np.array([[index.get(b, 0) for b in visit] for visit in bed_sequences],
                    dtype=np.int64)
    beds = grid.collapse[beds]
    trebles = {f"T{t}" for t in TARGETS}
    hit = np.array([[b in trebles for b in visit] for visit in bed_sequences],
                   dtype=bool)
    return beds, hit
