"""
Is a dart's landing point Gaussian? Alternative throwing distributions.

Notebook 20 could not fit real beds with a Gaussian at all. A throw tight enough
to hit the treble 20 at a professional rate puts essentially nothing in the
double 20 or off the board, and real players land there percents of the time; the
fit only worked once a second, much wider Gaussian was mixed in. That is a patch,
and an unconvincing one -- the wide component's width was unidentified for seven
of nineteen players, running off wider than the board itself, which is what an
over-flexible model looks like when the data cannot pin it down.

That patch also turned out to be fitting a data defect -- see :mod:`darts.real_data`
and notebook 21 -- but the question survives it: what *is* a dart's landing
point? This module holds six candidate families and the machinery to compare
them on equal terms.

Every candidate here is an **isotropic radial density**: the landing point is the
aim plus a displacement whose direction is uniform and whose length has some
distribution. They differ only in how much weight sits far from the middle, and
they are deliberately ordered by that:

* :class:`Gaussian` -- ``exp(-r^2)``, the thinnest tail, and what the project has
  assumed throughout.
* :class:`ExponentialPower` -- ``exp(-(r/sigma)^beta)``. At ``beta = 2`` it *is*
  the Gaussian; below 2 the tail is stretched, above 2 it is squashed. One extra
  parameter, and it can say the throw is *tighter*-tailed than Gaussian as easily
  as heavier, which a mixture cannot.
* :class:`StudentT` -- polynomial tails, ``r^-(nu+2)``. Heavier than any
  exponential. It is exactly a Gaussian whose width is redrawn for every dart,
  which is a physically sensible thing for a throw to be: the player's precision
  is not identical from dart to dart.
* :class:`CoreUniform` -- a Gaussian core plus a flat background over the board.
  One parameter, and the right shape if the excess is *contamination* -- darts
  that are not really throws at the target -- rather than a graded tail.
* :class:`TwoComponent` -- the mixture notebook 20 used, kept as the reference to
  beat rather than as a candidate. Two extra parameters.

One candidate is not about tails at all:

* :class:`EllipticalGaussian` -- a Gaussian group taller than it is wide. This is
  the **rival hypothesis**, not another option. An isotropic fit to an elongated
  group must account for the extra reach along the long axis somehow, and with
  only a scale to play with the one way left is a fatter tail -- so it predicts
  the same sort of excess for the same single parameter. Pricing them alike is
  what makes the comparison mean anything.

All six nest the Gaussian -- at ``beta = 2``, at ``nu -> infinity``, at
``eps = 0``, at ``ratio = 1`` -- so a comparison between them is a question about
whether the extra flexibility is used, not about which curve happens to fit.

Each family is parameterised by a scale of its own, because there is no
parameterisation in which one number means the same thing for all six. What is
comparable is the **implied standard deviation per axis**, which every family
reports, and which is what the rest of the project means by ``sigma``.

A warning about that number, from what the fits actually returned. The winning
Student-t sits at ``nu`` around 2.25, and below ``nu = 2`` a bivariate t has no
variance at all. So ``axis_sd`` exists but is not a useful summary of a real
throw -- the fitted *core scale* is (median 5.98 mm across professionals, against
11.47 mm for a Gaussian fitted to the same darts, which splits the difference
between core and tail and describes neither).
"""

import numpy as np
from scipy import optimize, special

from darts.dependence import BedGrid, TARGETS, bed_labels  # noqa: F401


class RadialFamily:
    """A radially symmetric landing distribution.

    Subclasses supply an unnormalised profile as a function of squared radius,
    the exact integral of that profile over the plane, and the variance per axis
    it implies. Working in squared radius keeps a square root out of the inner
    loop, which is evaluated over a quarter of a million pixels per call.
    """

    name = "?"
    shape_names = ()

    @property
    def n_shape(self):
        return len(self.shape_names)

    def profile(self, r2, scale, shape):
        """Unnormalised density at squared radius ``r2``."""
        raise NotImplementedError

    def squared_radius(self, dx, dy, shape):
        """
        The squared distance the profile is a function of.

        Circular by default. An anisotropic family overrides this to stretch one
        axis, which lets "the group is an ellipse" compete against "the tail is
        heavy" for the same single extra parameter -- they are rival
        explanations of the same excess and the comparison is only fair if they
        are priced alike.
        """
        return dx * dx + dy * dy

    def area_scale(self, shape):
        """How much ``squared_radius`` stretches area, so norm() stays exact."""
        return 1.0

    def norm(self, scale, shape):
        """``\\int profile dA`` over the whole plane, exactly."""
        raise NotImplementedError

    def axis_sd(self, scale, shape):
        """Standard deviation per axis, or ``inf`` where it does not exist."""
        raise NotImplementedError

    def start_shape(self):
        """Unconstrained starting values for the shape parameters."""
        return np.zeros(self.n_shape)

    def describe(self, shape):
        """Shape parameters on their natural scale, as a dict."""
        return {}

    def is_gaussian(self, shape, tol=1e-3):
        """Whether these shape parameters sit at the Gaussian special case."""
        return self.n_shape == 0


class Gaussian(RadialFamily):
    """``exp(-r^2 / 2 sigma^2)``. What the project has assumed throughout."""

    name = "gaussian"

    def profile(self, r2, scale, shape):
        return np.exp(-0.5 * r2 / scale ** 2)

    def norm(self, scale, shape):
        return 2.0 * np.pi * scale ** 2

    def axis_sd(self, scale, shape):
        return float(scale)

    def is_gaussian(self, shape, tol=1e-3):
        return True


class StudentT(RadialFamily):
    """
    Bivariate Student-t: ``(1 + r^2 / (nu sigma^2))^{-(nu + 2) / 2}``.

    A Gaussian whose width is redrawn from an inverse-gamma for every dart, which
    is why it is the natural candidate: it says a player's precision varies dart
    to dart rather than being one fixed number. Its tail is polynomial, so it
    reaches the double 20 and the floor without needing a second component.

    ``nu`` is carried as ``log(nu - 2)`` so that the variance always exists;
    below ``nu = 2`` the model would have infinite spread, which is not a claim
    any measurement of a darts player supports.
    """

    name = "student-t"
    shape_names = ("log_nu_minus_2",)

    def _nu(self, shape):
        return 2.0 + np.exp(np.clip(shape[0], -30.0, 30.0))

    def profile(self, r2, scale, shape):
        nu = self._nu(shape)
        return (1.0 + r2 / (nu * scale ** 2)) ** (-(nu + 2.0) / 2.0)

    def norm(self, scale, shape):
        # the nu cancels: the integral is 2 pi sigma^2 for every nu
        return 2.0 * np.pi * scale ** 2

    def axis_sd(self, scale, shape):
        nu = self._nu(shape)
        return float(scale * np.sqrt(nu / (nu - 2.0)))

    def start_shape(self):
        return np.array([np.log(6.0)])          # nu = 8, moderately heavy

    def describe(self, shape):
        return {"nu": float(self._nu(shape))}

    def is_gaussian(self, shape, tol=1e-3):
        return self._nu(shape) > 1.0 / tol


class ExponentialPower(RadialFamily):
    """
    ``exp(-(r / sigma)^beta)``, the isotropic exponential-power family.

    At ``beta = 2`` this is exactly a Gaussian, so the fitted ``beta`` is a direct
    read on the question: below 2 the throw has a heavier tail than Gaussian,
    above 2 a lighter one. It is the only candidate here that can come out on the
    *thin* side, which is what makes it a fair test rather than a search for
    heaviness.
    """

    name = "exp-power"
    shape_names = ("log_beta",)

    def _beta(self, shape):
        return float(np.exp(np.clip(shape[0], -3.0, 3.0)))

    def profile(self, r2, scale, shape):
        beta = self._beta(shape)
        return np.exp(-np.power(np.maximum(r2, 0.0) / scale ** 2, beta / 2.0))

    def norm(self, scale, shape):
        beta = self._beta(shape)
        return 2.0 * np.pi * scale ** 2 / beta * special.gamma(2.0 / beta)

    def axis_sd(self, scale, shape):
        beta = self._beta(shape)
        # E[r^2] = sigma^2 Gamma(4/beta) / Gamma(2/beta); per axis it is half
        return float(scale * np.sqrt(special.gamma(4.0 / beta)
                                     / (2.0 * special.gamma(2.0 / beta))))

    def start_shape(self):
        return np.array([np.log(2.0)])          # start at the Gaussian

    def describe(self, shape):
        return {"beta": self._beta(shape)}

    def is_gaussian(self, shape, tol=1e-3):
        return abs(self._beta(shape) - 2.0) < tol


class TwoComponent(RadialFamily):
    """
    ``(1 - eps) N(sigma) + eps N(kappa sigma)`` -- notebook 20's patch.

    Included as the incumbent to beat. It buys its tail with two parameters where
    the others use one, and notebook 20 found ``kappa`` unidentified for seven of
    nineteen players, which is the symptom this whole comparison exists to test.
    """

    name = "two-component"
    shape_names = ("logit_eps", "log_kappa_minus_1")

    def _parts(self, shape):
        eps = 1.0 / (1.0 + np.exp(-np.clip(shape[0], -30.0, 30.0)))
        kappa = 1.0 + np.exp(np.clip(shape[1], -10.0, 10.0))
        return float(eps), float(kappa)

    def profile(self, r2, scale, shape):
        eps, kappa = self._parts(shape)
        wide = kappa * scale
        return ((1.0 - eps) * np.exp(-0.5 * r2 / scale ** 2) / (2 * np.pi * scale ** 2)
                + eps * np.exp(-0.5 * r2 / wide ** 2) / (2 * np.pi * wide ** 2))

    def norm(self, scale, shape):
        return 1.0

    def axis_sd(self, scale, shape):
        eps, kappa = self._parts(shape)
        return float(np.sqrt((1.0 - eps) * scale ** 2 + eps * (kappa * scale) ** 2))

    def start_shape(self):
        return np.array([-2.0, np.log(4.0)])

    def describe(self, shape):
        eps, kappa = self._parts(shape)
        return {"eps": eps, "kappa": kappa}

    def is_gaussian(self, shape, tol=1e-3):
        return self._parts(shape)[0] < tol


class CoreUniform(RadialFamily):
    """
    A Gaussian core plus a flat background: ``(1 - eps) N(sigma) + eps / A``.

    The mixture's wide component kept running off to widths of 20-plus times the
    core, which is a distribution with no shape left -- at that width it is
    indistinguishable from "anywhere on the board". This says that outright, and
    with one parameter instead of two: a fraction ``eps`` of darts land uniformly
    over the board and are not really throws at the target at all.

    It is the right shape for genuine contamination -- a dart that bounced, a
    scoring error, a dart from another leg -- and the wrong shape for a throw
    whose accuracy merely varies. Which of those the data wants is the question.
    """

    name = "core+uniform"
    shape_names = ("logit_eps",)
    #: the flat component is spread over the scoring area, out to the double
    #: ring, and is zero beyond it -- a background dart lands *somewhere on the
    #: board*, which is what makes its contribution a proper distribution with a
    #: finite spread rather than a constant over the plane
    RADIUS = 170.0
    AREA = np.pi * RADIUS ** 2

    def _eps(self, shape):
        return 1.0 / (1.0 + np.exp(-np.clip(shape[0], -30.0, 30.0)))

    def profile(self, r2, scale, shape):
        eps = self._eps(shape)
        core = np.exp(-0.5 * r2 / scale ** 2) / (2 * np.pi * scale ** 2)
        # NB the radius is measured from the *target*, not the bull, so this is
        # a disc of board-sized area centred on where the player is aiming. The
        # exact placement matters little for a component this diffuse, and it
        # keeps the family radial like every other one here.
        flat = np.where(r2 <= self.RADIUS ** 2, eps / self.AREA, 0.0)
        return (1.0 - eps) * core + flat

    def norm(self, scale, shape):
        # the flat part is only spread over the board, so the profile integrates
        # to 1 over the board and the pixel sum supplies the rest
        return 1.0

    def axis_sd(self, scale, shape):
        eps = self._eps(shape)
        # a uniform disc of radius R has per-axis variance R^2 / 4
        return float(np.sqrt((1 - eps) * scale ** 2
                             + eps * self.RADIUS ** 2 / 4.0))

    def start_shape(self):
        return np.array([-3.0])

    def describe(self, shape):
        return {"eps": self._eps(shape)}

    def is_gaussian(self, shape, tol=1e-3):
        return self._eps(shape) < tol


class EllipticalGaussian(RadialFamily):
    """
    A Gaussian group that is taller than it is wide, or wider than tall.

    Not a heavy tail at all -- the rival explanation. An isotropic fit to an
    elliptical group has to account for the extra reach along the long axis
    somehow, and with only a scale to play with the only way is a fatter tail.
    So the two hypotheses predict the same *sort* of excess and cost the same one
    parameter, which is what makes this a test rather than an extra option.

    The axes are the board's, not the player's: ``ratio`` is the standard
    deviation along the radial direction at the treble 20 (up and down the 20
    segment) divided by the sideways one. Notebook 12 measured what shape costs
    in visits per leg; this asks whether the shape is there at all.
    """

    name = "elliptical"
    shape_names = ("log_ratio",)

    def _ratio(self, shape):
        return float(np.exp(np.clip(shape[0], -2.0, 2.0)))

    def squared_radius(self, dx, dy, shape):
        # dy runs radially at the treble 20, dx tangentially
        return dx * dx + (dy / self._ratio(shape)) ** 2

    def area_scale(self, shape):
        return self._ratio(shape)

    def profile(self, r2, scale, shape):
        return np.exp(-0.5 * r2 / scale ** 2)

    def norm(self, scale, shape):
        return 2.0 * np.pi * scale ** 2

    def axis_sd(self, scale, shape):
        # report the geometric mean, so it is comparable with the round families
        return float(scale * np.sqrt(self._ratio(shape)))

    def start_shape(self):
        return np.array([0.0])

    def describe(self, shape):
        return {"ratio": self._ratio(shape)}

    def is_gaussian(self, shape, tol=1e-3):
        return abs(self._ratio(shape) - 1.0) < tol


class EllipticalStudentT(RadialFamily):
    """
    A Student-t group that is taller than it is wide.

    Notebook 21 priced "the group is an ellipse" against "the tail is heavy" and
    the ellipse gained nothing (-0.02 a visit). But it tested the ellipse with a
    **Gaussian** core, and a Gaussian core fitted to heavy-tailed darts does not
    merely come out too wide -- it comes out the wrong *shape*. On a simulated
    2.11:1 elongated t it returns 2.81:1, and an off-diagonal five times too
    large. So a null result measured that way is not safe, and the two extra
    parameters have to be offered together before "no ellipse" means anything.

    ``squared_radius`` is :class:`EllipticalGaussian`'s and ``profile`` is
    :class:`StudentT`'s. That composes because the base class keeps the metric
    and the radial shape apart, which is the whole reason they are separate
    methods.
    """

    name = "elliptical-t"
    shape_names = ("log_nu_minus_2", "log_ratio")

    def _nu(self, shape):
        return 2.0 + np.exp(np.clip(shape[0], -30.0, 30.0))

    def _ratio(self, shape):
        return float(np.exp(np.clip(shape[1], -2.0, 2.0)))

    def squared_radius(self, dx, dy, shape):
        # dy runs radially at the treble 20, dx tangentially -- the same axes
        # EllipticalGaussian uses, so the fitted ratios are comparable
        return dx * dx + (dy / self._ratio(shape)) ** 2

    def area_scale(self, shape):
        return self._ratio(shape)

    def profile(self, r2, scale, shape):
        nu = self._nu(shape)
        return (1.0 + r2 / (nu * scale ** 2)) ** (-(nu + 2.0) / 2.0)

    def norm(self, scale, shape):
        return 2.0 * np.pi * scale ** 2

    def axis_sd(self, scale, shape):
        nu = self._nu(shape)
        # geometric mean across the axes, as EllipticalGaussian reports
        return float(scale * np.sqrt(self._ratio(shape)) * np.sqrt(nu / (nu - 2.0)))

    def start_shape(self):
        return np.array([np.log(6.0), 0.0])

    def describe(self, shape):
        return {"nu": float(self._nu(shape)), "ratio": self._ratio(shape)}

    def is_gaussian(self, shape, tol=1e-3):
        return self._nu(shape) > 1.0 / tol and abs(self._ratio(shape) - 1.0) < tol


class NormalInverseGaussian(RadialFamily):
    """
    A Gaussian whose width is redrawn from an **inverse Gaussian** each dart.

    The Student-t is the same idea with an inverse-*gamma* mixing law, and that
    choice is what gives it polynomial tails and a variance that only exists
    above ``nu = 2``. Five of the seventeen professionals notebook 21 fitted sat
    exactly on that boundary, which says the likelihood wanted a heavier tail
    than a finite variance permits. That is a statement about the parameterisation
    rather than about darts players, and this family is the reply to it.

    Writing ``Z = sqrt(W) * scale * G`` with ``W ~ InverseGaussian(mean 1, shape
    kappa)`` and ``q = r^2 / scale^2``, ``s = sqrt(kappa (q + kappa))``:

        profile(q) = (1 + 1/s) exp(kappa - s) / (q + kappa)

    Three things follow, and they are the reasons to prefer it as the bounded
    candidate:

    * **Every moment is finite.** The tail decays like ``exp(-sqrt(kappa) r)``
      rather than as a power of ``r``, so there is no ``nu = 2`` cliff to sit on
      and no need to clip anything to keep a variance.
    * **``scale`` is the per-axis standard deviation, exactly**, because ``W``
      has mean 1 by construction. Nothing else here can say that: the t's scale
      is a core and its SD is ``sqrt(nu/(nu-2))`` times larger when it exists at
      all.
    * **The core and the tail are no longer the same parameter.** A t has one
      ``nu`` setting both how peaked the middle is and how far the tail reaches;
      here the mixing law is free to be sharply peaked *and* long, which is what
      the boundary-hitting players appear to be asking for.

    ``kappa -> infinity`` is the Gaussian (the mixing law collapses onto 1), so it
    nests like the others and the comparison stays a question about whether the
    flexibility is used.
    """

    name = "nig"
    shape_names = ("log_kappa",)

    def _kappa(self, shape):
        return float(np.exp(np.clip(shape[0], -8.0, 20.0)))

    def profile(self, r2, scale, shape):
        kappa = self._kappa(shape)
        q = np.maximum(r2, 0.0) / scale ** 2
        s = np.sqrt(kappa * (q + kappa))
        # kappa - s is about -q/2 for large kappa, so this never underflows the
        # way exp(-s) alone would
        return (1.0 + 1.0 / s) * np.exp(kappa - s) / (q + kappa)

    def norm(self, scale, shape):
        # 2 pi scale^2 e^kappa (E1(kappa) + E2(kappa)/kappa), which collapses to
        # this by E2(x) = e^-x - x E1(x). Checked against direct integration of
        # the profile at kappa from 0.05 to 300.
        return 2.0 * np.pi * scale ** 2 / self._kappa(shape)

    def axis_sd(self, scale, shape):
        # exact: the mixing law has mean 1, so Var = scale^2 whatever kappa is
        return float(scale)

    def start_shape(self):
        return np.array([np.log(2.0)])

    def describe(self, shape):
        return {"kappa": self._kappa(shape)}

    def is_gaussian(self, shape, tol=1e-3):
        return self._kappa(shape) > 1.0 / tol


FAMILIES = {f.name: f for f in (Gaussian(), ExponentialPower(), StudentT(),
                                CoreUniform(), TwoComponent(),
                                EllipticalGaussian(), EllipticalStudentT(),
                                NormalInverseGaussian())}


class RadialBedGrid:
    """
    ``P(bed)`` for any radial family, over the whole board.

    Notebook 20's grid used a window around each target plus a separate
    whole-board pass for the wide component. That will not do here: the families
    being compared differ precisely in how much mass sits far out, so every one of
    them has to be integrated over the same region, to the same accuracy, with the
    same treatment of what falls off the board. Everything beyond the board edge
    is pooled into ``MISS``, which is what the data records it as.
    """

    def __init__(self, pixels=512, grid=None):
        self.grid = BedGrid(pixels) if grid is None else grid
        self.names = self.grid.names
        self.n_beds = self.grid.n_beds
        # forwarded so encode_visits() and the notebook-20 machinery can take
        # either grid without caring which
        self.collapse = self.grid.collapse
        self.reachable = self.grid.reachable
        self.codes = self.grid.codes
        self.pixels = self.grid.pixels
        self.scale = self.grid.scale
        self.pixel_area = self.scale ** 2
        self.codes_flat = self.grid.codes.ravel()

        coords = (np.arange(self.grid.pixels) - self.grid.pixels // 2) * self.scale
        gx, gy = np.meshgrid(coords, coords)
        from darts.dependence import treble_centre_mm
        self._r2 = {}
        for number in TARGETS:
            centre = treble_centre_mm(number)
            self._r2[number] = ((gx - centre[0]) ** 2 + (gy - centre[1]) ** 2).ravel()
        self._gx, self._gy = gx.ravel(), gy.ravel()

    def bed_pmf(self, family, number, offset, scale, shape):
        """``P(bed)`` for a dart aimed at a target's centre plus ``offset``."""
        from darts.dependence import treble_centre_mm
        centre = treble_centre_mm(number)
        dx = self._gx - (centre[0] + offset[0])
        dy = self._gy - (centre[1] + offset[1])
        r2 = family.squared_radius(dx, dy, shape)
        dens = family.profile(r2, scale, shape)
        pmf = np.bincount(self.codes_flat, weights=dens, minlength=self.n_beds)
        pmf *= self.pixel_area
        total = family.norm(scale, shape) * family.area_scale(shape)
        # what the board does not cover is off the board, which is a MISS
        pmf[0] += max(total - pmf.sum(), 0.0)
        out = pmf / max(pmf.sum(), 1e-300)
        return self._collapse(out)

    def _collapse(self, pmf):
        """Pool the beds no throw at any target can reach, as notebook 20 does."""
        collapsed = np.zeros_like(pmf)
        np.add.at(collapsed, self.grid.collapse, pmf)
        return collapsed


class FamilyVisitModel:
    """
    Fit a radial family to the beds of real scoring visits.

    Everything except the family is held at what notebook 20 established: the aim
    starts at the treble 20 and steps down the board after a miss, with one pair
    of probabilities driving the chain. Only the shape of a single dart's
    distribution changes between the models compared here, so a difference in
    held-out likelihood is a statement about that shape and nothing else.

    The radial bias is left out for the same reason it was in notebook 20: both
    single-20 beds carry one label, so a pull towards or away from the bull is
    barely identified from beds, while a sideways pull carries about thirty times
    the information.

    Args:
        family (RadialFamily): the candidate being fitted.
        grid (RadialBedGrid): board machinery, shared between models.
        shared_scale (bool): additionally give the visit a scale drawn once, the
            coupling notebook 20 selected. Off for the family comparison; used
            afterwards to ask whether a better per-dart family removes the need
            for it.
        n_quad (int): quadrature nodes, used only when ``shared_scale``.
    """

    def __init__(self, family, grid, shared_scale=False, n_quad=7):
        self.family = family
        self.grid = grid
        self.shared_scale = bool(shared_scale)
        self.n_quad = int(n_quad)
        x, w = np.polynomial.hermite_e.hermegauss(self.n_quad)
        self._s, self._ws = x, w / w.sum()

    @property
    def names(self):
        out = ["log_scale", "bias_x"] + list(self.family.shape_names)
        if self.shared_scale:
            out.append("log_nu_visit")
        return out + ["logit_s_hit", "logit_s_miss"]

    @property
    def n_params(self):
        return len(self.names)

    def unpack(self, theta):
        theta = np.asarray(theta, float)
        i = 2 + self.family.n_shape
        out = {"scale": float(np.exp(np.clip(theta[0], -5.0, 6.0))),
               "bias": np.array([theta[1], 0.0]),
               "shape": theta[2:2 + self.family.n_shape]}
        out["nu_visit"] = float(np.exp(theta[i])) if self.shared_scale else 0.0
        i += self.shared_scale
        out["s_hit"] = 1.0 / (1.0 + np.exp(-theta[i]))
        out["s_miss"] = 1.0 / (1.0 + np.exp(-theta[i + 1]))
        return out

    def start(self, scale=7.5):
        theta = [np.log(scale), 0.0] + list(self.family.start_shape())
        if self.shared_scale:
            theta.append(np.log(0.3))
        return np.array(theta + [-3.0, -1.0])

    def _target_pmfs(self, params):
        """``(n_nodes, n_targets, n_beds)`` and the node weights."""
        if self.shared_scale:
            nu = params["nu_visit"]
            factors = np.exp(self._s * nu - 0.5 * nu ** 2)
            weights = self._ws
        else:
            factors, weights = np.array([1.0]), np.array([1.0])
        pmfs = np.empty((len(factors), len(TARGETS), self.grid.n_beds))
        for k, f in enumerate(factors):
            for t, number in enumerate(TARGETS):
                pmfs[k, t] = self.grid.bed_pmf(self.family, number, params["bias"],
                                               params["scale"] * f, params["shape"])
        return pmfs, weights

    def log_likelihood(self, theta, beds, hit):
        params = self.unpack(theta)
        if not np.isfinite(np.asarray(theta, float)).all():
            return -np.inf
        pmfs, weights = self._target_pmfs(params)
        log_p = np.log(np.maximum(pmfs, 1e-300))

        n = len(beds)
        last = len(TARGETS) - 1
        total = np.zeros(n)
        for k, w in enumerate(weights):
            lp = log_p[k]
            alpha = np.full((n, len(TARGETS)), -np.inf)
            alpha[:, 0] = lp[0, beds[:, 0]]
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
            m = alpha.max(axis=1)
            total += w * np.exp(m) * np.exp(alpha - m[:, None]).sum(axis=1)
        return float(np.sum(np.log(np.maximum(total, 1e-300))))

    def fit(self, beds, hit, theta0=None, maxiter=4000, restarts=(6.0, 9.0)):
        """
        Maximum likelihood, from several starting scales.

        Notebook 20 was caught by a Nelder-Mead simplex too small to move a
        parameter off its starting value, so the scale is restarted from more
        than one place and the best fit kept.
        """
        best = None
        starts = [self.start() if theta0 is None else np.asarray(theta0, float)]
        starts += [self.start(scale=s) for s in restarts]
        for th0 in starts:
            res = optimize.minimize(
                lambda th: -self.log_likelihood(th, beds, hit), th0,
                method="Nelder-Mead",
                options={"maxiter": maxiter, "xatol": 1e-4, "fatol": 1e-4,
                         "initial_simplex": _simplex(th0)},
            )
            if best is None or res.fun < best.fun:
                best = res
        return best

    def simulate(self, theta, n_visits, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        params = self.unpack(theta)
        treble = [self.grid.names.index(f"T{n}") for n in TARGETS]
        beds = np.empty((n_visits, 3), dtype=np.int64)
        hit = np.zeros((n_visits, 3), dtype=bool)
        base, _ = self._target_pmfs(params)
        for v in range(n_visits):
            if self.shared_scale:
                f = np.exp(rng.normal() * params["nu_visit"]
                           - 0.5 * params["nu_visit"] ** 2)
                pmf = [self.grid.bed_pmf(self.family, n, params["bias"],
                                         params["scale"] * f, params["shape"])
                       for n in TARGETS]
            else:
                pmf = base[0]
            t = 0
            for d in range(3):
                p = np.maximum(pmf[t], 0.0)
                b = rng.choice(self.grid.n_beds, p=p / p.sum())
                beds[v, d] = b
                hit[v, d] = b == treble[t]
                s = params["s_hit"] if hit[v, d] else params["s_miss"]
                if rng.random() < s:
                    t = min(t + 1, len(TARGETS) - 1)
        return beds, hit


def _simplex(theta0, step=0.4):
    """A starting simplex wide enough to move every parameter."""
    n = len(theta0)
    simplex = np.repeat(theta0[None, :], n + 1, axis=0)
    for i in range(n):
        simplex[i + 1, i] += step
    return simplex
