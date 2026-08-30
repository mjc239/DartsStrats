"""The candidate throwing distributions, and the data cleaning underneath them."""
import os

import numpy as np
import pytest
from scipy import integrate

from darts.throw_families import (FAMILIES, Gaussian, RadialBedGrid,
                                  FamilyVisitModel)

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'data', 'real')
needs_data = pytest.mark.skipif(
    not os.path.exists(os.path.join(DATA, 'per_dart.csv')),
    reason='run scripts/build_real_data.py first (see data/real/README.md)')


@pytest.fixture(scope='module')
def grid():
    return RadialBedGrid(512)


@pytest.mark.parametrize('name', list(FAMILIES))
def test_every_family_is_a_probability_distribution(grid, name):
    """Each profile's analytic normaliser must match the board integral."""
    family = FAMILIES[name]
    for scale in (4.0, 9.0):
        pmf = grid.bed_pmf(family, 20, np.zeros(2), scale, family.start_shape())
        assert abs(pmf.sum() - 1.0) < 1e-9
        assert (pmf >= 0).all()


def test_the_exponential_power_at_beta_two_is_exactly_the_gaussian(grid):
    """The family that nests the Gaussian must reproduce it, not merely approach it.

    Its scale is defined by exp(-(r/s)^2) against the Gaussian's exp(-r^2/2 s^2),
    so the same distribution is s * sqrt(2) in one and s in the other.
    """
    ep = FAMILIES['exp-power']
    beta_two = np.array([np.log(2.0)])
    for sigma in (5.0, 8.5):
        same = grid.bed_pmf(ep, 20, np.zeros(2), sigma * np.sqrt(2.0), beta_two)
        gauss = grid.bed_pmf(Gaussian(), 20, np.zeros(2), sigma, ())
        assert np.abs(same - gauss).max() < 1e-12
        assert abs(ep.axis_sd(sigma * np.sqrt(2.0), beta_two) - sigma) < 1e-9


def test_each_family_contains_the_gaussian(grid):
    """At the nesting point every generalisation must equal the Gaussian."""
    gauss = grid.bed_pmf(Gaussian(), 20, np.zeros(2), 7.0, ())
    nested = {
        'student-t': np.array([np.log(1e8)]),          # nu -> infinity
        'core+uniform': np.array([-40.0]),             # eps -> 0
        'two-component': np.array([-40.0, np.log(3.0)]),
        'elliptical-t': np.array([np.log(1e8), 0.0]),  # nu -> infinity, round
        'nig': np.array([np.log(1e9)]),                # kappa -> infinity
    }
    for name, shape in nested.items():
        pmf = grid.bed_pmf(FAMILIES[name], 20, np.zeros(2), 7.0, shape)
        assert np.abs(pmf - gauss).max() < 1e-6, name


@pytest.mark.parametrize('name', list(FAMILIES))
def test_the_reported_spread_matches_the_distribution(name):
    """axis_sd is quoted in every table, so it has to be the real thing.

    Checked by sampling the radial profile directly rather than by trusting the
    algebra twice.
    """
    family = FAMILIES[name]
    shape, scale = family.start_shape(), 7.0
    r = np.linspace(1e-6, 4000.0, 400_000)
    dens = family.profile(r ** 2, scale, shape) * 2 * np.pi * r
    mass = np.trapezoid(dens, r)
    mean_r2 = np.trapezoid(dens * r ** 2, r) / mass
    assert abs(family.axis_sd(scale, shape) - np.sqrt(mean_r2 / 2.0)) < 0.05 * \
        family.axis_sd(scale, shape)


def test_a_heavier_family_puts_more_weight_far_out(grid):
    """Ordering the families by tail weight is the point of the comparison."""
    far = [grid.names.index(n) for n in ('D20', 'MISS')]
    tail = {}
    for name, family in FAMILIES.items():
        pmf = grid.bed_pmf(family, 20, np.zeros(2), 7.0, family.start_shape())
        tail[name] = pmf[far].sum()
    assert tail['gaussian'] < tail['student-t']
    assert tail['gaussian'] < tail['two-component']
    assert tail['gaussian'] < tail['core+uniform']


def test_the_fit_recovers_a_family_it_generated(grid):
    """Simulate a known heavy tail and measure it back."""
    family = FAMILIES['core+uniform']
    model = FamilyVisitModel(family, grid)
    truth = np.array([np.log(7.0), 0.0, np.log(0.06 / 0.94), -3.0, -1.0])
    beds, hit = model.simulate(truth, 3000, rng=np.random.default_rng(0))
    got = model.unpack(model.fit(beds, hit).x)
    assert abs(got['scale'] - 7.0) < 1.2
    assert abs(family.describe(got['shape'])['eps'] - 0.06) < 0.03


# -- the data defect ------------------------------------------------------

@needs_data
def test_the_2017_feed_leaks_checkout_darts_across_leg_boundaries():
    """A leg starts on 501; nobody throws at a double there.

    This is the defect notebooks 19 and 20 both missed. It is asserted rather
    than merely documented so that a future rebuild cannot quietly reintroduce
    it.
    """
    from darts.real_data import contamination_report
    report = contamination_report()
    dirty = report.loc['dartsviz_pdc_2017']
    clean = report.loc['sportradar_pdc_wc_2022']
    assert dirty.first_dart_odd > 0.03, 'the 2017 defect should be plainly visible'
    assert clean.first_dart_odd < 0.01, 'the 2022 feed is the clean comparison'
    # and the far tail it manufactures is what the cleaning has to remove
    assert dirty.dirty_D20 > 0.005 and dirty.clean_D20 < 0.001
    assert dirty.dirty_MISS > 0.02 and dirty.clean_MISS < 0.002


@needs_data
def test_cleaning_removes_the_tail_but_not_the_aim_rule():
    """The correction is confined: notebooks 19 and 20 are wrong about the tail
    and right about the aim."""
    import numpy as np
    from darts.real_data import scoring_visits
    N20 = {'T20', 'S20', 'S5', 'S1', 'T5', 'T1', 'D20', 'D5', 'D1'}
    N19 = {'T19', 'S19', 'S3', 'S7', 'T3', 'T7', 'D19', 'D3', 'D7'}

    def summarise(v):
        side = lambda b: np.where(b.isin(N20), '20',
                                  np.where(b.isin(N19), '19', 'other'))
        s1, s2 = side(v[1]), side(v[2])
        hit = v[1] == 'T20'
        stay = (s2[hit.values] == '20').mean()
        miss = (s1 == '20') & ~hit.values
        move = (s2[miss] == '19').mean()
        t2 = (v[2] == 'T20').astype(float)
        lift = 100 * (t2[hit.values].mean() - t2[~hit.values].mean())
        return stay, move, lift

    dirty = summarise(scoring_visits(clean=False))
    clean = summarise(scoring_visits(clean=True))
    for got, want, tol, what in zip(clean, dirty, (0.02, 0.03, 3.0),
                                    ('stay after a hit', 'move after a miss',
                                     'treble-20 lift')):
        assert abs(got - want) < tol, f'{what} should survive cleaning'
    assert clean[1] > 0.2, 'the step down to the 19 is the surviving finding'


# -- the two families added after notebook 21 ------------------------------

def test_an_elliptical_t_that_is_round_is_the_student_t(grid):
    """
    The metric and the radial shape are separate methods precisely so they can be
    composed. At ratio = 1 the composition has to collapse onto its parts exactly,
    or the extra parameter is not measuring elongation but a change of family.
    """
    ell, t = FAMILIES['elliptical-t'], FAMILIES['student-t']
    for scale, log_nu in ((6.0, np.log(0.25)), (9.0, np.log(6.0))):
        a = grid.bed_pmf(ell, 20, np.zeros(2), scale, np.array([log_nu, 0.0]))
        b = grid.bed_pmf(t, 20, np.zeros(2), scale, np.array([log_nu]))
        assert np.abs(a - b).max() < 1e-14


def test_an_elliptical_t_stretches_the_way_the_elliptical_gaussian_does(grid):
    """Same axes, so the two ratios mean the same thing and can be compared."""
    ell_t, ell_g = FAMILIES['elliptical-t'], FAMILIES['elliptical']
    for ratio in (0.6, 1.8):
        dx, dy = np.array([3.0, 0.0]), np.array([0.0, 3.0])
        st = ell_t.squared_radius(dx, dy, np.array([np.log(6.0), np.log(ratio)]))
        sg = ell_g.squared_radius(dx, dy, np.array([np.log(ratio)]))
        assert st == pytest.approx(sg)
        assert ell_t.area_scale(np.array([0.0, np.log(ratio)])) == pytest.approx(ratio)


def test_the_nig_normaliser_is_what_it_claims(grid):
    """
    ``norm`` is quoted as ``2 pi scale^2 / kappa``, which came out of collapsing
    two exponential integrals. That is exactly the sort of algebra worth checking
    against the profile rather than re-deriving, since a wrong normaliser would
    show up only as a quietly mis-scaled off-board mass.
    """
    nig = FAMILIES['nig']
    for kappa in (0.05, 0.5, 2.0, 20.0, 300.0):
        shape = np.array([np.log(kappa)])
        for scale in (3.0, 8.0):
            direct, _ = integrate.quad(
                lambda r: nig.profile(r ** 2, scale, shape) * 2 * np.pi * r,
                0.0, np.inf, limit=800, epsabs=1e-14, epsrel=1e-12)
            # it is exact, not merely close: adaptive quadrature agrees to the
            # last bit at every kappa from 0.05 to 300
            assert nig.norm(scale, shape) == pytest.approx(direct, rel=1e-12)


def test_the_nig_scale_is_exactly_the_per_axis_sd():
    """
    Its mixing law has mean 1 by construction, so ``scale`` *is* the standard
    deviation -- the one family here that can say that. The Student-t's cannot:
    its scale is a core, and its SD is larger by sqrt(nu/(nu-2)) when it exists.
    """
    nig, t = FAMILIES['nig'], FAMILIES['student-t']
    for kappa in (0.1, 1.0, 50.0):
        shape = np.array([np.log(kappa)])
        dens = lambda r, k: nig.profile(r ** 2, 7.0, shape) * 2 * np.pi * r * r ** k
        m0, _ = integrate.quad(dens, 0.0, np.inf, args=(0,), limit=800,
                               epsabs=1e-14, epsrel=1e-12)
        m2, _ = integrate.quad(dens, 0.0, np.inf, args=(2,), limit=800,
                               epsabs=1e-12, epsrel=1e-12)
        assert np.sqrt(m2 / m0 / 2.0) == pytest.approx(7.0, rel=1e-8)
        assert nig.axis_sd(7.0, shape) == pytest.approx(7.0)
    assert t.axis_sd(7.0, np.array([np.log(0.25)])) > 7.0


def test_the_nig_tail_is_exponential_and_the_t_tail_is_not():
    """
    The reason for the family. A Student-t's tail is a power of r, so moments
    above ``nu`` do not exist and five of notebook 21's seventeen players sat on
    the ``nu > 2`` clip. The NIG's decays like exp(-sqrt(kappa) r), so every
    moment exists and there is no boundary to sit on.
    """
    nig, t = FAMILIES['nig'], FAMILIES['student-t']
    # far enough out to be in the tail, near enough that exp(-sqrt(kappa) r)
    # is still a representable double
    r = np.array([100.0, 200.0, 300.0])
    for kappa in (0.5, 2.0):
        shape = np.array([np.log(kappa)])
        lp = np.log(nig.profile(r ** 2, 1.0, shape))
        slopes = np.diff(lp) / np.diff(r)
        # a constant slope in r is an exponential tail, and it is -sqrt(kappa)
        assert slopes == pytest.approx(-np.sqrt(kappa) * np.ones(2), rel=0.02)
    # the t's slope in log r is constant instead -- a power law
    nu = 2.25
    lp = np.log(t.profile(r ** 2, 1.0, np.array([np.log(nu - 2.0)])))
    slopes = np.diff(lp) / np.diff(np.log(r))
    assert slopes == pytest.approx(-(nu + 2.0) * np.ones(2), rel=0.01)


@pytest.mark.parametrize('name', ['elliptical-t', 'nig'])
def test_the_fit_recovers_the_new_families(grid, name):
    """Simulate each and measure it back, as core+uniform is measured back above."""
    family = FAMILIES[name]
    model = FamilyVisitModel(family, grid)
    if name == 'nig':
        truth = np.array([np.log(9.0), 0.0, np.log(1.5), -3.0, -1.0])
    else:
        truth = np.array([np.log(6.0), 0.0, np.log(0.4), np.log(1.6), -3.0, -1.0])
    beds, hit = model.simulate(truth, 4000, rng=np.random.default_rng(1))
    got = model.unpack(model.fit(beds, hit).x)
    described = family.describe(got['shape'])
    if name == 'nig':
        assert abs(got['scale'] - 9.0) < 2.0
        assert 0.5 < described['kappa'] < 5.0
    else:
        assert abs(got['scale'] - 6.0) < 1.5
        assert abs(described['ratio'] - 1.6) < 0.5


def test_a_tilted_t_that_is_square_to_the_board_is_the_elliptical_t(grid):
    """
    The tilt is an addition, not a replacement: at e2 = 0 the ellipse is square
    to the board and the family has to reproduce the one it generalises.
    """
    tilt, ell = FAMILIES['tilted-t'], FAMILIES['elliptical-t']
    # Only for ratio >= 1. Below it the two agree on the ellipse and disagree on
    # which axis carries the scale, so `scale` means different things -- see the
    # canonicalisation test below.
    for log_nu, log_ratio in ((np.log(0.4), np.log(1.6)), (np.log(3.0), np.log(2.1))):
        a = grid.bed_pmf(tilt, 20, np.zeros(2), 6.0,
                         np.array([log_nu, log_ratio, 0.0]))
        b = grid.bed_pmf(ell, 20, np.zeros(2), 6.0, np.array([log_nu, log_ratio]))
        assert np.abs(a - b).max() < 1e-14


def test_the_ellipse_coordinates_round_trip_and_have_no_degenerate_point(grid):
    """
    Carried as (log_ratio, tilt), a round group has no angle -- every tilt
    describes it equally well, so the likelihood has a ridge of equivalent
    parameters and an optimiser walks along it. Since the elongations fitted to
    real players sit near 1, that ridge is where the fits live. In (e1, e2) a
    round group is one point.
    """
    fam = FAMILIES['tilted-t']
    for ratio, deg in ((1.6, 0.0), (1.6, 30.0), (1.6, 45.0), (1.6, 150.0),
                       (2.4, 115.0)):
        lr, th = np.log(ratio), np.radians(deg)
        shape = np.array([np.log(0.4), lr * np.cos(2 * th), lr * np.sin(2 * th)])
        got = fam.describe(shape)
        assert got['ratio'] == pytest.approx(ratio, rel=1e-12)
        assert got['tilt_deg'] == pytest.approx(deg % 180.0, abs=1e-9)
    # one point, not a ridge
    assert fam.describe(np.array([0.0, 0.0, 0.0]))['ratio'] == pytest.approx(1.0)
    # and the coordinates are canonical: a group squashed by 0.7 along one axis
    # is the same group stretched by 1/0.7 along the other, and comes back that
    # way rather than as a second name for the same thing
    lr, th = np.log(0.7), np.radians(115.0)
    got = fam.describe(np.array([0.0, lr * np.cos(2 * th), lr * np.sin(2 * th)]))
    assert got['ratio'] == pytest.approx(1 / 0.7, rel=1e-12)
    assert got['tilt_deg'] == pytest.approx(25.0, abs=1e-9)
    # and a tilt of theta and theta + 180 are the same group
    for deg in (20.0, 75.0):
        lr = np.log(1.4)
        a = np.array([0.0, lr * np.cos(2 * np.radians(deg)),
                      lr * np.sin(2 * np.radians(deg))])
        b = np.array([0.0, lr * np.cos(2 * np.radians(deg + 180)),
                      lr * np.sin(2 * np.radians(deg + 180))])
        assert np.abs(a - b).max() < 1e-12


def test_rho_is_zero_in_the_two_ways_it_should_be(grid):
    """
    rho measures the *interaction* of elongation and lean, so it vanishes both
    for a round group and for an elongated group square to the board. Quoting it
    as "the lean" without that caveat would be wrong, and the peak at 45 degrees
    is where an axis-aligned model is blind.
    """
    fam = FAMILIES['tilted-t']

    def shape(ratio, deg):
        lr, th = np.log(ratio), np.radians(deg)
        return np.array([np.log(0.4), lr * np.cos(2 * th), lr * np.sin(2 * th)])

    assert fam.rho(shape(1.0, 30.0)) == pytest.approx(0.0, abs=1e-12)
    assert fam.rho(shape(1.6, 0.0)) == pytest.approx(0.0, abs=1e-12)
    assert fam.rho(shape(1.6, 90.0)) == pytest.approx(0.0, abs=1e-12)
    assert fam.rho(shape(1.6, 45.0)) == pytest.approx(0.4382, abs=1e-3)
    assert fam.rho(shape(1.6, 135.0)) == pytest.approx(-0.4382, abs=1e-3)
    # the maximum a given elongation can reach, at 45 degrees
    for ratio in (1.3, 1.6, 2.2):
        k = ratio ** 2
        assert fam.rho(shape(ratio, 45.0)) == pytest.approx((k - 1) / (k + 1),
                                                            rel=1e-9)


def test_the_scale_matrix_agrees_with_the_metric_it_came_from(grid):
    """``scale_matrix`` is quoted to the rest of the project, so it must be the
    same quadratic form the profile is actually evaluated on."""
    fam = FAMILIES['tilted-t']
    rng = np.random.default_rng(0)
    for _ in range(5):
        shape = np.array([np.log(0.5), rng.normal(0, 0.4), rng.normal(0, 0.4)])
        S = fam.scale_matrix(7.0, shape)
        inv = np.linalg.inv(S)
        pts = rng.normal(0, 20, (30, 2))
        direct = fam.squared_radius(pts[:, 0], pts[:, 1], shape) / 7.0 ** 2
        quad = np.einsum('ij,jk,ik->i', pts, inv, pts)
        assert direct == pytest.approx(quad, rel=1e-10)
