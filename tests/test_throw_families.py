"""The candidate throwing distributions, and the data cleaning underneath them."""
import os

import numpy as np
import pytest

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
