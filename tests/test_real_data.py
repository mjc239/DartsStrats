"""Notebook 19's claims, restated as assertions.

These need the real scoring data, which is not committed -- see
`data/real/README.md`. Without it every test here skips, which is the correct
behaviour for a clone that has not run `scripts/build_real_data.py`.
"""
import os

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from darts.calibration import SCORING_FLOOR

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'data', 'real')
KEY = ['source', 'player', 'leg_id', 'visit_index']

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(DATA, 'per_dart.csv')),
    reason='run scripts/build_real_data.py first (see data/real/README.md)')


@pytest.fixture(scope='module')
def visits():
    """One row per pure-scoring visit, with a T20 flag for each of its darts."""
    per_dart = pd.read_csv(os.path.join(DATA, 'per_dart.csv'), low_memory=False)
    # the 2017 feed keeps subtracting after a bust, and 15 darts carry a
    # negative dart_index that would silently poison the pivot
    d = per_dart[(per_dart.post_bust_visit == 0) & per_dart.dart_index.isin([1, 2, 3])]
    info = d.groupby(KEY).agg(n=('dart_index', 'size'), start=('score_before', 'max'),
                              total=('value', 'sum')).reset_index()
    ok = info[(info.n == 3) & ((info.start - info.total) >= SCORING_FLOOR)]
    scoring = d.merge(ok[KEY], on=KEY)
    scoring['t20'] = (scoring.bed == 'T20').astype(int)
    vis = (scoring.pivot_table(index=KEY, columns='dart_index', values='t20')
           .dropna().reset_index().merge(ok[KEY + ['start']], on=KEY))
    assert len(vis) > 30_000, 'far fewer pure-scoring visits than expected'
    return vis


def _lift(flag, following):
    """Percentage-point lift in `following` given `flag`, with its standard error."""
    h = following[flag == 1].mean()
    m = following[flag == 0].mean()
    n1 = int((flag == 1).sum())
    n0 = int((flag == 0).sum())
    return 100 * (h - m), 100 * np.sqrt(h * (1 - h) / n1 + m * (1 - m) / n0)


def test_the_three_darts_of_a_visit_are_not_independent(visits):
    """The assumption under every transition matrix in the project is false."""
    for a, b in ((1, 2), (2, 3)):
        lift, se = _lift(visits[a], visits[b])
        assert lift > 15.0, f'dart {a} -> {b}: lift only {lift:.1f} points'
        assert lift / se > 20.0, f'dart {a} -> {b}: only {lift / se:.1f} sigma'


def test_visits_are_over_dispersed_in_trebles(visits):
    """More all-or-nothing than three independent darts can be."""
    k = visits[[1, 2, 3]].sum(axis=1)
    counts = np.array([(k == i).sum() for i in range(4)], float)
    obs = counts / counts.sum()
    p = float(visits[[1, 2, 3]].mean().mean())
    binom = np.array([stats.binom.pmf(i, 3, p) for i in range(4)])
    assert obs[0] > binom[0] and obs[3] > binom[3], 'both ends should be fattened'
    assert obs[3] / binom[3] > 2.0, 'maximums should be at least twice as common'
    assert obs[1] < binom[1], 'the one-treble column pays for it'


def test_the_dependence_is_not_an_artefact_of_the_scoring_filter(visits):
    """Selecting on the visit total induces *negative* dependence, not positive.

    So restricting to visits where the filter cannot bite must not weaken the
    effect -- notebook 19 measures it getting stronger.
    """
    unfiltered = visits[visits.start >= SCORING_FLOOR + 180]
    assert len(unfiltered) > 20_000
    filtered_lift, _ = _lift(visits[1], visits[2])
    clean_lift, _ = _lift(unfiltered[1], unfiltered[2])
    assert clean_lift > filtered_lift


def test_the_dependence_does_not_survive_the_walk_to_the_board(visits):
    """Within a visit, not across it -- which rules out slow form drift."""
    q = visits.sort_values(['source', 'player', 'leg_id', 'visit_index']).copy()
    grp = q.groupby(['source', 'player', 'leg_id'])
    q['next1'] = grp[1].shift(-1)
    q['gap'] = grp['visit_index'].shift(-1) - q['visit_index']
    adj = q[(q.gap == 2) & q.next1.notna()]      # visit_index alternates by player
    lift, se = _lift(adj[3], adj.next1)
    assert lift < 0, f'cross-visit lift {lift:+.1f} should not be positive'
    assert abs(lift) < abs(_lift(visits[2], visits[3])[0])


def test_the_dependence_is_not_pooling_of_different_players(visits):
    """It reproduces inside individual players, so it is not Simpson's paradox."""
    lifts, weights = [], []
    for _, s in visits.groupby(['source', 'player']):
        if len(s) < 300:
            continue
        est, var = [], []
        for a, b in ((1, 2), (2, 3)):
            if min((s[a] == 1).sum(), (s[a] == 0).sum()) < 30:
                continue
            lift, se = _lift(s[a], s[b])
            est.append(lift)
            var.append(se ** 2)
        if len(est) == 2:
            lifts.append(np.mean(est))
            weights.append(1 / (sum(var) / 4))
    lifts, weights = np.array(lifts), np.array(weights)
    assert len(lifts) >= 30, 'expected at least 30 players with enough visits'
    assert (lifts > 0).sum() >= len(lifts) - 2, 'should be positive for nearly all'
    pooled = (weights * lifts).sum() / weights.sum()
    assert pooled > 15.0, f'within-player pooled lift only {pooled:.1f} points'


def test_all_twenty_doubles_are_equally_hard_when_players_are_pooled():
    """The isotropic model's flattest prediction survives the pooled test."""
    dbl = pd.read_csv(os.path.join(DATA, 'double_attempts.csv'))
    pool = (dbl[dbl.double != 'DB'].groupby('double')
            .agg(attempts=('attempts', 'sum'), hits=('hits', 'sum')))
    assert len(pool) == 20
    p0 = pool.hits.sum() / pool.attempts.sum()
    exp_hit = pool.attempts * p0
    exp_miss = pool.attempts - exp_hit
    chi2 = (((pool.hits - exp_hit) ** 2 / exp_hit)
            + (((pool.attempts - pool.hits) - exp_miss) ** 2 / exp_miss)).sum()
    assert stats.chi2.sf(chi2, len(pool) - 1) > 0.05


def test_but_the_players_disagree_with_each_other_about_which_doubles_are_hard():
    """A null on average, real heterogeneity underneath -- notebook 12's signature."""
    dbl = pd.read_csv(os.path.join(DATA, 'double_attempts.csv'))
    nums = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
    angle = {f'D{n}': (90 - 18 * i) % 360 for i, n in enumerate(nums)}
    dd = dbl[dbl.double != 'DB'].copy()
    c2 = np.cos(2 * np.deg2rad(dd.double.map(angle)))
    dd['band'] = np.where(c2 > 0.5, 'side', np.where(c2 < -0.5, 'top/bottom', None))

    log_or, se = [], []
    for _, s in dd.dropna(subset=['band']).groupby('player'):
        g = s.groupby('band').agg(a=('attempts', 'sum'), h=('hits', 'sum'))
        if set(g.index) != {'side', 'top/bottom'} or g.a.min() < 40:
            continue
        if min(g.h.min(), (g.a - g.h).min()) == 0:
            continue
        h1, n1 = g.loc['side', 'h'], g.loc['side', 'a']
        h2, n2 = g.loc['top/bottom', 'h'], g.loc['top/bottom', 'a']
        log_or.append(np.log((h1 / (n1 - h1)) / (h2 / (n2 - h2))))
        se.append(np.sqrt(1 / h1 + 1 / (n1 - h1) + 1 / h2 + 1 / (n2 - h2)))

    log_or, w = np.array(log_or), 1 / np.array(se) ** 2
    assert len(log_or) >= 15
    mu = (w * log_or).sum() / w.sum()
    Q = (w * (log_or - mu) ** 2).sum()
    dof = len(log_or) - 1
    assert stats.chi2.sf(Q, dof) < 0.05, f'Q = {Q:.1f} on {dof} df is not heterogeneous'
    assert 100 * (Q - dof) / Q > 25, 'less than a quarter of the spread is real'
