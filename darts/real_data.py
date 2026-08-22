"""
Loading the real scoring data, with the cleaning the first pass missed.

Notebooks 19 and 20 each built their own filter inline. They agreed with each
other, and they were both wrong in the same way, which is the argument for having
one definition in one place.

**The defect.** In the 2017 feed, 5.65% of player-legs begin with a dart that is a
double or a miss. A leg starts at 501; nobody throws at a double there. The
values give it away -- 0, 40, 32, 28, 26, 36 -- which are missed doubles and made
doubles, the signature of a *checkout*. The previous leg's finishing darts are
being carried across the boundary into the next leg's opening visit. The 2022
feed does the same thing 0.13% of the time, which is the rate genuine wayward
first darts occur at, so essentially all of the 2017 excess is contamination.

**Why it mattered.** Every dart in the pure scoring phase with a remaining score
of 430 or more is, almost by definition, near the start of a leg, so the
contaminated opening visit dominated that sample. In it, 100% of double-20s and
99.5% of misses sat at exactly ``score_before == 501``; at any other score they
were absent. Notebook 20 read that as a throw with tails far too heavy for a
Gaussian, and spent two parameters on a wide mixture component to reproduce it.
The tail was leaked checkout darts.

What survives untouched is everything about the *aim*: the four-target step-down
rule, the treble-20 lift and the target-invariant lift all move by about a point
when the contaminated legs are removed, and the 2022 feed reproduces them on its
own.

**The rule.** Drop any player-leg whose first dart is a double or a miss. It is
blunt, and it removes the ~0.1% of legs that genuinely open with a wayward dart
along with the contamination, so it biases the far tail very slightly *down*.
Anything subtler would need to know which of two identical-looking darts was
real. :func:`contamination_report` quantifies both the defect and the cost.
"""

import os

import numpy as np
import pandas as pd

from darts.calibration import SCORING_FLOOR

#: Directory the build script writes to.
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "real")

#: A visit is keyed by the player, not just the leg: two players share a leg and
#: both of them start it on 501.
VISIT_KEY = ["source", "player", "leg_id", "visit_index"]

#: A player's own half of a leg.
PLAYER_LEG = ["source", "match_id", "leg_id", "player"]


def load_darts(path=None):
    """Every dart, with the three-dart index enforced and busts dropped."""
    path = os.path.join(DATA, "per_dart.csv") if path is None else path
    d = pd.read_csv(path, low_memory=False)
    # the 2017 feed keeps subtracting after a bust, and a handful of darts carry
    # a negative dart_index that would silently poison any pivot
    return d[(d.post_bust_visit == 0) & d.dart_index.isin([1, 2, 3])].copy()


def suspect_player_legs(darts):
    """
    Player-legs whose first dart is a double or a miss.

    Returns a boolean Series aligned to ``darts``, true for every dart belonging
    to such a leg.
    """
    ordered = darts.sort_values(PLAYER_LEG + ["visit_index", "dart_index"])
    first = ordered.groupby(PLAYER_LEG, sort=False).head(1)
    bad = first[first.bed.str.startswith("D") | (first.bed == "MISS")]
    keys = set(map(tuple, bad[PLAYER_LEG].values))
    return pd.Series(list(map(tuple, darts[PLAYER_LEG].values)),
                     index=darts.index).isin(keys)


def scoring_visits(darts=None, clean=True, unfiltered_only=True,
                   floor=SCORING_FLOOR):
    """
    One row per pure-scoring visit, with the three beds as columns 1, 2, 3.

    Args:
        clean (bool): drop the contaminated player-legs. Leave it on unless the
            point is to measure the contamination.
        unfiltered_only (bool): keep only visits starting at ``floor + 180`` or
            more, where the scoring filter cannot select on a visit's own
            outcome. Notebook 19 showed that selection biases the measured
            dependence downward.
    """
    d = load_darts() if darts is None else darts
    if clean:
        d = d[~suspect_player_legs(d)]
    info = d.groupby(VISIT_KEY).agg(n=("dart_index", "size"),
                                    start=("score_before", "max"),
                                    total=("value", "sum")).reset_index()
    ok = info[(info.n == 3) & ((info.start - info.total) >= floor)]
    if unfiltered_only:
        ok = ok[ok.start >= floor + 180]
    scoring = d.merge(ok[VISIT_KEY], on=VISIT_KEY)
    visits = (scoring.pivot_table(index=VISIT_KEY, columns="dart_index",
                                  values="bed", aggfunc="first")
              .dropna().reset_index())
    return visits.merge(ok[VISIT_KEY + ["start"]], on=VISIT_KEY)


def contamination_report(darts=None):
    """
    What the defect looks like, per source, as a table.

    The columns that matter are ``first_dart_odd`` -- how often a player-leg
    opens on a double or a miss -- and the far-tail rates before and after
    cleaning. A clean feed shows a first-dart oddity rate near 0.001 and loses
    almost nothing to the rule.
    """
    d = load_darts() if darts is None else darts
    suspect = suspect_player_legs(d)
    rows = []
    for source, g in d.groupby("source"):
        s = suspect.loc[g.index]
        ordered = g.sort_values(PLAYER_LEG + ["visit_index", "dart_index"])
        first = ordered.groupby(PLAYER_LEG, sort=False).head(1)
        row = {"source": source, "darts": len(g),
               "player_legs": len(first),
               "first_dart_odd": float((first.bed.str.startswith("D")
                                        | (first.bed == "MISS")).mean()),
               "darts_dropped": float(s.mean())}
        for label, sub in (("dirty", g), ("clean", g[~s])):
            v = scoring_visits(sub, clean=False)
            d1 = v[1]
            row[f"{label}_n"] = len(v)
            row[f"{label}_T20"] = float((d1 == "T20").mean())
            row[f"{label}_D20"] = float((d1 == "D20").mean())
            row[f"{label}_MISS"] = float((d1 == "MISS").mean())
        rows.append(row)
    return pd.DataFrame(rows).set_index("source")
