"""
Matches: legs, sets, and the value of throwing first.

Once a leg has been solved, it collapses to two numbers -- the probability of
winning a leg when you throw first, and when your opponent does. A match is
then a small Markov chain over those, and the whole thing solves in
milliseconds. That makes questions like "how much is winning the bull-up worth
in a best of 11?" exact rather than simulated.

Conventions
-----------
* The throw alternates every leg, throughout the match, including across set
  boundaries. This is the standard rule.
* A set is itself a first-to-``legs_per_set`` mini-match.
* ``clear_by=2`` implements the "two clear legs" rule. At deuce the match
  becomes a repeating two-leg cycle -- the throw returns to the same player
  after two legs -- so it has a closed form rather than needing a cap.

All probabilities are from player A's point of view.
"""

from collections import OrderedDict
from functools import lru_cache


# --------------------------------------------------------------------------
# Real formats
# --------------------------------------------------------------------------

#: name -> kwargs for :func:`match_win_probability`
FORMATS = OrderedDict([
    # pub and league
    ("best of 3 legs", dict(legs_to_win=2)),
    ("best of 5 legs", dict(legs_to_win=3)),
    ("best of 7 legs", dict(legs_to_win=4)),
    # PDC leg formats
    ("best of 11 legs (Premier League, UK Open early)", dict(legs_to_win=6)),
    ("best of 19 legs (Grand Slam late)", dict(legs_to_win=10)),
    ("best of 21 legs (UK Open final)", dict(legs_to_win=11)),
    ("first to 17, 2 clear (World Matchplay semi)",
     dict(legs_to_win=17, clear_by=2)),
    ("first to 18, 2 clear (World Matchplay final)",
     dict(legs_to_win=18, clear_by=2)),
    # PDC set formats (World Championship); a set is first to 3 legs
    ("best of 5 sets (World Ch. R1)", dict(sets_to_win=3, legs_per_set=3)),
    ("best of 7 sets (World Ch. R3)", dict(sets_to_win=4, legs_per_set=3)),
    ("best of 9 sets (World Ch. QF)", dict(sets_to_win=5, legs_per_set=3)),
    ("best of 11 sets (World Ch. SF)", dict(sets_to_win=6, legs_per_set=3)),
    ("best of 13 sets (World Ch. final)", dict(sets_to_win=7, legs_per_set=3)),
])


# --------------------------------------------------------------------------

def _deuce(a1, a2):
    """
    Probability A eventually wins from level pegging under a "win by two" rule,
    with A throwing first in the next leg.

    Over two legs the throw returns to where it started, so either someone wins
    both (settled) or the legs are split (back to the start). Hence a geometric
    series with a closed form.
    """
    win_both = a1 * a2
    lose_both = (1.0 - a1) * (1.0 - a2)
    settled = win_both + lose_both
    if settled <= 0.0:
        return 0.5
    return win_both / settled


def match_win_probability(p_first, p_second, legs_to_win=None, sets_to_win=None,
                          legs_per_set=3, clear_by=1, a_throws_first=True):
    """
    Probability that A wins the match.

    Args:
        p_first (float): P(A wins a leg | A throws first in it).
        p_second (float): P(A wins a leg | B throws first in it).
        legs_to_win (int): legs needed, for a leg-format match.
        sets_to_win (int): sets needed, for a set-format match. If given,
            ``legs_per_set`` legs win a set and ``legs_to_win`` is ignored.
        legs_per_set (int): legs needed to take a set.
        clear_by (int): 1 for straight first-to-N; 2 for "two clear legs",
            which only applies to leg formats.
        a_throws_first (bool): whether A throws first in the opening leg.

    Returns:
        float: A's match win probability.
    """
    if sets_to_win is None and legs_to_win is None:
        raise ValueError("give either legs_to_win or sets_to_win")

    def leg_p(a_throws):
        return p_first if a_throws else p_second

    if sets_to_win is None:
        target = legs_to_win

        @lru_cache(maxsize=None)
        def f(a, b, a_throws):
            if clear_by == 2 and a >= target - 1 and b >= target - 1:
                # deuce: only the difference matters from here
                if a - b >= 2:
                    return 1.0
                if b - a >= 2:
                    return 0.0
                if a == b:
                    a1 = leg_p(a_throws)
                    a2 = leg_p(not a_throws)
                    return _deuce(a1, a2)
                # one leg ahead or behind: play it out one leg at a time
                p = leg_p(a_throws)
                return p * f(a + 1, b, not a_throws) + (1 - p) * f(a, b + 1, not a_throws)
            if a >= target and a - b >= clear_by:
                return 1.0
            if b >= target and b - a >= clear_by:
                return 0.0
            p = leg_p(a_throws)
            return p * f(a + 1, b, not a_throws) + (1 - p) * f(a, b + 1, not a_throws)

        return f(0, 0, a_throws_first)

    # --- set format -------------------------------------------------------
    @lru_cache(maxsize=None)
    def g(sa, sb, la, lb, a_throws):
        if sa >= sets_to_win:
            return 1.0
        if sb >= sets_to_win:
            return 0.0
        if la >= legs_per_set:
            return g(sa + 1, sb, 0, 0, a_throws)
        if lb >= legs_per_set:
            return g(sa, sb + 1, 0, 0, a_throws)
        p = leg_p(a_throws)
        return (p * g(sa, sb, la + 1, lb, not a_throws)
                + (1 - p) * g(sa, sb, la, lb + 1, not a_throws))

    return g(0, 0, 0, 0, a_throws_first)


def throw_advantage(p_first, p_second, **fmt):
    """
    How much winning the bull-up is worth in a given format.

    Returns:
        dict: A's win probability throwing first, throwing second, and the
        difference in percentage points.
    """
    first = match_win_probability(p_first, p_second, a_throws_first=True, **fmt)
    second = match_win_probability(p_first, p_second, a_throws_first=False, **fmt)
    return {"throwing first": first, "throwing second": second,
            "advantage (pp)": 100.0 * (first - second)}


def format_table(p_first, p_second, formats=None):
    """
    Win probability and throw advantage across every format in ``FORMATS``.

    Returns:
        list[dict]: one row per format, ready for a DataFrame.
    """
    formats = formats or FORMATS
    rows = []
    for name, fmt in formats.items():
        r = throw_advantage(p_first, p_second, **fmt)
        rows.append({"format": name,
                     "P(win) throwing first": round(r["throwing first"], 4),
                     "P(win) throwing second": round(r["throwing second"], 4),
                     "bull-up worth (pp)": round(r["advantage (pp)"], 2)})
    return rows


def required_edge(p_first_curve, target=0.5, **fmt):
    """
    Given a callable mapping a leg-level edge to ``(p_first, p_second)``, find
    the edge at which A's match win probability reaches ``target`` when A
    throws second. Used to answer "how much better must the underdog be to
    give away the throw?".
    """
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        p1, p2 = p_first_curve(mid)
        if match_win_probability(p1, p2, a_throws_first=False, **fmt) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
