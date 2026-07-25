"""Tests for the match (legs and sets) model."""

import numpy as np
import pytest

from darts.match import (FORMATS, format_table, match_win_probability,
                         throw_advantage)


def simulate_match(p1, p2, legs_to_win, clear_by=1, a_first=True,
                   n=200000, seed=0, max_legs=600):
    rng = np.random.default_rng(seed)
    wins = 0
    for _ in range(n):
        a = b = 0
        turn = a_first
        for _ in range(max_legs):
            if rng.random() < (p1 if turn else p2):
                a += 1
            else:
                b += 1
            turn = not turn
            if a >= legs_to_win and a - b >= clear_by:
                wins += 1
                break
            if b >= legs_to_win and b - a >= clear_by:
                break
    return wins / n


def simulate_sets(p1, p2, sets_to_win, legs_per_set=3, a_first=True,
                  n=200000, seed=0):
    rng = np.random.default_rng(seed)
    wins = 0
    for _ in range(n):
        sa = sb = 0
        turn = a_first
        while sa < sets_to_win and sb < sets_to_win:
            la = lb = 0
            while la < legs_per_set and lb < legs_per_set:
                if rng.random() < (p1 if turn else p2):
                    la += 1
                else:
                    lb += 1
                turn = not turn
            sa += la >= legs_per_set
            sb += lb >= legs_per_set
        wins += sa >= sets_to_win
    return wins / n


@pytest.mark.parametrize("p1,p2,target,clear", [
    (0.5725, 0.4275, 6, 1),
    (0.5725, 0.4275, 17, 2),
    (0.70, 0.55, 6, 1),
    (0.70, 0.55, 3, 2),
])
def test_leg_formats_match_simulation(p1, p2, target, clear):
    exact = match_win_probability(p1, p2, legs_to_win=target, clear_by=clear)
    mc = simulate_match(p1, p2, target, clear)
    se = np.sqrt(max(mc * (1 - mc), 1e-6) / 200000)
    assert abs(exact - mc) < 5 * se


@pytest.mark.parametrize("p1,p2,sets", [(0.5725, 0.4275, 3), (0.65, 0.5, 5)])
def test_set_formats_match_simulation(p1, p2, sets):
    exact = match_win_probability(p1, p2, sets_to_win=sets, legs_per_set=3)
    mc = simulate_sets(p1, p2, sets)
    se = np.sqrt(max(mc * (1 - mc), 1e-6) / 200000)
    assert abs(exact - mc) < 5 * se


def test_degenerate_inputs():
    for fmt in FORMATS.values():
        assert match_win_probability(0.5, 0.5, **fmt) == pytest.approx(0.5)
        assert match_win_probability(1.0, 1.0, **fmt) == pytest.approx(1.0)
        assert match_win_probability(0.0, 0.0, **fmt) == pytest.approx(0.0)


def test_best_of_one_is_the_leg_itself():
    assert match_win_probability(0.6, 0.4, legs_to_win=1) == pytest.approx(0.6)
    assert match_win_probability(0.6, 0.4, legs_to_win=1,
                                 a_throws_first=False) == pytest.approx(0.4)


def test_symmetry_between_equal_players():
    """With equal players the two throw orders must sum to one."""
    p1, p2 = 0.5725, 1 - 0.5725
    for fmt in FORMATS.values():
        first = match_win_probability(p1, p2, a_throws_first=True, **fmt)
        second = match_win_probability(p1, p2, a_throws_first=False, **fmt)
        assert first + second == pytest.approx(1.0, abs=1e-12)


def test_throw_advantage_shrinks_with_match_length():
    """The longer the match, the less the bull-up is worth."""
    p1, p2 = 0.5725, 1 - 0.5725
    adv = [throw_advantage(p1, p2, legs_to_win=n)["advantage (pp)"]
           for n in (2, 3, 6, 10, 11)]
    assert all(a > b for a, b in zip(adv, adv[1:]))


def test_two_clear_legs_removes_the_throw_advantage():
    """
    Between equal players, "win by two" makes the throw worthless: the leg
    difference over any two consecutive legs is symmetric, because each player
    throws first in one of them.
    """
    p1, p2 = 0.5725, 1 - 0.5725
    r = throw_advantage(p1, p2, legs_to_win=17, clear_by=2)
    assert r["advantage (pp)"] == pytest.approx(0.0, abs=1e-9)
    # but it is still worth something to the better player
    r2 = throw_advantage(0.70, 0.55, legs_to_win=17, clear_by=2)
    assert r2["throwing first"] > 0.9


def test_format_table_covers_every_format():
    rows = format_table(0.5725, 0.4275)
    assert len(rows) == len(FORMATS)
    assert all(0.0 <= r["P(win) throwing first"] <= 1.0 for r in rows)
