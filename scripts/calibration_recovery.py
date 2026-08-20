#!/usr/bin/env python
"""
Can a player be measured from a scoresheet, and how many darts does it take?

Simulates legs from a known throw, keeps only what a per-visit scoresheet would
record -- the score before each visit and what the visit scored, never which
dart scored what -- and refits. The gap between the fitted and true sigma is
what real match data would buy.

The interesting comparison is whether the aim is fitted or fixed. From a single
target the aim and the spread are confounded (notebook 09), so a fit that lets
the aim float can explain an unluckily tight group as a displaced aim instead.
Holding the aim at the treble 20 bed centre removes that freedom, at the cost of
assuming the player really does aim at the middle of the bed.

Writes ``results/calibration/recovery.csv``.

Usage:
    python scripts/calibration_recovery.py --seeds 12
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from darts.calibration import SCORING_FLOOR, T20_MM, fit_from_visits
from darts.dartboards import generate_dartboard
from darts.utils import mm_per_pixel


def simulate_scoresheet(sigma, n_legs, seed, board, pixels, floor):
    """Legs of pure scoring, recorded as a scoresheet would record them."""
    mm, centre = mm_per_pixel(pixels), pixels // 2
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_legs):
        score = 501
        while score > floor:
            before, total = score, 0
            for _ in range(3):
                land = T20_MM + sigma * rng.standard_normal(2)
                col = int(round(land[0] / mm)) + centre
                row = int(round(land[1] / mm)) + centre
                total += (int(board[row, col])
                          if 0 <= row < pixels and 0 <= col < pixels else 0)
            rows.append((before, total, 3))
            score -= total
    return pd.DataFrame(rows, columns=["score_before", "visit_score", "darts_used"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pixels", type=int, default=512)
    ap.add_argument("--sigmas", nargs="*", type=float, default=[6.5, 8.0, 11.0])
    ap.add_argument("--legs", nargs="*", type=int, default=[25, 50, 100, 200, 400])
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--out", default="recovery.csv")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    board, _ = generate_dartboard(args.pixels)

    rows, t0 = [], time.perf_counter()
    for sigma in args.sigmas:
        for n_legs in args.legs:
            for seed in range(args.seeds):
                d = simulate_scoresheet(sigma, n_legs, seed * 1000 + n_legs
                                        + int(sigma * 10), board, args.pixels,
                                        SCORING_FLOOR)
                for fix in (False, True):
                    f = fit_from_visits(d.score_before, d.visit_score,
                                        d.darts_used, board_pixels=args.pixels,
                                        board=board, sigma_init=10.0, fix_mu=fix)
                    rows.append({"sigma": sigma, "legs": n_legs, "seed": seed,
                                 "fix_mu": fix, "fitted": f["sigma_mm"],
                                 "darts": f["n_darts"], "mu_y": f["mu"][1]})
            print(f"  sigma {sigma}, {n_legs} legs "
                  f"[{time.perf_counter()-t0:.0f}s]", flush=True)

    out_dir = os.path.join(root, "results", "calibration")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, args.out)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
