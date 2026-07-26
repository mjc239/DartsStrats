#!/usr/bin/env python
"""
Build the lookup table "given a player's sigma, where should they throw to be
measured?".

This is the machine-readable form of the design result: one row per sigma on
the shared grid, giving the best single target, the best equally-weighted pair,
and the standard error each achieves for a reference session. It is what the
two-stage routine consults after its first batch of darts, and it is the table
to read off if you already have an estimate of a player's ability.

Writes ``results/manifest_best_target.csv``.

Usage:
    python scripts/best_target_by_sigma.py
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from darts import players
from darts.dartboards import generate_dartboard
from darts.design import (best_single_target, c_criterion, candidate_targets,
                          greedy_design, information_at_points,
                          information_maps, optimal_design, sigma_gradient)
from darts.utils import aim_description, mm_per_pixel

PIXELS = 256
N_REF = 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--sigmas", nargs="*", type=float, default=None)
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    board, _ = generate_dartboard(PIXELS)
    mm = mm_per_pixel(PIXELS)
    centre = PIXELS // 2
    cand = candidate_targets(PIXELS, point_stride=args.stride)
    sigmas = args.sigmas or list(players.SIGMA_GRID)

    rows = []
    t0 = time.perf_counter()
    for sigma in sigmas:
        maps = information_maps(PIXELS, float(sigma), board=board)
        I_pts = information_at_points(maps, cand)
        c = sigma_gradient(float(sigma))
        i1, v1, _ = best_single_target(I_pts, c)
        pair, v2 = greedy_design(I_pts, c, 2)
        opt = optimal_design(I_pts, c)

        p = cand[i1]
        rows.append({
            "sigma_mm": float(sigma),
            "best_target": aim_description(p, PIXELS),
            "best_row": int(p[0]), "best_col": int(p[1]),
            "best_x_mm": float((p[1] - centre) * mm),
            "best_y_mm": float((p[0] - centre) * mm),
            "best_r_mm": float(np.hypot((p[1] - centre) * mm, (p[0] - centre) * mm)),
            "se_best1": float(np.sqrt(v1 / N_REF)),
            "pair_targets": " + ".join(aim_description(cand[j], PIXELS) for j in pair),
            "pair_x_mm": ";".join(f"{(cand[j][1]-centre)*mm:.1f}" for j in pair),
            "pair_y_mm": ";".join(f"{(cand[j][0]-centre)*mm:.1f}" for j in pair),
            "se_best2": float(np.sqrt(v2 / N_REF)),
            "se_optimal": float(np.sqrt(opt["value"] / N_REF)),
            "certificate": float(opt["certificate"]),
            "se_bull": float(np.sqrt(
                c_criterion(maps["info"][centre, centre], c) / N_REF)),
        })
        print(f"  sigma {sigma:5.1f}: {rows[-1]['best_target']:>8s} "
              f"(r={rows[-1]['best_r_mm']:5.1f}mm)  SE {rows[-1]['se_best1']:.4f}",
              flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    path = os.path.join(root, "results", "manifest_best_target.csv")
    df.to_csv(path, index=False)
    print(f"\nwrote {path} in {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
