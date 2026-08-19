#!/usr/bin/env python
"""
Search for the aim points that measure a player most precisely.

For every shared ability band this finds:

* the best single target, exhaustively over the candidate grid;
* the best equally-weighted pair, exhaustively over all pairs;
* equally-weighted designs of 3 and 4 targets, by greedy selection with
  exchange refinement;
* the continuous optimal design, with the general equivalence theorem
  certificate that proves it optimal.

and evaluates each against a few targets a player would pick unprompted (the
bull, T20, D20), so the cost of choosing badly is visible.

Writes ``results/design/design_{band}.npz`` and the summary
``results/manifest_design.csv``.

Usage:
    python scripts/optimal_measurement_design.py
    python scripts/optimal_measurement_design.py --bands league club --stride 4
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from darts import players
from darts.dartboards import generate_dartboard
from darts.design import (best_pair, best_single_target, c_criterion,
                          candidate_targets, d_criterion, design_information,
                          equivalence_certificate, greedy_design,
                          information_at_points, information_maps,
                          optimal_design, sigma_gradient)
from darts.utils import aim_description, mm_per_pixel

BOARD_PIXELS = 512          # an 8mm bed is 9.1 pixels across here; at 256 it
                            # is 4.5, and the SE at a bed-centred target such as
                            # T20 is then off by more than 15%
N_REFERENCE = 200           # darts in a reference session, for quoting SEs


def named_targets(px):
    """A few targets a player would plausibly choose without being told."""
    mm = mm_per_pixel(px)
    c = px // 2

    def polar(r, deg):
        return (int(round(c + r * np.cos(np.deg2rad(deg)) / mm)),
                int(round(c + r * np.sin(np.deg2rad(deg)) / mm)))

    return {"bull": polar(0, 0), "T20": polar(103, 0), "D20": polar(166, 0),
            "20 outer single": polar(138, 0), "25 ring": polar(11, 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", nargs="*", default=None)
    ap.add_argument("--stride", type=int, default=4,
                    help="candidate grid stride in pixels")
    ap.add_argument("--pixels", type=int, default=BOARD_PIXELS)
    ap.add_argument("--outdir", default="results/design")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outdir = os.path.join(root, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    px = args.pixels
    board, _ = generate_dartboard(px)
    mm = mm_per_pixel(px)
    centre = px // 2
    cand = candidate_targets(px, point_stride=args.stride)
    named = named_targets(px)
    bands = args.bands or list(players.ABILITY_BANDS)

    def to_mm(p):
        return np.array([(p[1] - centre) * mm, (p[0] - centre) * mm])

    rows = []
    for band in bands:
        sigma = players.ABILITY_BANDS[band]
        print(f"\n=== {band} (sigma {sigma} mm) ===", flush=True)
        t0 = time.perf_counter()
        maps = information_maps(px, sigma, board=board)
        I_pts = information_at_points(maps, cand)
        c = sigma_gradient(sigma)
        print(f"  information maps in {time.perf_counter()-t0:.1f}s, "
              f"{len(cand)} candidates", flush=True)

        row = {"band": band, "sigma_mm": sigma}

        for name, p in named.items():
            I = maps["info"][p[0], p[1]]
            row[f"se_{name}"] = float(np.sqrt(c_criterion(I, c) / N_REFERENCE))

        i1, v1, _ = best_single_target(I_pts, c)
        row["se_best1"] = float(np.sqrt(v1 / N_REFERENCE))
        row["best1"] = aim_description(cand[i1], px)
        row["best1_r_mm"] = float(np.hypot(*to_mm(cand[i1])))

        t0 = time.perf_counter()
        (a, b), v2 = best_pair(I_pts, c)
        row["se_best2"] = float(np.sqrt(v2 / N_REFERENCE))
        row["best2"] = f"{aim_description(cand[a], px)} + {aim_description(cand[b], px)}"
        print(f"  best pair exhaustively in {time.perf_counter()-t0:.0f}s", flush=True)

        designs = {1: [i1], 2: [a, b]}
        for k in (3, 4, 6):
            idx, v = greedy_design(I_pts, c, k)
            designs[k] = idx
            row[f"se_best{k}"] = float(np.sqrt(v / N_REFERENCE))

        opt = optimal_design(I_pts, c)
        row["se_optimal"] = float(np.sqrt(opt["value"] / N_REFERENCE))
        row["certificate"] = opt["certificate"]
        row["support_size"] = int(len(opt["support"]))
        # how much of the total gain an equal-weight 4-target design captures
        row["ratio_single_over_optimal"] = float(v1 / opt["value"])
        row["cert_of_best_single"] = equivalence_certificate(I_pts, c, I_pts[i1])

        # D-optimality, where all five parameters are of interest
        row["logdet_best1"] = float(d_criterion(I_pts[i1]))
        d_idx, _ = greedy_design(I_pts, c, 4)
        row["logdet_best4"] = float(d_criterion(
            design_information(I_pts[d_idx], np.ones(4))))

        rows.append(row)
        print(f"  best single {row['best1']:>10s} (r={row['best1_r_mm']:.0f}mm)"
              f"  SE {row['se_best1']:.4f}", flush=True)
        print(f"  optimum SE {row['se_optimal']:.4f}  certificate "
              f"{row['certificate']:.6f}  support {row['support_size']}", flush=True)
        print(f"  T20 would give SE {row['se_T20']:.4f} "
              f"({(row['se_T20']/row['se_best1'])**2:.1f}x the variance)", flush=True)

        np.savez_compressed(
            os.path.join(outdir, f"design_{band}.npz"),
            sigma=sigma, band=band, board_pixels=px,
            candidates=cand, weights=opt["weights"], support=opt["support"],
            **{f"design_{k}": np.array(v) for k, v in designs.items()},
            **{f"design_{k}_mm": np.array([to_mm(cand[j]) for j in v])
               for k, v in designs.items()},
        )

        import pandas as pd
        pd.DataFrame(rows).to_csv(
            os.path.join(root, "results", "manifest_design.csv"), index=False)

    import pandas as pd
    df = pd.DataFrame(rows)
    path = os.path.join(root, "results", "manifest_design.csv")
    df.to_csv(path, index=False)
    print(f"\nwrote {path}")
    cols = ["band", "sigma_mm", "se_bull", "se_T20", "se_best1", "se_best2",
            "se_best4", "se_optimal", "certificate"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
