#!/usr/bin/env python
"""
A measurement routine to hand to a player whose ability you do not yet know.

The optimal design depends on the sigma being measured, which is circular: you
must choose the targets before you have the answer. This quantifies the damage
and finds a design that hedges.

Produces two things:

* a **cross-efficiency matrix** -- the design built for band A, used on a
  player who is really band B, relative to the design built for B. This is the
  cost of guessing wrong;
* a **minimax design** -- the equally-weighted set of targets whose worst-case
  efficiency over the whole ability range is highest, i.e. the routine to use
  when you know nothing about the player.

Writes ``results/manifest_design_robust.csv`` and
``results/design/robust.npz``.

Usage:
    python scripts/robust_measurement_design.py
    python scripts/robust_measurement_design.py --k 1 2 3 4
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from darts import players
from darts.dartboards import generate_dartboard
from darts.design import (c_criterion, candidate_targets, design_information,
                          information_at_points, information_maps,
                          optimal_design, robust_design, sigma_gradient)
from darts.utils import aim_description, mm_per_pixel

PIXELS = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", nargs="*", default=None)
    ap.add_argument("--k", nargs="*", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    board, _ = generate_dartboard(PIXELS)
    mm = mm_per_pixel(PIXELS)
    centre = PIXELS // 2
    cand = candidate_targets(PIXELS, point_stride=args.stride)
    bands = args.bands or list(players.ABILITY_BANDS)

    def to_mm(p):
        return np.array([(p[1] - centre) * mm, (p[0] - centre) * mm])

    scenarios, local_best, local_idx = [], {}, {}
    for band in bands:
        sigma = players.ABILITY_BANDS[band]
        t0 = time.perf_counter()
        maps = information_maps(PIXELS, sigma, board=board)
        I_pts = information_at_points(maps, cand)
        c = sigma_gradient(sigma)
        opt = optimal_design(I_pts, c)
        scenarios.append((I_pts, c, opt["value"]))
        local_best[band] = opt["value"]
        vals = c_criterion(I_pts, c)
        local_idx[band] = int(np.argmin(vals))
        print(f"{band:>13s} sigma {sigma:4.1f}  local optimum "
              f"{opt['value']:.4f}  best single "
              f"{aim_description(cand[local_idx[band]], PIXELS):>8s}"
              f"  ({time.perf_counter()-t0:.0f}s)", flush=True)

    # --- cross efficiency: best single target for A, used on B -------------
    print("\ncross efficiency (rows: design built for; cols: player really is)")
    cross = np.zeros((len(bands), len(bands)))
    for i, a in enumerate(bands):
        for j, b in enumerate(bands):
            I_pts, c, best = scenarios[j]
            cross[i, j] = best / c_criterion(I_pts[local_idx[a]], c)
    header = "".join(f"{b[:6]:>8s}" for b in bands)
    print(f"{'':>14s}{header}")
    for i, a in enumerate(bands):
        print(f"{a:>14s}" + "".join(f"{cross[i, j]:8.3f}" for j in range(len(bands))))

    # --- naive choices, for reference --------------------------------------
    def polar(r, deg):
        return (int(round(centre + r * np.cos(np.deg2rad(deg)) / mm)),
                int(round(centre + r * np.sin(np.deg2rad(deg)) / mm)))

    naive = {"bull": polar(0, 0), "T20": polar(103, 0), "D20": polar(166, 0)}
    lookup = {tuple(p): i for i, p in enumerate(map(tuple, cand))}
    print("\nworst-case efficiency of an unconsidered choice:")
    naive_eff = {}
    for name, p in naive.items():
        idx = lookup.get(tuple(p))
        if idx is None:
            continue
        effs = [best / c_criterion(I_pts[idx], c) for I_pts, c, best in scenarios]
        naive_eff[name] = effs
        print(f"  {name:>5s}: worst {min(effs):.3f}   " +
              " ".join(f"{b[:4]} {e:.2f}" for b, e in zip(bands, effs)))

    # --- the minimax designs -----------------------------------------------
    rows, payload = [], {}
    print("\nminimax designs over the whole ability range:")
    for k in args.k:
        t0 = time.perf_counter()
        idx, worst, effs = robust_design(scenarios, k)
        labs = " + ".join(aim_description(cand[j], PIXELS) for j in idx)
        radii = [float(np.hypot(*to_mm(cand[j]))) for j in idx]
        print(f"  k={k}: worst-case efficiency {worst:.3f}   {labs}")
        print(f"        radii {['%.0f' % r for r in radii]} mm   "
              f"per band " + " ".join(f"{b[:4]} {e:.2f}"
                                      for b, e in zip(bands, effs))
              + f"   ({time.perf_counter()-t0:.0f}s)", flush=True)
        row = {"k": k, "worst_efficiency": worst, "targets": labs,
               "radii_mm": ";".join(f"{r:.0f}" for r in radii)}
        row.update({f"eff_{b}": e for b, e in zip(bands, effs)})
        rows.append(row)
        payload[f"robust_{k}"] = np.array([cand[j] for j in idx])
        payload[f"robust_{k}_mm"] = np.array([to_mm(cand[j]) for j in idx])

    import pandas as pd
    df = pd.DataFrame(rows)
    path = os.path.join(root, "results", "manifest_design_robust.csv")
    df.to_csv(path, index=False)
    pd.DataFrame(cross, index=bands, columns=bands).to_csv(
        os.path.join(root, "results", "manifest_design_cross.csv"))
    np.savez_compressed(os.path.join(root, "results", "design", "robust.npz"),
                        candidates=cand, cross=cross, bands=np.array(bands),
                        **payload)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
