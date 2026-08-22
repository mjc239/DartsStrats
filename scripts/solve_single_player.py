#!/usr/bin/env python
"""
Solve every single-player model at the shared ability bands and save the
results as machine-readable arrays.

Writes one ``.npz`` per band per objective into ``results/single_player/``,
plus a summary ``results/manifest_single_player.csv``.

Each ``.npz`` holds everything needed to answer "what should I aim at, and
what is it worth?" without re-solving:

    V1              (502,)      value at the start of a visit, by score
    pol1            (502,)      index into `points` of the optimal first dart
    pol2tab/pol3tab (502,)      optimal dart 2 / dart 3 where the visit's start
                                score no longer matters (see notebook 01)
    pol2low/pol3low (s, u)      the same below those thresholds, indexed by
                                [visit start, current score]
    V2tab/V3tab, V2low/V3low    the matching values
    points          (n, 2)      pixel coordinates of the aiming grid
    checkout_pct    (171,)      P(finish this visit) by starting score
    expected_score  (n,)        expected score of one dart at each aim point

``--nu`` solves the same bands with a Student-t throw instead of a Gaussian,
writing into ``results/student_t/`` and ``results/manifest_student_t.csv`` so
the Gaussian results stay where they are and the two can be compared. The band
is held fixed at the same three-dart average rather than the same sigma -- see
``darts.transitions.matched_scale`` for why that is not a free choice.

Usage:
    python scripts/solve_single_player.py                  # all bands
    python scripts/solve_single_player.py --bands league club
    python scripts/solve_single_player.py --grid           # the fine sigma grid
    python scripts/solve_single_player.py --nu 2.25 3 5    # Student-t throws
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from darts import players
from darts.mdp_3turn import ThreeDartMDP
from darts.transitions import matched_scale, transition_arrays

OBJECTIVES = {"darts": dict(dart_cost=1.0, turn_cost=0.0),
              "visits": dict(dart_cost=0.0, turn_cost=1.0)}


def checkout_percentages(model, probs, checkout_probs, scores, max_score=170):
    """
    Exact probability of finishing within one visit, for every starting score.

    Propagates the score distribution through the three darts of the visit
    under the solved policy; a bust or a non-double hit ends the visit.
    """
    out = np.zeros(max_score + 1)
    for start in range(2, max_score + 1):
        dist, win = {start: 1.0}, 0.0
        for dart in (1, 2, 3):
            nxt = {}
            for u, w in dist.items():
                i = model.policy(u, dart, start)
                p, cp = probs[i], checkout_probs[i]
                hit = scores == u
                if hit.any():
                    win += w * cp[hit].sum()
                valid = scores <= u - 2
                for sj, pj in zip(scores[valid], p[valid]):
                    if pj > 1e-12:
                        nxt[u - sj] = nxt.get(u - sj, 0.0) + w * pj
            dist = nxt
        out[start] = win
    return out


def solve_and_save(sigma, label, outdir, objectives=("darts", "visits"),
                   board_pixels=None, point_stride=None, game_start=None,
                   with_checkouts=True, nu=None, band=None):
    board_pixels = board_pixels or players.BOARD_PIXELS
    point_stride = point_stride or players.POINT_STRIDE_SINGLE
    game_start = game_start or players.GAME_START

    # A t of the same scale is a different player, so the band is carried over
    # by its three-dart average and the scale follows from that.
    scale = matched_scale(sigma, nu, board_pixels=board_pixels)
    tr = transition_arrays(board_pixels, scale, point_stride=point_stride, nu=nu)
    P, CP, S = tr["probs"], tr["checkout_probs"], tr["allowed_scores"]

    rows = []
    for obj in objectives:
        t0 = time.perf_counter()
        m = ThreeDartMDP(P, CP, S, game_start, **OBJECTIVES[obj]).solve()
        elapsed = time.perf_counter() - t0

        payload = dict(
            V1=m.V1, pol1=m.pol1,
            V2tab=m.V2tab, V3tab=m.V3tab, pol2tab=m.pol2tab, pol3tab=m.pol3tab,
            V2low=m.V2low, V3low=m.V3low, pol2low=m.pol2low, pol3low=m.pol3low,
            u2_indep=m.u2_indep, u3_indep=m.u3_indep, s_indep=m.s_indep,
            points=tr["points"], allowed_scores=S,
            expected_score=P @ S,
            sigma=sigma, board_pixels=board_pixels, point_stride=point_stride,
            game_start=game_start, objective=obj, label=label,
            nu=np.nan if nu is None else nu, scale_mm=scale,
        )
        if with_checkouts:
            payload["checkout_pct"] = checkout_percentages(m, P, CP, S)

        path = os.path.join(outdir, f"{label}_{obj}.npz")
        np.savez_compressed(path, **payload)

        row = {"band": band or label, "sigma_mm": sigma,
               "objective": obj, "aiming_points": len(tr["points"]),
               "value_501": round(-m.V1[501], 4),
               "seconds": round(elapsed, 1),
               "path": os.path.relpath(path)}
        if nu is not None:
            # Only for a t, so the Gaussian manifest keeps the columns it has.
            row["nu"], row["scale_mm"] = nu, round(scale, 4)
        if obj == "darts":
            row["three_dart_average"] = round(players.three_dart_average(-m.V1[501]), 2)
            row["darts_from_170"] = round(-m.V1[170], 3)
            row["darts_from_40"] = round(-m.V1[40], 3)
        rows.append(row)
        print(f"  {label:>13} {obj:>6}: {-m.V1[501]:7.3f}  ({elapsed:5.1f}s, "
              f"{len(tr['points'])} points)", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", nargs="*", default=None,
                    help="ability bands to solve (default: all)")
    ap.add_argument("--grid", action="store_true",
                    help="also solve the fine sigma grid (visits objective only, "
                         "no checkout tables) for sensitivity work")
    ap.add_argument("--outdir", default=None,
                    help="default: results/single_player, or results/student_t "
                         "when --nu is given")
    ap.add_argument("--nu", nargs="*", type=float, default=None,
                    help="solve with a Student-t throw at these degrees of "
                         "freedom instead of a Gaussian. 2.25 is what notebook "
                         "21 fitted; inf is the Gaussian and is a useful check")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    student_t = args.nu is not None
    default_out = "results/student_t" if student_t else "results/single_player"
    outdir = os.path.join(root, args.outdir or default_out)
    os.makedirs(outdir, exist_ok=True)

    bands = args.bands or list(players.ABILITY_BANDS)
    nus = args.nu if student_t else [None]
    print(f"board {players.BOARD_PIXELS}px, stride {players.POINT_STRIDE_SINGLE}, "
          f"{len(bands)} bands"
          + (f", nu in {nus}" if student_t else ""))

    rows = []
    for nu in nus:
        for band in bands:
            label = band if nu is None else f"{band}_nu{nu:g}"
            rows += solve_and_save(players.ABILITY_BANDS[band], label, outdir,
                                   nu=nu, band=band)

    import pandas as pd
    df = pd.DataFrame(rows)
    if student_t:
        lead = ["band", "sigma_mm", "nu", "scale_mm", "objective"]
        df = df[lead + [c for c in df.columns if c not in lead]]
    name = "manifest_student_t.csv" if student_t else "manifest_single_player.csv"
    manifest = os.path.join(root, "results", name)
    df.to_csv(manifest, index=False)
    print(f"\nwrote {manifest}")
    cols = ["band", "sigma_mm", "value_501", "three_dart_average"]
    if student_t:
        cols.insert(2, "nu")
        cols.insert(3, "scale_mm")
    print(df[df.objective == "darts"][cols].to_string(index=False))

    if args.grid:
        griddir = os.path.join(outdir, "grid")
        os.makedirs(griddir, exist_ok=True)
        print(f"\nfine grid: {len(players.SIGMA_GRID)} sigma values")
        grows = []
        for sigma in players.SIGMA_GRID:
            grows += solve_and_save(float(sigma), f"sigma{sigma:g}", griddir,
                                    objectives=("visits", "darts"),
                                    with_checkouts=False)
        pd.DataFrame(grows).to_csv(
            os.path.join(root, "results", "manifest_sigma_grid.csv"), index=False)
        print("wrote results/manifest_sigma_grid.csv")


if __name__ == "__main__":
    main()
