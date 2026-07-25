#!/usr/bin/env python
"""
Solve a leg at every shared ability band, so match play can be costed.

For each band this writes ``results/two_player/leg_{band}.npz`` holding both
the one-dart and three-dart games, and appends to
``results/manifest_two_player.csv`` the two numbers the match model needs:
the probability of winning a leg throwing first, and receiving.

The three-dart leg takes roughly a quarter of an hour per band, so the script
checkpoints and can be resumed; bands already on disk are skipped unless
``--force`` is given.

Usage:
    python scripts/solve_legs_all_bands.py
    python scripts/solve_legs_all_bands.py --bands league club --one-dart-only
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from darts import players
from darts.mdp_2player import OneDartLeg, ThreeDartLeg, candidate_points
from darts.transitions import transition_arrays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", nargs="*", default=None)
    ap.add_argument("--one-dart-only", action="store_true",
                    help="skip the expensive three-dart leg")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--outdir", default="results/two_player")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outdir = os.path.join(root, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    bands = args.bands or list(players.ABILITY_BANDS)
    G = players.GAME_START
    rows = []

    for band in bands:
        sigma = players.ABILITY_BANDS[band]
        path = os.path.join(outdir, f"leg_{band}.npz")
        if os.path.exists(path) and not args.force:
            d = np.load(path)
            print(f"{band}: already solved, skipping")
            rows.append({"band": band, "sigma_mm": sigma,
                         "p_first_1dart": float(d["W_one"][G, G]),
                         "p_first_3dart": (float(d["W_three"][G, G])
                                           if "W_three" in d else None)})
            continue

        print(f"\n=== {band} (sigma {sigma} mm) ===", flush=True)
        tr = transition_arrays(players.BOARD_PIXELS, sigma,
                               point_stride=players.POINT_STRIDE_TWO)
        P, CP, S = tr["probs"], tr["checkout_probs"], tr["allowed_scores"]
        keep = candidate_points(P, CP, S, game_start=G)
        Pk = np.ascontiguousarray(P[keep])
        CPk = np.ascontiguousarray(CP[keep])
        print(f"  aiming points {len(tr['points'])} -> {len(keep)}", flush=True)

        t0 = time.perf_counter()
        one = OneDartLeg(Pk, CPk, S, G).solve()
        print(f"  one-dart leg:   {time.perf_counter()-t0:6.0f}s  "
              f"W[501,501] = {one.W[G, G]:.4f}", flush=True)

        payload = dict(W_one=one.W, policy_one=one.policy,
                       points=tr["points"][keep], allowed_scores=S,
                       sigma=sigma, band=band,
                       board_pixels=players.BOARD_PIXELS)

        if not args.one_dart_only:
            ck = path + ".checkpoint.npz"
            t0 = time.perf_counter()
            three = ThreeDartLeg(Pk, CPk, S, G).solve(
                progress=True, checkpoint_path=ck, checkpoint_every=25,
                resume=os.path.exists(ck))
            print(f"  three-dart leg: {time.perf_counter()-t0:6.0f}s  "
                  f"W[501,501] = {three.W[G, G]:.4f}", flush=True)
            payload.update(W_three=three.W, policy_three=three.policy,
                           Y2=three.Y2, Y3=three.Y3)
            if os.path.exists(ck):
                os.remove(ck)

        np.savez_compressed(path, **payload)
        rows.append({"band": band, "sigma_mm": sigma,
                     "p_first_1dart": float(one.W[G, G]),
                     "p_first_3dart": (float(payload["W_three"][G, G])
                                       if "W_three" in payload else None)})
        print(f"  wrote {os.path.relpath(path, root)}", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    manifest = os.path.join(root, "results", "manifest_two_player.csv")
    df.to_csv(manifest, index=False)
    print(f"\nwrote {manifest}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
