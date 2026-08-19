#!/usr/bin/env python
"""
Why splitting a session helps far more than the asymptotics predict.

The Fisher calculation says spreading the darts over several targets is worth
about 9% of the variance. Simulated sessions of 100 darts say it is worth
considerably more than that. This script isolates the reason.

It is *not* that the aim point is an expensive nuisance parameter -- the
asymptotic cost of not knowing it is 1.04x at the best single target, i.e.
nothing. The effect is a finite-sample one, and it lives entirely in the lower
tail: from a single target, an unluckily tight group of darts is equally well
explained by "tight thrower, aiming where we thought" and by "ordinary thrower,
aiming somewhere else". The likelihood cannot separate those two stories, so
sigma is occasionally badly underestimated. Darts at a second and third target
kill the trade-off, because one displaced aim point cannot explain three
different score histograms at once.

The diagnostic is the correlation between the estimated sigma and the error in
the estimated aim point, which should be strongly negative for one target and
near zero for several.

Writes ``results/design/why_splitting.csv`` (per-replicate estimates).

Usage:
    python scripts/why_splitting_helps.py --reps 300
"""
import argparse
import os
import sys
from functools import partial
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from darts import players
from darts.dartboards import generate_dartboard
from darts.fitting import fit_multi_target, simulate_session

PIXELS = 512
TRUE_BIAS = np.array([2.0, -3.0])

_BOARD = None


def _board():
    global _BOARD
    if _BOARD is None:
        _BOARD = generate_dartboard(PIXELS)[0]
    return _BOARD


def one(seed, targets, n, sigma):
    board = _board()
    k = len(targets)
    base, rem = divmod(n, k)
    alloc = [base + (1 if i < rem else 0) for i in range(k)]
    sessions = simulate_session(targets, alloc, TRUE_BIAS,
                                sigma ** 2 * np.eye(2), board=board, seed=seed)
    best = None
    for s0 in (30.0, 12.0):
        try:
            f = fit_multi_target(sessions, board=board,
                                 Sigma_init=s0 ** 2 * np.eye(2), max_iter=400)
        except (np.linalg.LinAlgError, ValueError):
            continue
        if best is None or f["log_likelihood"] > best["log_likelihood"]:
            best = f
    if best is None:
        return np.nan, np.nan
    return best["sigma_mm"], float(np.hypot(*(best["b"] - TRUE_BIAS)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", default="league")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sigma = players.ABILITY_BANDS[args.band]
    d = np.load(os.path.join(root, "results", "design",
                             f"design_{args.band}.npz"), allow_pickle=True)

    designs = {}
    for k in (1, 2, 3):
        key = f"design_{k}_mm"
        if key in d:
            designs[f"best {k}"] = [np.asarray(p) for p in d[key]]

    import pandas as pd
    rows = []
    for name, targets in designs.items():
        with Pool(args.procs) as pool:
            out = np.array(pool.map(
                partial(one, targets=targets, n=args.n, sigma=sigma),
                range(2000, 2000 + args.reps)), dtype=float)
        ok = np.isfinite(out[:, 0])
        s, berr = out[ok, 0], out[ok, 1]
        lo = s < np.percentile(s, 25)
        q = np.percentile(s, [1, 5, 25, 50, 75, 95, 99])
        print(f"{name}: median {np.median(s):.2f}  IQR {q[2]:.2f}-{q[4]:.2f}  "
              f"5-95% {q[1]:.2f}-{q[5]:.2f}")
        print(f"   rmse {np.sqrt(((s - sigma) ** 2).mean()):.3f}   "
              f"corr(sigma_hat, aim error) {np.corrcoef(s, berr)[0, 1]:+.3f}   "
              f"aim error in the low-sigma quartile {berr[lo].mean():.2f}mm "
              f"vs {berr[~lo].mean():.2f}mm", flush=True)
        for si, bi in zip(s, berr):
            rows.append({"design": name, "k": len(targets), "n": args.n,
                         "band": args.band, "sigma_true": sigma,
                         "sigma_hat": si, "aim_error_mm": bi})

    df = pd.DataFrame(rows)
    path = os.path.join(root, "results", "design", "why_splitting.csv")
    df.to_csv(path, index=False)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
