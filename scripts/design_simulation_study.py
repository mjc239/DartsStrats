#!/usr/bin/env python
"""
Test the design predictions against actual fits at realistic session lengths.

The Fisher information calculation in :mod:`darts.design` is asymptotic: it
says what the *variance* of the estimate tends to as the number of darts grows.
A practice session is 100 to 1000 darts, the estimator is biased there, and
sigma is bounded below by zero, so the asymptotics are a prediction and not a
guarantee.

This simulates whole sessions under each design, fits every one by EM, and
reports the bias, spread and root mean squared error that actually result --
which is the number a player cares about, and the honest test of whether
splitting a session across targets is worth doing.

Every design is fitted with the *same* estimator (:func:`fit_multi_target`), so
the comparison isolates the design and not the fitting code.

Writes ``results/design/simulation_{band}.csv``.

Usage:
    python scripts/design_simulation_study.py --bands league
    python scripts/design_simulation_study.py --bands league --n 100 200 400 1000
"""
import argparse
import os
import sys
import time
from functools import partial
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from darts import players
from darts.dartboards import generate_dartboard
from darts.design import (c_criterion, design_information, information_maps,
                          sigma_gradient)
from darts.fitting import fit_multi_target, simulate_session
from darts.utils import mm_per_pixel

PIXELS = 256
TRUE_BIAS = np.array([2.0, -3.0])      # a player who pulls right and low

_BOARD = None


def _board():
    global _BOARD
    if _BOARD is None:
        _BOARD = generate_dartboard(PIXELS)[0]
    return _BOARD


def allocate(n, k):
    """Split ``n`` darts as evenly as possible over ``k`` targets."""
    base, rem = divmod(n, k)
    return [base + (1 if i < rem else 0) for i in range(k)]


def one_replicate(seed, targets_mm, n, sigma, starts):
    """Simulate one session and fit it, returning the estimated sigma."""
    board = _board()
    Sigma = sigma ** 2 * np.eye(2)
    sessions = simulate_session(targets_mm, allocate(n, len(targets_mm)),
                                TRUE_BIAS, Sigma, board=board, seed=seed)
    best = None
    # EM on score data can have local optima; try a couple of starting spreads
    # and keep the better likelihood, so the comparison between designs is not
    # contaminated by one design being unluckier with the default start.
    for s0 in starts:
        try:
            f = fit_multi_target(sessions, board=board,
                                 Sigma_init=s0 ** 2 * np.eye(2), max_iter=400)
        except (np.linalg.LinAlgError, ValueError):
            continue
        if best is None or f["log_likelihood"] > best["log_likelihood"]:
            best = f
    if best is None:
        return np.nan, np.nan, np.nan
    return best["sigma_mm"], best["b"][0], best["b"][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", nargs="*", default=["league"])
    ap.add_argument("--n", nargs="*", type=int, default=[200])
    ap.add_argument("--reps", type=int, default=250)
    ap.add_argument("--starts", nargs="*", type=float, default=[30.0, 12.0])
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--designdir", default="results/design")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mm = mm_per_pixel(PIXELS)
    centre = PIXELS // 2

    def polar(r, deg):
        return np.array([r * np.sin(np.deg2rad(deg)), r * np.cos(np.deg2rad(deg))])

    import pandas as pd

    for band in args.bands:
        sigma = players.ABILITY_BANDS[band]
        path = os.path.join(root, args.designdir, f"design_{band}.npz")
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}; run optimal_measurement_design.py first")
        d = np.load(path, allow_pickle=True)

        # designs a player might pick unprompted, and the optimised ones
        designs = {
            "T20": [polar(103, 0)],
            "bull": [np.array([0.0, 0.0])],
            "D20": [polar(166, 0)],
        }
        for k in (1, 2, 3, 4, 6):
            key = f"design_{k}_mm"
            if key in d:
                designs[f"best {k}"] = [np.asarray(p) for p in d[key]]

        maps = information_maps(PIXELS, sigma, board=_board())
        c = sigma_gradient(sigma)

        def fisher_se(targets_mm, n):
            idx = [(int(round(centre + t[1] / mm)), int(round(centre + t[0] / mm)))
                   for t in targets_mm]
            I = np.stack([maps["info"][i, j] for i, j in idx])
            w = np.array(allocate(n, len(idx)), dtype=float)
            return float(np.sqrt(c_criterion(design_information(I, w), c) / n))

        rows = []
        for n in args.n:
            for name, targets in designs.items():
                t0 = time.perf_counter()
                seeds = list(range(10_000, 10_000 + args.reps))
                fn = partial(one_replicate, targets_mm=targets, n=n,
                             sigma=sigma, starts=args.starts)
                with Pool(args.procs) as pool:
                    out = pool.map(fn, seeds)
                out = np.array(out, dtype=float)
                s = out[:, 0]
                ok = np.isfinite(s)
                s = s[ok]
                rows.append({
                    "band": band, "sigma_true": sigma, "n": n, "design": name,
                    "k": len(targets),
                    "mean_sigma_hat": float(s.mean()),
                    "bias": float(s.mean() - sigma),
                    "bias_pct": float(100 * (s.mean() - sigma) / sigma),
                    "sd": float(s.std(ddof=1)),
                    "rmse": float(np.sqrt(((s - sigma) ** 2).mean())),
                    "fisher_se": fisher_se(targets, n),
                    "bias_x": float(np.nanmean(out[ok, 1]) - TRUE_BIAS[0]),
                    "bias_y": float(np.nanmean(out[ok, 2]) - TRUE_BIAS[1]),
                    "n_ok": int(ok.sum()),
                })
                r = rows[-1]
                print(f"{band} n={n:5d} {name:>8s} (k={r['k']}): "
                      f"rmse {r['rmse']:.3f}  bias {r['bias_pct']:+6.1f}%  "
                      f"sd {r['sd']:.3f}  fisher {r['fisher_se']:.3f}  "
                      f"[{time.perf_counter()-t0:.0f}s]", flush=True)

        df = pd.DataFrame(rows)
        out_path = os.path.join(root, args.designdir, f"simulation_{band}.csv")
        df.to_csv(out_path, index=False)
        print(f"wrote {out_path}\n")


if __name__ == "__main__":
    main()
