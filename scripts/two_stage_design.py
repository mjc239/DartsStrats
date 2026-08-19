#!/usr/bin/env python
"""
The circularity, resolved: measure roughly, then measure well.

The best target depends on the sigma being measured, which is not known until
after the session. The natural fix is to do it in two stages -- throw a first
batch at a target that works tolerably for everybody, get a rough sigma, then
spend the rest of the darts at the target that is best for *that* player.

This simulates the whole procedure, including the fact that stage two's target
is chosen from a noisy stage-one estimate, and compares it with:

* spending everything on the robust (minimax) design;
* spending everything at the bull, or at T20;
* the **oracle** -- spending everything at the target that is best for the
  player's true sigma, which nobody can actually do.

The oracle is the ceiling. The question is how close two stages get to it.

Writes ``results/design/two_stage.csv``.

Usage:
    python scripts/two_stage_design.py --bands league club pub
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
from darts.fitting import fit_multi_target, simulate_session

PIXELS = 512
TRUE_BIAS = np.array([2.0, -3.0])

_BOARD = None


def _board():
    global _BOARD
    if _BOARD is None:
        _BOARD = generate_dartboard(PIXELS)[0]
    return _BOARD


def target_for(sigma_hat, table):
    """
    The best single target for an estimated sigma, from the lookup table.

    ``table`` is passed explicitly rather than read from a module global so the
    workers do not depend on fork semantics -- macOS spawns instead, and would
    otherwise start them with an empty table.
    """
    sigmas, xs, ys = table
    i = int(np.argmin(np.abs(sigmas - sigma_hat)))
    return np.array([xs[i], ys[i]])


def run_one(seed, band_sigma, n_total, first_fraction, robust_mm, oracle_mm,
            table):
    """One simulated player, measured every way, returning sigma estimates."""
    board = _board()
    Sigma = band_sigma ** 2 * np.eye(2)
    rng = np.random.default_rng(seed)
    out = {}

    def fit(sessions):
        best = None
        for s0 in (30.0, 12.0):
            try:
                f = fit_multi_target(sessions, board=board,
                                     Sigma_init=s0 ** 2 * np.eye(2), max_iter=400)
            except (np.linalg.LinAlgError, ValueError):
                continue
            if best is None or f["log_likelihood"] > best["log_likelihood"]:
                best = f
        return np.nan if best is None else best["sigma_mm"]

    n1 = int(round(n_total * first_fraction))
    n2 = n_total - n1

    # --- two stage -------------------------------------------------------
    k = len(robust_mm)
    base, rem = divmod(n1, k)
    alloc = [base + (1 if i < rem else 0) for i in range(k)]
    stage1 = simulate_session(robust_mm, alloc, TRUE_BIAS, Sigma, board=board,
                              seed=int(rng.integers(1 << 31)))
    sigma_1 = fit(stage1)
    out["stage 1 only"] = sigma_1
    if np.isfinite(sigma_1):
        chosen = target_for(sigma_1, table)
        stage2 = simulate_session([chosen], [n2], TRUE_BIAS, Sigma, board=board,
                                  seed=int(rng.integers(1 << 31)))
        out["two stage"] = fit(stage1 + stage2)
        out["chosen_r_mm"] = float(np.hypot(*chosen))
    else:
        out["two stage"] = np.nan
        out["chosen_r_mm"] = np.nan

    # --- the fixed alternatives, same total darts ------------------------
    alts = {"robust (all darts)": robust_mm,
            "oracle": [oracle_mm],
            "bull": [np.zeros(2)],
            "T20": [np.array([0.0, 103.0])]}
    for name, targets in alts.items():
        k = len(targets)
        base, rem = divmod(n_total, k)
        alloc = [base + (1 if i < rem else 0) for i in range(k)]
        s = simulate_session(targets, alloc, TRUE_BIAS, Sigma, board=board,
                             seed=int(rng.integers(1 << 31)))
        out[name] = fit(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", nargs="*", default=["county", "league", "pub"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--first-fraction", type=float, default=0.25)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    import pandas as pd

    table_path = os.path.join(root, "results", "manifest_best_target.csv")
    if not os.path.exists(table_path):
        raise SystemExit("run scripts/best_target_by_sigma.py first")
    tbl = pd.read_csv(table_path)
    table = (tbl["sigma_mm"].values, tbl["best_x_mm"].values,
             tbl["best_y_mm"].values)

    robust = np.load(os.path.join(root, "results", "design", "robust.npz"),
                     allow_pickle=True)
    robust_mm = [np.asarray(p) for p in robust["robust_3_mm"]]
    print("robust stage-one design (mm):",
          [f"({p[0]:.0f},{p[1]:.0f})" for p in robust_mm])

    rows = []
    for band in args.bands:
        sigma = players.ABILITY_BANDS[band]
        oracle_mm = target_for(sigma, table)
        print(f"\n=== {band} (sigma {sigma}) oracle target r="
              f"{np.hypot(*oracle_mm):.0f}mm ===", flush=True)
        t0 = time.perf_counter()
        fn = partial(run_one, band_sigma=sigma, n_total=args.n,
                     first_fraction=args.first_fraction, robust_mm=robust_mm,
                     oracle_mm=oracle_mm, table=table)
        with Pool(args.procs) as pool:
            out = pool.map(fn, range(5000, 5000 + args.reps))

        for method in ["T20", "bull", "robust (all darts)", "stage 1 only",
                       "two stage", "oracle"]:
            s = np.array([o[method] for o in out], dtype=float)
            s = s[np.isfinite(s)]
            rows.append({
                "band": band, "sigma_true": sigma, "n": args.n,
                "method": method,
                "bias_pct": float(100 * (s.mean() - sigma) / sigma),
                "sd": float(s.std(ddof=1)),
                "rmse": float(np.sqrt(((s - sigma) ** 2).mean())),
                "n_ok": int(len(s)),
            })
            print(f"  {method:>20s}: rmse {rows[-1]['rmse']:.3f}  "
                  f"bias {rows[-1]['bias_pct']:+6.1f}%", flush=True)
        print(f"  [{time.perf_counter()-t0:.0f}s]", flush=True)

    df = pd.DataFrame(rows)
    path = os.path.join(root, "results", "design", "two_stage.csv")
    df.to_csv(path, index=False)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
