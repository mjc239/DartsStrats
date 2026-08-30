#!/usr/bin/env python
"""
Fit every player a tilted Student-t, with a standard error on each parameter.

Two experiments need the same expensive thing, so it is computed once here.

**Which way does a player lean?** Notebook 16 established that a lean matters --
0.61 visits a leg between the best and worst orientation of an identical ellipse
-- and that it is learnable from about 93 darts *at the bull*, because every
segment boundary meets there. It was entirely simulated, and it ends by saying
the prior on ``rho`` is a guess and nobody knows what real players' leans look
like. This measures them. The bull caveat matters: competition darts are thrown
at the treble 20 and its step-down neighbours, nowhere near the bull, so notebook
16's sample size does not transfer and the uncertainty has to be reported rather
than assumed small.

**Is one player seventeen times better than seventeen players?** Every fit in this
project has been per player and independent. That is the only way to discover
that players differ, but for a weakly measured parameter it is close to the worst
estimator available. The alternative pools partially -- see
:mod:`darts.hierarchical` -- and whether it is actually better is settled on
held-out legs against both of the things it sits between.

So each player gets, on their *training* legs only:

* the fitted tilted-t, seven parameters;
* the observed information, hence a standard error on each of them, with the
  finite-difference step checked at two sizes;
* the held-out log-likelihood of their own fit.

Plus one **completely pooled** fit to every player's training visits at once, and
each player's held-out likelihood under it. The partial-pooling estimate is a
cheap function of the per-player numbers, so it is left to the notebook, which is
also where the beds are re-used -- they are saved here so that any parameter
vector can be scored on held-out legs without refitting anything.

Writes ``results/heterogeneity/players.csv`` and ``visits.npz``.
Needs the real data -- see ``data/real/README.md``.
"""

import argparse
import functools
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from darts.dependence import encode_visits
from darts.hierarchical import parameter_covariance
from darts.real_data import scoring_visits
from darts.throw_families import FAMILIES, FamilyVisitModel, RadialBedGrid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "heterogeneity")

FAMILY = "tilted-t"
#: the parameter vector, in the order FamilyVisitModel packs it
PARAMS = ("log_scale", "bias_x", "log_nu_minus_2", "e1", "e2",
          "logit_s_hit", "logit_s_miss")


def split_by_leg(frame, seed=0):
    """Train/test on whole legs -- the same split the family comparison used."""
    legs = frame.leg_id.astype(str).unique()
    rng = np.random.default_rng(seed)
    test = set(rng.choice(legs, size=len(legs) // 2, replace=False))
    is_test = frame.leg_id.astype(str).isin(test).values
    return frame[~is_test], frame[is_test]


def encode(sub, grid, seed):
    train, test = split_by_leg(sub, seed=seed)
    return (encode_visits(train[[1, 2, 3]].values, grid),
            encode_visits(test[[1, 2, 3]].values, grid))


def fit_one(job, pixels, seed):
    """One player: fit on their training legs, then measure how well."""
    source, player, sub = job
    grid = RadialBedGrid(pixels)
    (b_tr, h_tr), (b_te, h_te) = encode(sub, grid, seed)
    if min(len(b_tr), len(b_te)) < 100:
        return None

    family = FAMILIES[FAMILY]
    model = FamilyVisitModel(family, grid)
    t0 = time.time()
    res = model.fit(b_tr, h_tr)
    theta = res.x
    params = model.unpack(theta)

    cov = parameter_covariance(lambda x: model.log_likelihood(x, b_tr, h_tr), theta)
    row = {"source": source, "player": player,
           "n_train": len(b_tr), "n_test": len(b_te),
           "train_ll": -res.fun,
           "test_ll": model.log_likelihood(theta, b_te, h_te),
           "scale": params["scale"], "seconds": time.time() - t0,
           "information_pd": cov["pd"],
           "step_sensitivity": cov["step_sensitivity"]}
    row.update(family.describe(params["shape"]))
    for i, name in enumerate(PARAMS):
        row[name] = float(theta[i])
        row[f"se_{name}"] = float(cov["se"][i]) if cov["pd"] else np.nan
    row["_theta"] = theta
    row["_cov"] = cov["cov"]
    row["_beds"] = (b_tr, h_tr, b_te, h_te)
    print(f"  {player[:20]:<20} rho {row['rho']:+.3f}  ratio {row['ratio']:.3f}  "
          f"tilt {row['tilt_deg']:5.1f}  nu {row['nu']:.2f}  "
          f"pd {cov['pd']}  ({time.time() - t0:.0f}s)", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-visits", type=int, default=400)
    ap.add_argument("--pixels", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    visits = scoring_visits(clean=True)
    counts = visits.groupby(["source", "player"]).size()
    keep = counts[counts >= args.min_visits].sort_values(ascending=False)
    print(f"{len(visits):,} visits; {len(keep)} players with "
          f">= {args.min_visits} of them\n", flush=True)

    work = [(source, player,
             visits[(visits.source == source) & (visits.player == player)])
            for (source, player), _ in keep.items()]
    run = functools.partial(fit_one, pixels=args.pixels, seed=args.seed)
    if args.jobs > 1:
        with mp.Pool(args.jobs) as pool:
            rows = pool.map(run, work, chunksize=1)
    else:
        rows = [run(job) for job in work]
    rows = [r for r in rows if r is not None]

    # the completely pooled fit: one parameter vector for everybody
    print("\npooling every player's training visits into one fit", flush=True)
    grid = RadialBedGrid(args.pixels)
    b_all = np.concatenate([r["_beds"][0] for r in rows])
    h_all = np.concatenate([r["_beds"][1] for r in rows])
    pooled_model = FamilyVisitModel(FAMILIES[FAMILY], grid)
    t0 = time.time()
    pooled = pooled_model.fit(b_all, h_all).x
    print(f"  {len(b_all):,} visits, {time.time() - t0:.0f}s", flush=True)
    for r in rows:
        r["pooled_test_ll"] = pooled_model.log_likelihood(
            pooled, r["_beds"][2], r["_beds"][3])

    # the beds, so any parameter vector can be scored later without refitting
    store = {"pooled_theta": pooled, "params": np.array(PARAMS),
             "players": np.array([r["player"] for r in rows])}
    for i, r in enumerate(rows):
        b_tr, h_tr, b_te, h_te = r.pop("_beds")
        store[f"{i}_b_tr"], store[f"{i}_h_tr"] = b_tr, h_tr
        store[f"{i}_b_te"], store[f"{i}_h_te"] = b_te, h_te
        store[f"{i}_theta"] = r.pop("_theta")
        cov = r.pop("_cov")
        store[f"{i}_cov"] = cov if cov is not None else np.full((7, 7), np.nan)
    np.savez_compressed(os.path.join(args.out, "visits.npz"), **store)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "players.csv"), index=False)
    print(f"\nwrote {args.out}/players.csv and visits.npz")
    print(df[["player", "rho", "ratio", "tilt_deg", "nu", "scale",
              "information_pd"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
