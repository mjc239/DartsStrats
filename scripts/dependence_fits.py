"""
Fit the dependence model family to real scoring visits, per player, held out.

Notebook 19 measured the three darts of a visit failing to be independent.
Notebook 20 asks what to replace the assumption with, and this script does the
fitting behind it. Six models are fitted to every player with enough visits, on
a training split of whole legs, and scored on the held-out rest.

The models are nested so that each comparison isolates one mechanism:

  A  gaussian, one target, independent darts   -- what the project assumes today
  B  + the aim rule (the treble 19 after a miss)
  C  + a contaminating wide component per dart (the dart that gets away)
  D  + a location offset shared by the visit    -- couples darts in direction
  E  + a scale shared by the visit              -- couples them in magnitude
  F  + both

A to B prices the aim rule, B to C the shape of a single dart, and C to D/E is
the question notebook 19 raised: once the aim and the tails are modelled, is
there any coupling left?

Writes ``results/dependence/fits.csv`` and ``results/dependence/signatures.csv``.
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

from darts.dependence import BedGrid, VisitModel, encode_visits, signatures
from darts.real_data import scoring_visits

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "real")
OUT = os.path.join(ROOT, "results", "dependence")

KEY = ["source", "player", "leg_id", "visit_index"]

MODELS = {
    "A gaussian iid": dict(switching=False),
    "B + aim rule": dict(switching=True),
    "C + wide tail": dict(switching=True, contamination=True),
    "D + shared offset": dict(switching=True, contamination=True, shared_offset=True),
    "E + shared scale": dict(switching=True, contamination=True, shared_scale=True),
}

# A seventh model -- both couplings at once -- was fitted and dropped. On the few
# hundred visits a single player supplies it is not identified: the optimiser
# lands with half the darts assigned to the wide component and sigma collapsed to
# 2.7mm, and it scores *worse* on held-out visits than either coupling alone.
# That is a statement about eight parameters and 500 visits, not about the
# mechanism, so reporting it as evidence either way would be misleading.


def load_visits(clean=True):
    """Pure-scoring visits the selection filter cannot bite on, cleaned.

    The cleaning is not cosmetic here. The 2017 feed carries the previous leg's
    finishing darts into the next leg's opening visit, and every visit with 430
    or more remaining is near the start of a leg, so that opening visit is most
    of this sample. Fitting a throw to it measures the contamination: it is what
    put the far tail into notebook 20's data and bought the wide component its
    five log-likelihood units a visit. See ``darts.real_data``.
    """
    return scoring_visits(clean=clean)


def split_by_leg(frame, seed=0):
    """Train/test split on whole legs.

    Visits inside one leg share a player, an evening and a scoreline, so
    splitting on visits would leak. Splitting on legs keeps the same players on
    both sides, which is what we want -- the question is whether a model
    generalises to new visits, not to new players.
    """
    legs = frame.leg_id.astype(str).unique()
    rng = np.random.default_rng(seed)
    test = set(rng.choice(legs, size=len(legs) // 2, replace=False))
    is_test = frame.leg_id.astype(str).isin(test).values
    return frame[~is_test], frame[is_test]


def fit_one(job, pixels, seed, n_sim, n_quad):
    """Fit every model to one player, and draw predictive checks from each fit.

    Self-contained so it can run in its own process: players are independent, so
    the whole study parallelises across them with nothing shared.
    """
    source, player, sub = job
    grid = BedGrid(pixels)
    train, test = split_by_leg(sub, seed=seed)
    b_tr, h_tr = encode_visits(train[[1, 2, 3]].values, grid)
    b_te, h_te = encode_visits(test[[1, 2, 3]].values, grid)
    if min(len(b_tr), len(b_te)) < 100:
        return [], []

    def sig_row(model_name, beds, hit, n):
        s = signatures(beds, hit, grid)
        return {"source": source, "player": player, "model": model_name, "n": n,
                **{k: v for k, v in s.items() if k != "p_k"},
                **{f"p_k{i}": s["p_k"][i] for i in range(4)}}

    fits = []
    sigs = [sig_row("OBSERVED", b_te, h_te, len(b_te))]
    print(f"{player} ({source}): {len(b_tr)} train / {len(b_te)} test", flush=True)
    for name, kw in MODELS.items():
        model = VisitModel(grid, n_quad=n_quad, **kw)
        t0 = time.time()
        res = model.fit(b_tr, h_tr)
        params = model.unpack(res.x)
        test_ll = model.log_likelihood(res.x, b_te, h_te)
        fits.append({
            "source": source, "player": player, "model": name,
            "n_train": len(b_tr), "n_test": len(b_te),
            "n_params": model.n_params,
            "train_ll": -res.fun, "test_ll": test_ll,
            "test_ll_per_visit": test_ll / len(b_te),
            "bic": model.n_params * np.log(len(b_tr)) + 2 * res.fun,
            "sigma": params["sigma"], "bias_x": params["bias"][0],
            "tau": params["tau"], "nu": params["nu"],
            "eps": params["eps"], "kappa": params["kappa"],
            "s_hit": params["s_hit"], "s_miss": params["s_miss"],
            "seconds": time.time() - t0, "nfev": int(res.nfev),
        })
        print("  %-18s %-14s logL/visit %8.4f  sigma %5.2f tau %5.2f nu %5.3f "
              "eps %.3f  (%.0fs)" % (player[:18], name, test_ll / len(b_te),
                                     params["sigma"], params["tau"], params["nu"],
                                     params["eps"], time.time() - t0), flush=True)

        rng = np.random.default_rng(seed + 1)
        sim_b, sim_h = model.simulate(res.x, n_sim, rng=rng)
        sigs.append(sig_row(name, sim_b, sim_h, n_sim))
    return fits, sigs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-visits", type=int, default=400,
                    help="players with fewer pure-scoring visits are skipped")
    ap.add_argument("--pixels", type=int, default=512)
    ap.add_argument("--n-sim", type=int, default=15000,
                    help="visits drawn from each fit for the predictive checks")
    ap.add_argument("--n-quad", type=int, default=7,
                    help="quadrature nodes per latent dimension. 7 is converged: "
                         "against 13 nodes the log-likelihood moves by 4e-5 per "
                         "visit, some 500x smaller than the smallest effect here")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dirty", action="store_true",
                    help="fit the uncleaned data, to price the contamination")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    visits = load_visits(clean=not args.dirty)

    counts = visits.groupby(["source", "player"]).size()
    keep = counts[counts >= args.min_visits].sort_values(ascending=False)
    print(f"{len(visits):,} visits; {len(keep)} players with "
          f">= {args.min_visits} of them", flush=True)

    work = [(source, player,
             visits[(visits.source == source) & (visits.player == player)])
            for (source, player), _ in keep.items()]
    run = functools.partial(fit_one, pixels=args.pixels, seed=args.seed,
                            n_sim=args.n_sim, n_quad=args.n_quad)
    if args.jobs > 1:
        with mp.Pool(args.jobs) as pool:
            results = pool.map(run, work, chunksize=1)
    else:
        results = [run(job) for job in work]

    fits = [row for player_fits, _ in results for row in player_fits]
    sigs = [row for _, player_sigs in results for row in player_sigs]

    pd.DataFrame(fits).to_csv(os.path.join(args.out, "fits.csv"), index=False)
    pd.DataFrame(sigs).to_csv(os.path.join(args.out, "signatures.csv"), index=False)
    print(f"\nwrote {args.out}/fits.csv and signatures.csv")


if __name__ == "__main__":
    main()
