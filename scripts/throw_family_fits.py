"""
Which distribution is a dart's landing point? Fitted per player, held out.

Notebook 20 needed a second, much wider Gaussian mixed in before real beds could
be fitted at all. This script asks whether that patch is the right description of
a throw, or merely the first thing that worked, by fitting a family of candidates
on equal terms.

The design crosses the shape of **one dart's** distribution with whether the
visit shares a scale.

                              per-dart family
              gaussian  elliptical  exp-power  student-t  core+uniform  two-comp
  per-visit
    none         .          .           .          .           .           .
    shared       .                                 .

The two axes are the same mechanism at different timescales, which is the point.
A Student-t *is* a Gaussian whose width is redrawn -- for every dart. Notebook
20's shared scale is a Gaussian whose width is redrawn -- once per visit. Fitting
both, and both together, asks where the extra spread actually lives: in the dart,
in the visit, or in each separately.

``elliptical`` is the odd one out: it is not a tail at all but the rival
explanation of the same excess, priced at the same one parameter, so that "the
tail is heavy" has something to beat.

Fits are per player on a training split of whole legs and scored on the held-out
rest, exactly as ``dependence_fits.py`` does, so the numbers are comparable with
notebook 20's. Everything else -- the four-target aim rule, the sideways-only
bias -- is held fixed. Isotropy is held fixed for every model *except* the
elliptical one, which exists to test it.

Writes ``results/throw_family/fits.csv``. Needs the real data; see
``data/real/README.md``.
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

from darts.dependence import encode_visits, signatures
from darts.real_data import scoring_visits
from darts.throw_families import FAMILIES, RadialBedGrid, FamilyVisitModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "real")
OUT = os.path.join(ROOT, "results", "throw_family")
KEY = ["source", "player", "leg_id", "visit_index"]

MODELS = [
    ("gaussian", False),
    ("elliptical", False),
    ("exp-power", False),
    ("student-t", False),
    ("core+uniform", False),
    ("two-component", False),
    # added after notebook 21, and both are replies to it: the ellipse it priced
    # against a heavy tail was only ever tested with a Gaussian core, and five of
    # its seventeen players sat on the nu > 2 boundary a finite variance imposes
    ("elliptical-t", False),
    ("nig", False),
    ("gaussian", True),
    ("student-t", True),
]


def load_visits(clean=True):
    """Pure-scoring visits, with the leaked checkout darts removed.

    See ``darts.real_data``: the 2017 feed carries the previous leg's finishing
    darts into the next leg's opening visit, and that opening visit is most of
    this sample. Fitting a throw to it measures the contamination.
    """
    return scoring_visits(clean=clean)


def split_by_leg(frame, seed=0):
    """Train/test on whole legs, so no visit's own leg is on both sides."""
    legs = frame.leg_id.astype(str).unique()
    rng = np.random.default_rng(seed)
    test = set(rng.choice(legs, size=len(legs) // 2, replace=False))
    is_test = frame.leg_id.astype(str).isin(test).values
    return frame[~is_test], frame[is_test]


def fit_one(job, pixels, seed, n_quad, n_sim, models=None):
    source, player, sub = job
    grid = RadialBedGrid(pixels)
    train, test = split_by_leg(sub, seed=seed)
    b_tr, h_tr = encode_visits(train[[1, 2, 3]].values, grid)
    b_te, h_te = encode_visits(test[[1, 2, 3]].values, grid)
    if min(len(b_tr), len(b_te)) < 100:
        return []

    observed = signatures(b_te, h_te, grid.grid)
    rows = []
    for family_name, shared in (MODELS if models is None else models):
        family = FAMILIES[family_name]
        model = FamilyVisitModel(family, grid, shared_scale=shared, n_quad=n_quad)
        t0 = time.time()
        res = model.fit(b_tr, h_tr)
        params = model.unpack(res.x)
        test_ll = model.log_likelihood(res.x, b_te, h_te)
        shape = family.describe(params["shape"])
        row = {
            "source": source, "player": player,
            "family": family_name, "shared_scale": shared,
            "model": family_name + (" + visit scale" if shared else ""),
            "n_train": len(b_tr), "n_test": len(b_te),
            "n_params": model.n_params,
            "train_ll": -res.fun, "test_ll": test_ll,
            "test_ll_per_visit": test_ll / len(b_te),
            "scale": params["scale"],
            "axis_sd": family.axis_sd(params["scale"], params["shape"]),
            "bias_x": params["bias"][0],
            "nu_visit": params["nu_visit"],
            "s_hit": params["s_hit"], "s_miss": params["s_miss"],
            "at_boundary": bool(family.is_gaussian(params["shape"]))
                           or bool(shape.get("nu", 99) < 2.05),
            "seconds": time.time() - t0, "nfev": int(res.nfev),
            **{f"shape_{k}": v for k, v in shape.items()},
        }
        # does the fitted model reproduce what the player actually did?
        sim_b, sim_h = model.simulate(res.x, n_sim,
                                      rng=np.random.default_rng(seed + 1))
        sim = signatures(sim_b, sim_h, grid.grid)
        for stat in ("p_t20", "t20_lift_12", "mag_corr"):
            row[f"sim_{stat}"] = sim[stat]
            row[f"obs_{stat}"] = observed[stat]
        # the far tail is the thing the families disagree about
        far = [grid.names.index(n) for n in ("D20", "MISS", "D19", "D18")]
        row["sim_far"] = float(np.isin(sim_b, far).mean())
        row["obs_far"] = float(np.isin(b_te, far).mean())
        rows.append(row)
        print("  %-18s %-22s logL/visit %8.4f  axis sd %7.2f  %s (%.0fs)"
              % (player[:18], row["model"], row["test_ll_per_visit"],
                 row["axis_sd"], {k: round(v, 3) for k, v in shape.items()},
                 row["seconds"]), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-visits", type=int, default=400)
    ap.add_argument("--pixels", type=int, default=512)
    ap.add_argument("--n-quad", type=int, default=7)
    ap.add_argument("--n-sim", type=int, default=8000)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", default=None,
                    help="comma-separated family names to fit, for adding a "
                         "candidate without refitting the rest")
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
    models = MODELS
    if args.models:
        wanted = {m.strip() for m in args.models.split(",")}
        models = [m for m in MODELS if m[0] in wanted]
        print(f"fitting only: {[m[0] for m in models]}", flush=True)
    run = functools.partial(fit_one, pixels=args.pixels, seed=args.seed,
                            n_quad=args.n_quad, n_sim=args.n_sim, models=models)
    if args.jobs > 1:
        with mp.Pool(args.jobs) as pool:
            results = pool.map(run, work, chunksize=1)
    else:
        results = [run(job) for job in work]

    rows = [r for player_rows in results for r in player_rows]
    out_path = os.path.join(args.out, "fits.csv")
    new = pd.DataFrame(rows)
    if args.models and os.path.exists(out_path):
        # adding a candidate: keep the fits already made, replace only these
        old = pd.read_csv(out_path)
        old = old[~old.model.isin(new.model.unique())]
        new = pd.concat([old, new], ignore_index=True)
        print(f"merged with {len(old)} existing rows", flush=True)
    new.to_csv(out_path, index=False)
    print(f"\nwrote {args.out}/fits.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()
