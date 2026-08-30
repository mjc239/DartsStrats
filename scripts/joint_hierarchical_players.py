#!/usr/bin/env python
"""
The hierarchical model over players fitted properly, instead of from summaries.

Everything in notebook 26 pools in **two stages**: each player's likelihood is
collapsed to a maximum and a curvature by ``player_heterogeneity.py``, and the
population is then fitted to those seventeen summaries as though they were the
data. That is the standard thing to do and it is an approximation twice over --
it treats each player's likelihood as Gaussian around its *unpenalised* maximum,
and it treats the curvature there as known.

Two of its costs are visible in this project rather than hypothetical.

**Five players are thrown away.** The ones whose ``nu`` ran to the ``nu > 2``
clip have a likelihood that is flat in that direction, no covariance to
summarise, and so no way into a two-stage fit -- the joint model over all seven
parameters ran on **12** of the 17. Here the curvature at the penalised mode is
``-grad^2 L_p + T^-1``, positive definite whenever ``T`` is, so all seventeen are
ordinary.

**The modes cannot move.** Two-stage shrinkage averages a player's answer with
the population's *after* the fact. Fitting jointly puts the population in as a
prior and re-finds each player's mode under it, which is a different estimate,
not a weighted version of the same one.

Fitted by EM with a Laplace approximation to the inner integral, accelerated by
SQUAREM -- see :func:`darts.hierarchical.joint_hierarchical`, where plain EM
needed 800 iterations on a synthetic case the accelerated version does in 11.

Scores the penalised modes on the same held-out legs everything else in notebook
26 is scored on, so the number is comparable with no pooling, coordinate-wise
partial pooling, the two-stage joint fit, and complete pooling.

Reads ``results/heterogeneity/visits.npz`` -- nothing is refitted from the raw
data and nothing is re-scraped. Writes ``joint.npz`` alongside it.

It runs in **chunks**, because each outer iteration re-finds seventeen penalised
modes in seven dimensions and the whole fit is hours rather than minutes. Every
time the fit reaches a better population than it has seen, ``joint_state.npz``
is rewritten with it; ``--resume`` starts from that file. So the run can be
killed at any point and loses at most one E step:

    until python scripts/joint_hierarchical_players.py --resume --max-iter 8 \
        | grep -q "^converged"; do :; done

Resuming costs one iteration's worth of SQUAREM momentum and nothing else -- the
EM map is the same map wherever it is entered from.
"""

import argparse
import os
import sys
import time
from multiprocessing.dummy import Pool as ThreadPool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from darts.hierarchical import joint_hierarchical
from darts.throw_families import FAMILIES, FamilyVisitModel, RadialBedGrid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "heterogeneity")

FAMILY = "tilted-t"
PARAMS = ("log_scale", "bias_x", "log_nu_minus_2", "e1", "e2",
          "logit_s_hit", "logit_s_miss")


def load(path, pixels=512):
    """Per-player training and held-out likelihoods, plus the per-player fits."""
    df = pd.read_csv(os.path.join(path, "players.csv"))
    z = np.load(os.path.join(path, "visits.npz"), allow_pickle=True)
    model = FamilyVisitModel(FAMILIES[FAMILY], RadialBedGrid(pixels))
    n = len(df)

    def make(i, which):
        b, h = z[f"{i}_b_{which}"], z[f"{i}_h_{which}"]
        return lambda x: model.log_likelihood(x, b, h)

    train = [make(i, "tr") for i in range(n)]
    test = [make(i, "te") for i in range(n)]
    theta = np.stack([z[f"{i}_theta"] for i in range(n)])
    return df, z, theta, train, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=OUT, help="a results/heterogeneity[/seedN]")
    ap.add_argument("--pixels", type=int, default=512)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--max-iter", type=int, default=60)
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--no-accelerate", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="continue from joint_state.npz if it is there")
    args = ap.parse_args()

    df, z, theta, train, test = load(args.dir, args.pixels)
    state_path = os.path.join(args.dir, "joint_state.npz")
    mu = T = None
    start = theta
    if args.resume and os.path.exists(state_path):
        s = np.load(state_path)
        mu, T, start = s["mu"], s["T"], s["theta"]
        print(f"resuming from {state_path} after {int(s['iters_so_far'])} "
              f"iterations", flush=True)
        iters_so_far = int(s["iters_so_far"])
    else:
        iters_so_far = 0

    n_test = df.n_test.values
    base = np.array([test[i](theta[i]) for i in range(len(df))])
    pooled = np.tile(z["pooled_theta"], (len(df), 1))
    pooled_ll = np.array([test[i](pooled[i]) for i in range(len(df))])

    print(f"{len(df)} players, {df.n_train.sum():,} training visits, "
          f"{n_test.sum():,} held out")
    print(f"no pooling       {base.sum():10.1f}")
    print(f"complete pooling {pooled_ll.sum():10.1f}  "
          f"{pooled_ll.sum() - base.sum():+8.1f} vs no pooling\n", flush=True)

    t0 = time.time()
    seen = [iters_so_far]

    trace_path = os.path.join(args.dir, "joint_trace.csv")
    if not os.path.exists(trace_path):
        with open(trace_path, "w") as fh:
            fh.write("iteration,marginal,held_out_vs_no_pooling\n")

    def save(mu_v, T_v, theta_v, marginal):
        seen[0] += 1
        np.savez_compressed(state_path, mu=mu_v, T=T_v, theta=theta_v,
                            iters_so_far=seen[0], marginal=marginal,
                            converged=False)
        # Scoring the held-out legs costs 17 likelihood evaluations against the
        # ~7,000 the E step just spent, and it is the only way to know whether
        # stopping early would have invented an answer. It would have: this
        # number reads +8.3 after one iteration and drifts down for twenty more.
        gain = sum(test[i](theta_v[i]) for i in range(len(df))) - base.sum()
        with open(trace_path, "a") as fh:
            fh.write(f"{seen[0]},{marginal:.6f},{gain:.4f}\n")

    pool = ThreadPool(args.jobs) if args.jobs > 1 else None
    try:
        out = joint_hierarchical(train, start, mu=mu, T=T,
                                 max_iter=args.max_iter, tol=args.tol,
                                 accelerate=not args.no_accelerate,
                                 checkpoint=save, verbose=True, pool=pool)
    finally:
        if pool is not None:
            pool.close()
    print(f"  {time.time() - t0:.0f}s", flush=True)

    iters_so_far = seen[0]
    np.savez_compressed(state_path, mu=out["mu"], T=out["T"],
                        theta=out["theta"], iters_so_far=iters_so_far,
                        marginal=out["marginal"], converged=out["converged"])

    joint_ll = np.array([test[i](out["theta"][i]) for i in range(len(df))])
    print(f"\nfull joint       {joint_ll.sum():10.1f}  "
          f"{joint_ll.sum() - base.sum():+8.1f} vs no pooling "
          f"({(joint_ll.sum() - base.sum()) / n_test.sum():+.4f} a visit)")

    sd = np.sqrt(np.diag(out["T"]))
    corr = out["T"] / np.outer(sd, sd)
    print("\nthe population, from the joint fit:")
    for j, name in enumerate(PARAMS):
        print(f"   {name:<16} mu {out['mu'][j]:+8.4f}   tau {sd[j]:.4f}   "
              f"moved {np.abs(out['theta'][:, j] - theta[:, j]).mean():.4f}")
    print("\npopulation correlations:")
    print(pd.DataFrame(np.round(corr, 2), index=PARAMS,
                       columns=[p[:9] for p in PARAMS]).to_string())

    np.savez_compressed(
        os.path.join(args.dir, "joint.npz"),
        mu=out["mu"], T=out["T"], theta=out["theta"], cov=out["cov"],
        marginal=out["marginal"], history=np.array(out["history"]),
        converged=out["converged"], n_iter=out["n_iter"],
        n_em_steps=out["n_em_steps"], test_ll=joint_ll, base_test_ll=base,
        pooled_test_ll=pooled_ll, params=np.array(PARAMS),
        players=df.player.values.astype(str), iters_total=iters_so_far)
    print(f"\nwrote {args.dir}/joint.npz ({iters_so_far} iterations in total)")
    # the grep target for the resume loop in the module docstring
    print("converged" if out["converged"] else "not converged, run again")


if __name__ == "__main__":
    main()
