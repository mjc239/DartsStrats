#!/usr/bin/env python
"""
The weight matrix that turns "measure me best" into "make me score best".

A design that minimises estimation error is not the same as one that minimises
*visits lost*, because the parameters do not matter equally to the policy. Get
the bias slightly wrong and every dart is displaced; get the correlation
slightly wrong and the recommendation barely moves. A design should know that.

The link is a second-order expansion. Let

    L(theta') = expected visits lost per leg by playing the policy for theta'
                in a world that is really theta

which is zero at ``theta' = theta`` and non-negative everywhere, hence locally
quadratic:

    L(theta + delta) ~ 0.5 * delta^T H delta

With an estimator of variance ``M(w)^-1 / n``, the expected loss is then
``0.5 * tr(H M(w)^-1) / n``, so the design should minimise ``tr(H M^-1)`` --
which is the L-criterion of :func:`darts.design.l_criterion` with ``W = H``.
c-optimality for sigma alone is the special case ``H = c c^T``.

``H`` is estimated here from ``L``, which needs one MDP solve per perturbed
covariance (a bias perturbation is free -- it only translates the action set).
Common random numbers are used across evaluations so that the differences are
far less noisy than the individual losses.

One wrinkle worth knowing about. ``L`` is not actually smooth: the recommender
chooses from a finite grid of aiming points, so a small enough parameter error
changes no recommendation at all and costs *exactly* nothing. Being wrong about
the bias by less than half a grid step (1.8 mm here) is free, and the loss then
climbs in steps rather than as a parabola. A single central difference picks up
that staircase and comes out asymmetric -- 0.105 against 0.211 for a 3 mm error
either way, which says more about where the grid points fall than about the
player. The diagonal is therefore fitted by least squares through several
perturbation sizes, forcing through the origin, which averages the steps out.

Writes ``results/design/decision_weight.npz``.

Usage:
    python scripts/decision_weight.py --legs 200
"""
import argparse
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from darts.throw_shape import (ShapeRecommender, ThrowPosterior, correlation,
                               play_leg)

PIXELS = 256
PARAMS = ("b_x", "b_y", "S_xx", "S_xy", "S_yy")


def sigma_from(theta):
    """(S_xx, S_xy, S_yy) -> a covariance matrix."""
    return np.array([[theta[2], theta[3]], [theta[3], theta[4]]])


def theta_from(Sigma, bias):
    return np.array([bias[0], bias[1], Sigma[0, 0], Sigma[0, 1], Sigma[1, 1]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma-x", type=float, default=14.0)
    ap.add_argument("--sigma-y", type=float, default=18.0)
    ap.add_argument("--rho", type=float, default=0.2)
    ap.add_argument("--legs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--multipliers", nargs="*", type=float,
                    default=[0.5, 1.0, 1.5, 2.0],
                    help="perturbation sizes, as multiples of the base step")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sx, sy, rho = args.sigma_x, args.sigma_y, args.rho
    S_true = np.array([[sx ** 2, rho * sx * sy], [rho * sx * sy, sy ** 2]])
    b_true = np.zeros(2)
    theta0 = theta_from(S_true, b_true)

    # Step sizes, each chosen to be a *meaningful* change in its own units:
    # 3 mm of pull, 1.5 mm on each spread, 0.15 of correlation.
    step = np.array([3.0, 3.0, 2 * sx * 1.5, 0.15 * sx * sy, 2 * sy * 1.5])
    print("true theta:", np.round(theta0, 2))
    print("steps     :", np.round(step, 2))

    # every covariance the finite differences will need
    needed, keys = [], {}

    def want(dS):
        """Register a covariance perturbation and return its key."""
        key = tuple(np.round(dS, 6))
        if key not in keys:
            keys[key] = len(needed)
            needed.append(sigma_from(theta0 + np.concatenate([[0, 0], dS])))
        return keys[key]

    want(np.zeros(3))
    for i in range(3):
        for mult in args.multipliers:
            for s in (+1, -1):
                d = np.zeros(3)
                d[i] = s * mult * step[2 + i]
                want(d)
    for i, j in itertools.combinations(range(3), 2):
        for si, sj in itertools.product((+1, -1), repeat=2):
            d = np.zeros(3)
            d[i], d[j] = si * step[2 + i], sj * step[2 + j]
            want(d)

    print(f"\n{len(needed)} covariances to solve")
    t0 = time.perf_counter()
    rec = ShapeRecommender(np.stack(needed), board_pixels=PIXELS, point_stride=2)
    print(f"  solved in {time.perf_counter()-t0:.0f}s", flush=True)

    post = ThrowPosterior(rec.Sigmas, board=rec.board, checkouts=rec.checkouts)
    cache = {}

    def loss(delta):
        """Expected visits lost per leg by believing theta0 + delta."""
        key = tuple(np.round(delta, 6))
        if key in cache:
            return cache[key]
        dS = np.asarray(delta[2:], float)
        k = keys.get(tuple(np.round(dS, 6)))
        if k is None:
            raise KeyError("unregistered covariance perturbation")
        belief = (needed[k], b_true + np.asarray(delta[:2], float))
        rng = np.random.default_rng(args.seed)          # common random numbers
        tot = 0.0
        for _ in range(args.legs):
            log = []
            play_leg(rec, post, S_true, b_true, rng, policy=belief, record=log)
            tot += pd.DataFrame(log).value_loss.sum()
        cache[key] = tot / args.legs
        return cache[key]

    def d(i, s):
        v = np.zeros(5)
        v[i] = s * step[i]
        return v

    t0 = time.perf_counter()
    H = np.zeros((5, 5))
    profile = {}
    # Diagonal by least squares through the origin over several magnitudes:
    # L = 0.5 H x^2, so H = 2 * sum(x^2 L) / sum(x^4). This averages over the
    # aiming grid's staircase, which a single central difference cannot.
    for i in range(5):
        xs, ys = [], []
        for mult in args.multipliers:
            for s_ in (+1, -1):
                v = np.zeros(5)
                v[i] = s_ * mult * step[i]
                if tuple(np.round(v[2:], 6)) not in keys:
                    continue
                xs.append(s_ * mult * step[i])
                ys.append(loss(v))
        xs, ys = np.array(xs), np.array(ys)
        H[i, i] = 2 * (xs ** 2 * ys).sum() / (xs ** 4).sum()
        profile[PARAMS[i]] = (xs, ys)
        shown = "  ".join(f"{x:+.1f}:{y:.3f}" for x, y in zip(xs, ys))
        print(f"  {PARAMS[i]:>5}: {shown}   H={H[i,i]:.3e}", flush=True)

    for i, j in itertools.combinations(range(5), 2):
        pp = loss(d(i, +1) + d(j, +1))
        mm = loss(d(i, -1) + d(j, -1))
        pm = loss(d(i, +1) + d(j, -1))
        mp = loss(d(i, -1) + d(j, +1))
        H[i, j] = H[j, i] = (pp + mm - pm - mp) / (4 * step[i] * step[j])
    print(f"  finite differences in {time.perf_counter()-t0:.0f}s", flush=True)

    # L >= 0 with a minimum at theta0, so H must be positive semi-definite;
    # Monte Carlo noise can violate that, so project onto the PSD cone.
    w, V = np.linalg.eigh(0.5 * (H + H.T))
    n_neg = int((w < 0).sum())
    H_psd = V @ np.diag(np.maximum(w, 0.0)) @ V.T

    os.makedirs(os.path.join(root, "results", "design"), exist_ok=True)
    path = os.path.join(root, "results", "design", "decision_weight.npz")
    np.savez(path, H=H, H_psd=H_psd, theta0=theta0, step=step,
             Sigma_true=S_true, bias_true=b_true, eigenvalues=w,
             legs=args.legs, params=np.array(PARAMS),
             **{f"profile_x_{k}": v[0] for k, v in profile.items()},
             **{f"profile_y_{k}": v[1] for k, v in profile.items()})
    print(f"\neigenvalues: {np.array2string(w, precision=3)}")
    print(f"({n_neg} negative, clipped to zero)")
    print(f"wrote {path}")

    scale = np.sqrt(np.diag(H_psd))
    corr = H_psd / np.outer(np.where(scale > 0, scale, 1), np.where(scale > 0, scale, 1))
    print("\nrelative importance (sqrt of the diagonal, in per-unit terms):")
    for p, s in zip(PARAMS, scale):
        print(f"  {p:>5}: {s:.3e}")
    print("\nhow much each parameter matters over its own step:")
    for i, p in enumerate(PARAMS):
        print(f"  {p:>5}: {0.5 * H_psd[i, i] * step[i] ** 2:.4f} visits per leg")


if __name__ == "__main__":
    main()
