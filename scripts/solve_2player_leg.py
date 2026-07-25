#!/usr/bin/env python
"""
Solve a full two-player 501 leg.

This is the long job: the three-dart zero-sum game has ``game_start**2`` states
and every pair whose score is below ``3 * max_dart + 2 = 182`` needs its own
within-turn sweep. It checkpoints as it goes and can be resumed, so it is safe
to leave running overnight and safe to interrupt.

Typical use:

    # measure the cost on a small game first (a minute or two)
    python scripts/solve_2player_leg.py --game-start 120 --out results/probe.npz

    # then the real thing
    python scripts/solve_2player_leg.py --out results/leg3_501.npz

    # if it was interrupted, add --resume; it picks up from the last checkpoint
    python scripts/solve_2player_leg.py --out results/leg3_501.npz --resume

Output (``.npz``):
    W          (G+1, G+1)  W[u, v] = P(player on u, about to throw, wins)
    policy     (G+1, G+1)  index into `points` of the optimal first dart
    Y3, Y2     (G+1, G+1)  within-turn values, [opponent score, my score]
    points     (n, 2)      pixel coordinates of the retained aiming points
    W_onedart  (G+1, G+1)  the same game with one dart per turn (cheap, for
                           comparison), unless --skip-one-dart
"""
import argparse
import os
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sigma", type=float, default=15.0,
                    help="throwing standard deviation in mm (default 15, a "
                         "league-standard player)")
    ap.add_argument("--board-pixels", type=int, default=512,
                    help="board resolution; 512 is converged for the strategy "
                         "questions, 256 is noticeably coarse")
    ap.add_argument("--point-stride", type=int, default=4,
                    help="stride of the initial aiming grid, in pixels")
    ap.add_argument("--game-start", type=int, default=501)
    ap.add_argument("--out", default="results/leg3_501.npz")
    ap.add_argument("--checkpoint-every", type=int, default=25,
                    help="diagonals between checkpoints")
    ap.add_argument("--resume", action="store_true",
                    help="continue from the checkpoint next to --out")
    ap.add_argument("--all-points", action="store_true",
                    help="do NOT reduce the aiming grid (much slower; the "
                         "reduction costs <1e-6 darts in the single-player game)")
    ap.add_argument("--skip-one-dart", action="store_true")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from darts.transitions import transition_arrays
    from darts.mdp_2player import OneDartLeg, ThreeDartLeg, candidate_points

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    ckpt = args.out + ".checkpoint.npz"

    print(f"board {args.board_pixels}px, sigma {args.sigma}mm, "
          f"stride {args.point_stride}, game_start {args.game_start}")
    tr = transition_arrays(args.board_pixels, args.sigma,
                           point_stride=args.point_stride)
    P, CP, S, points = (tr["probs"], tr["checkout_probs"],
                        tr["allowed_scores"], tr["points"])
    print(f"  {len(points)} aiming points, {len(S)} distinct board scores")

    if not args.all_points:
        t0 = time.perf_counter()
        keep = candidate_points(P, CP, S, game_start=args.game_start)
        P = np.ascontiguousarray(P[keep])
        CP = np.ascontiguousarray(CP[keep])
        points = points[keep]
        print(f"  reduced to {len(points)} candidate aiming points "
              f"({time.perf_counter() - t0:.0f}s)")

    out = {"points": points, "sigma": args.sigma,
           "board_pixels": args.board_pixels, "allowed_scores": S}

    if not args.skip_one_dart:
        t0 = time.perf_counter()
        one = OneDartLeg(P, CP, S, args.game_start).solve(progress=True)
        out["W_onedart"] = one.W
        out["policy_onedart"] = one.policy
        print(f"  one-dart leg done in {time.perf_counter() - t0:.0f}s, "
              f"W[{args.game_start},{args.game_start}] = "
              f"{one.W[args.game_start, args.game_start]:.4f}")

    t0 = time.perf_counter()
    three = ThreeDartLeg(P, CP, S, args.game_start).solve(
        progress=True, checkpoint_path=ckpt,
        checkpoint_every=args.checkpoint_every, resume=args.resume)
    print(f"  three-dart leg done in {time.perf_counter() - t0:.0f}s, "
          f"W[{args.game_start},{args.game_start}] = "
          f"{three.W[args.game_start, args.game_start]:.4f}")

    out.update(W=three.W, policy=three.policy, Y3=three.Y3, Y2=three.Y2)
    np.savez_compressed(args.out, **out)
    print(f"wrote {args.out}")
    if os.path.exists(ckpt):
        os.remove(ckpt)


if __name__ == "__main__":
    main()
