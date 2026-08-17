"""
Learning a player's throw *while they play*, and recommending accordingly.

The fitting in :mod:`darts.fitting` is offline: collect a session of scores,
run EM, get a point estimate. The measurement designs in :mod:`darts.design`
improve the session, but it is still a session -- a block of darts thrown for
the sake of being measured.

Match play offers something better for free. Every dart of a real leg is
thrown at a target the policy already knows, and its score is observed. That
is one observation per dart about the spread, at no cost in practice time --
and the targets vary as the policy moves the player around the board (trebles
while scoring, then setup shots, then doubles), so some of every leg is spent
at targets that are genuinely informative about sigma.

Two distinct benefits of that variety should not be conflated. Under this
module's assumption of a known (zero) bias, the benefit is simply that varied
targets include informative ones -- notebook 09's information analysis, not
its confounding result. The confounding result (a tight group at one target
being indistinguishable from a displaced aim) matters the moment the bias is
unknown, and is then the strong reason match play beats a single-target drill;
extending this posterior over (b, sigma) is where that machinery would earn
its keep.

So the loop is Bayesian:

* **prior** -- the player names a band ("club"), which places a prior over the
  shared sigma grid;
* **update** -- after each dart, multiply by the likelihood of the observed
  score given the aim point, ``p(s | t, sigma)``, evaluated on the same pixel
  grid the rest of the project uses. One multiply over the grid, microseconds;
* **act** -- recommend the aim point maximising the *posterior-weighted*
  Q-value across MDPs solved at a grid of sigmas. Not the plug-in action at
  the posterior mean: a posterior straddling two abilities can prefer a third
  aim point that hedges between what either ability would do.

Assumptions, stated plainly: the throw is isotropic Gaussian with no
systematic bias, the same at every target, and the *score value* of every dart
is observed (players see where their darts land, including on a bust). The
bias-free assumption is the strong one; ``fit_multi_target`` relaxes it
offline, and carrying a bias in this posterior would mean a 3-D grid rather
than a 1-D one -- the machinery generalises, the demo does not need it.
"""

import numpy as np

from darts.dartboards import generate_dartboard
from darts.mdp_3turn import ThreeDartMDP
from darts.transitions import transition_arrays
from darts.utils import mm_per_pixel


def band_prior(band, sigmas, spread=1.4):
    """
    A prior over the sigma grid from a named ability band.

    Lognormal in shape: the player who says "club" is probably near 20mm,
    might be league or pub, and is almost certainly not elite. ``spread`` is
    the multiplicative width -- the default 1.4 puts the adjacent bands at
    roughly one standard deviation, so a player who names a band is trusted to
    within a band either side and no further.

    Args:
        band (str or float): a key of ``players.ABILITY_BANDS``, or a sigma in
            mm directly.
        sigmas (np.ndarray): the grid the posterior lives on.
        spread (float): multiplicative one-sigma width of the prior.

    Returns:
        np.ndarray: prior probabilities over ``sigmas``, summing to 1.
    """
    from darts import players
    centre = players.ABILITY_BANDS[band] if isinstance(band, str) else float(band)
    z = (np.log(np.asarray(sigmas, float)) - np.log(centre)) / np.log(spread)
    p = np.exp(-0.5 * z ** 2)
    return p / p.sum()


def flat_prior(sigmas):
    """Complete ignorance over the grid."""
    sigmas = np.asarray(sigmas, float)
    return np.full(len(sigmas), 1.0 / len(sigmas))


class SigmaPosterior:
    """
    A discrete posterior over sigma, updated one dart at a time.

    The likelihood of a dart is exact on the pixel grid: for the observed
    score value ``s`` and aim point ``t``,

        p(s | t, sigma) = sum over pixels scoring s of N(t, sigma^2 I)

    evaluated only over the pixels of that score, so an update costs a few
    thousand exponentials per sigma -- microseconds in practice.
    """

    def __init__(self, sigmas, prior=None, board_pixels=256, board=None,
                 checkouts=None):
        self.sigmas = np.asarray(sigmas, dtype=float)
        if board is None:
            board, checkouts = generate_dartboard(board_pixels)
        elif checkouts is None:
            raise ValueError("pass checkouts along with a prebuilt board")
        self.board = board
        self.checkouts = checkouts
        self.pixels = board.shape[0]
        self.mm_per_pixel = mm_per_pixel(self.pixels)

        offs = (np.arange(self.pixels) - self.pixels // 2) * self.mm_per_pixel
        x, y = np.meshgrid(offs, offs)          # x along columns, y along rows
        self._coords = np.stack([x.ravel(), y.ravel()], axis=1)
        flat = board.ravel().astype(np.int64)
        co_flat = checkouts.ravel().astype(bool)
        self._index = {int(s): np.flatnonzero(flat == s) for s in np.unique(flat)}
        self._co_index = {int(s): np.flatnonzero((flat == s) & co_flat)
                          for s in np.unique(flat[co_flat])}

        self.log_prior = np.log(flat_prior(self.sigmas) if prior is None
                                else np.asarray(prior, float))
        self.log_post = self.log_prior.copy()
        self.n_updates = 0

    # -- geometry helpers ---------------------------------------------------
    def point_to_mm(self, point):
        """(row, col) pixel -> (x, y) mm from the centre."""
        c = self.pixels // 2
        return np.array([(point[1] - c) * self.mm_per_pixel,
                         (point[0] - c) * self.mm_per_pixel])

    # -- the update ---------------------------------------------------------
    def _log_likelihood(self, aim_mm, score, checkout=False):
        idx = (self._co_index if checkout else self._index).get(int(score))
        if idx is None or len(idx) == 0:
            raise ValueError(f"score {score} (checkout={checkout}) not on this board")
        d2 = ((self._coords[idx] - np.asarray(aim_mm, float)) ** 2).sum(axis=1)
        # p_sigma = sum exp(-d2 / 2 sigma^2) / (2 pi sigma^2) * pixel_area
        s2 = self.sigmas ** 2
        p = np.exp(-0.5 * d2[None, :] / s2[:, None]).sum(axis=1) / (2 * np.pi * s2)
        p *= self.mm_per_pixel ** 2
        return np.log(np.maximum(p, 1e-300))

    def update(self, aim, score, checkout=False, pixel=True):
        """
        Fold in one dart.

        Args:
            aim: the recommended/attempted target -- (row, col) pixel
                coordinates if ``pixel`` (the convention of the MDP's candidate
                points), else (x, y) millimetres from the centre.
            score (int): the score value of the bed the dart landed in.
            checkout (bool): True if this dart ended the leg, in which case the
                dart is known to have landed in the double (or inner bull) bed
                of that value, which is more informative than the value alone.
        """
        aim_mm = self.point_to_mm(aim) if pixel else np.asarray(aim, float)
        self.log_post = self.log_post + self._log_likelihood(aim_mm, score, checkout)
        self.log_post -= self.log_post.max()
        self.n_updates += 1
        return self

    # -- summaries ----------------------------------------------------------
    @property
    def probs(self):
        p = np.exp(self.log_post - self.log_post.max())
        return p / p.sum()

    def mean(self):
        return float(self.probs @ self.sigmas)

    def std(self):
        p = self.probs
        m = p @ self.sigmas
        return float(np.sqrt(p @ (self.sigmas - m) ** 2))

    def interval(self, level=0.9):
        """Central credible interval by cumulative mass."""
        p = self.probs
        cdf = np.cumsum(p)
        lo = self.sigmas[np.searchsorted(cdf, (1 - level) / 2)]
        hi = self.sigmas[min(np.searchsorted(cdf, 1 - (1 - level) / 2),
                             len(self.sigmas) - 1)]
        return float(lo), float(hi)

    def map(self):
        return float(self.sigmas[int(np.argmax(self.log_post))])


class BayesRecommender:
    """
    Aim-point recommendations under sigma uncertainty.

    Solves the single-player 3-dart MDP (minimum-visits objective, the model
    the checkout charts argued for) at each sigma of a grid, all sharing one
    candidate-point set, and recommends

        argmax_a  sum_sigma  w(sigma) Q_sigma(state, a)

    for posterior weights ``w``. With all mass on one sigma this is exactly
    that sigma's optimal policy; with a spread posterior it hedges, which a
    plug-in policy at the posterior mean cannot.

    The solve grid can be coarser than the posterior grid -- value functions
    move smoothly in sigma -- and posterior mass is aggregated onto the
    nearest solved sigma.
    """

    def __init__(self, solve_sigmas, board_pixels=256, point_stride=2,
                 margin_mm=10.0, game_start=501, progress=False):
        self.solve_sigmas = np.asarray(solve_sigmas, dtype=float)
        self.game_start = game_start
        self.models = []
        self.tr = None
        for s in self.solve_sigmas:
            # margin_mm is fixed so every model shares one candidate-point
            # set; the default margin scales with sigma and would not.
            tr = transition_arrays(board_pixels, float(s),
                                   margin_mm=margin_mm, point_stride=point_stride)
            if self.tr is None:
                self.tr = tr
            elif not np.array_equal(tr["points"], self.tr["points"]):
                raise AssertionError("candidate grids differ between sigmas")
            m = ThreeDartMDP(tr["probs"], tr["checkout_probs"],
                             tr["allowed_scores"], game_start,
                             dart_cost=0.0, turn_cost=1.0).solve()
            self.models.append(m)
            if progress:
                print(f"  solved sigma {s:.1f}", flush=True)
        self.points = self.tr["points"]
        self.board = self.tr["board"]
        self.checkouts = self.tr["checkouts"]

    def weights_from(self, posterior):
        """Aggregate posterior mass onto the nearest solved sigma."""
        w = np.zeros(len(self.solve_sigmas))
        for sig, p in zip(posterior.sigmas, posterior.probs):
            w[int(np.argmin(np.abs(self.solve_sigmas - sig)))] += p
        return w

    def qbar(self, weights, score, dart, round_start=None):
        """Posterior-weighted Q-values over the shared candidate points."""
        q = None
        for w, m in zip(weights, self.models):
            if w < 1e-12:
                continue
            qi = m.q_values(score, dart, round_start)
            q = w * qi if q is None else q + w * qi
        return q

    def recommend(self, posterior, score, dart, round_start=None):
        """The Bayes-optimal aim point index for a state, given the posterior."""
        q = self.qbar(self.weights_from(posterior), score, dart, round_start)
        return int(np.argmax(q))

    def oracle_model(self, sigma):
        """The solved model nearest a true sigma, for scoring regret."""
        return self.models[int(np.argmin(np.abs(self.solve_sigmas - sigma)))]

    def value_loss(self, true_sigma, action, score, dart, round_start=None):
        """
        Visits lost, in expectation, by taking ``action`` in this state rather
        than the action optimal for the true sigma.
        """
        q = self.oracle_model(true_sigma).q_values(score, dart, round_start)
        return float(q.max() - q[action])


def play_leg(recommender, posterior, true_sigma, rng, game_start=None,
             policy="bayes", record=None):
    """
    Simulate one leg of 501, recommending each dart from the current posterior
    and updating it with each observed score.

    Args:
        recommender (BayesRecommender): the solved sigma grid.
        posterior (SigmaPosterior): updated in place, dart by dart.
        true_sigma (float): the player's actual spread in mm.
        rng: numpy Generator.
        policy: ``"bayes"`` (posterior-weighted), ``"oracle"`` (true sigma,
            the unattainable benchmark), or a float sigma (a fixed assumption,
            e.g. the centre of the claimed band, never updated).
        record (list): if given, appended with one dict per dart.

    Returns:
        int: darts thrown to finish the leg.
    """
    board, checkouts = recommender.board, recommender.checkouts
    px = board.shape[0]
    mmpp = mm_per_pixel(px)
    game_start = game_start or recommender.game_start

    score = game_start
    darts = 0
    while score > 0:
        round_start = score
        for dart in (1, 2, 3):
            if policy == "bayes":
                a = recommender.recommend(posterior, score, dart, round_start)
            elif policy == "oracle":
                m = recommender.oracle_model(true_sigma)
                a = int(np.argmax(m.q_values(score, dart, round_start)))
            else:
                m = recommender.oracle_model(float(policy))
                a = int(np.argmax(m.q_values(score, dart, round_start)))
            aim = recommender.points[a]

            land_mm = posterior.point_to_mm(aim) + rng.normal(0.0, true_sigma, 2)
            col = int(round(land_mm[0] / mmpp)) + px // 2
            row = int(round(land_mm[1] / mmpp)) + px // 2
            if 0 <= row < px and 0 <= col < px:
                v = int(board[row, col])
                on_double = bool(checkouts[row, col])
            else:
                v, on_double = 0, False
            darts += 1

            finished = (score - v == 0) and on_double
            bust = (not finished) and (score - v <= 1)
            posterior.update(aim, v, checkout=finished)
            if record is not None:
                record.append({
                    "dart": darts, "state_score": score, "state_dart": dart,
                    "aim_row": int(aim[0]), "aim_col": int(aim[1]),
                    "scored": v,
                    "post_mean": posterior.mean(), "post_std": posterior.std(),
                    "value_loss": recommender.value_loss(
                        true_sigma, a, score, dart, round_start),
                })
            if finished:
                return darts
            if bust:
                score = round_start
                break
            score -= v
    return darts
