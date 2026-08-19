# Where to take the darts MDP work next

A ranked guide to the directions that look most promising, judged on two axes:
**feasibility** (can it be computed, and is the data available?) and
**applicability** (would it change what a real darts player does?).

The experiments themselves are indexed in
[`notebooks/experiments/README.md`](../notebooks/experiments/README.md), which
also lists what everything costs to re-run.

---

## Where things stand

| Model | Status | Cost |
|---|---|---|
| Single dart, expected score | published | FFT, milliseconds |
| Single player, memoryless MDP (`darts/mdp.py`) | published | minutes |
| Single player, 3-dart visits (`darts/mdp_3turn.py`) | solved exactly, values + policy | ~2.6 s for 501 at 8.1k aiming points, ~11 s at 32k |
| Two players, 1 dart per turn (`darts/mdp_2player.py`) | solved exactly | one GEMM per diagonal |
| Two players, 3 darts per turn (`darts/mdp_2player.py`) | solved exactly, values + policy | ~15 min for a full 501 leg on a reduced aiming grid (`scripts/solve_2player_leg.py`) |
| Sets and legs (`darts/match.py`) | solved exactly | a small Markov chain, instant |
| Fitting a player from scores (`darts/fitting.py`) | exact EM, SQUAREM accelerated | ~1.4 s for a 200-dart session |
| Anisotropic throw with bias (`darts/throw_shape.py`) | solved; grid *and* particle posterior | one MDP solve per covariance; bias is free |
| Measurement design (`darts/design.py`) | optimum found and certified, c- / D- / L-criteria | ~7 s for the information at every target |

Everything runs on a **512-pixel board** with a **3.52 mm aiming grid**. That is
not a free parameter: an 8 mm bed is 9.1 pixels across at 512 and 4.5 at 256, and
the coarse board misjudges bed-defined targets badly (notebook 02, and the
appendix of notebook 09).

Two structural facts do most of the work in the current solvers and are worth
keeping in mind for anything new:

1. **A dart thrown at a score of 62 or more cannot bust.** So the value of the
   last dart of a visit is independent of what the visit started on, and that
   propagates: the second dart is independent above 122, the whole visit above
   182. Only ~180 low scores need per-visit-start states. This is what turned an
   `O(game_start²)` sweep into a near-linear one.
2. **The only circular dependency inside a visit is the scalar value of the
   score the visit started from** (busts return to it). Solving that one scalar
   fixed point with Aitken acceleration replaces sweeping the whole state space
   to convergence.

Both facts survive into the two-player game, where the analogous ordering is by
the *total* of the two scores.

A third has since joined them, and it is what made the richer throw model
affordable:

3. **A bias does not change the board.** Aiming at `a` with bias `b` lands where
   aiming at `a + b` would land unbiased, so the biased game is the unbiased game
   with its action set translated — same values, same shots, relabelled. No MDP
   is ever re-solved for a different bias. Anisotropy has no such shortcut: every
   covariance needs its own solve, and a *tilt* adds another axis to that grid.

---

## Done since this was written

The original list put "fit a per-player covariance" first. That is now done, and
so is most of what followed from it.

* **§1.1 per-player covariance** — notebooks 12, 14, 16. A tall throw is worth
  **0.28 visits per leg** over a round one of equal ellipse area, and modelling
  it costs one MDP solve per covariance.
* **§3.1 sets and legs** — notebook 06. The bull-up is worth 7.4 points in a best
  of 3 and 1.0 in a best of 13 sets.
* **§3.2 two-player, three darts** — notebook 04. Ignoring the opponent costs at
  most ~1.4 points of win probability, and only below 120.
* **§4.1 ability-tailored checkout charts** — notebook 05.
* **§4.2 practice-value calculator** — notebook 08. A millimetre is worth
  **0.43–0.58 visits per leg**, flat across ability; trebles are the better
  practice for strong players and doubles for weak ones.
* **§4.3 measurement protocol** — notebooks 09, 10, 17, and the design scripts.
* **The measurement/advice loop** — notebook 11, and its extensions to bias
  (13), the joint model (14), real-time inference (15) and a tilt (16).

Three results from that work are worth carrying forward, because they change how
the remaining items should be approached.

**Bias dominates.** Three millimetres of sideways pull costs 0.144 visits per
leg; a millimetre and a half of spread costs 0.005. Anything that improves
knowledge of where a player's darts actually centre is worth roughly thirty times
the equivalent work on the shape of the group.

**Match play is a better measuring instrument than a drill**, not merely a
cheaper one. A single-target session cannot separate the spread from the aim
point; the varied targets a leg visits naturally break that ridge (notebook 13).
The exception is the *lean*, which is read from angular structure at the centre
of the board, and match play essentially never goes there (notebook 17).

**Do not measure a policy difference by counting argmax changes.** It overstates
by about fivefold, in every model where it has been checked.

---

## 1. Modelling fidelity

### 1.1 Accuracy that depends on where you aim — **the biggest remaining gap**

The model assumes one `Σ` everywhere on the board. Real players are far more
accurate at the treble 20 than at the treble 11, because that is what they have
practised, and many are measurably worse throwing at a double under pressure.

Both are one-line changes to the transition builder — `Σ` becomes a function of
the aiming point — and neither costs the solver anything, because the solver only
ever sees the resulting `(n_points, n_scores)` matrix.

The strategic consequences are large and testable:

* A player who is 20% worse at doubles than at trebles should leave themselves on
  a *different* number than the model currently says.
* A player with a strong 19 and a weak 20 may have a completely different scoring
  route, and the MDP will find it automatically.

The measurement side is now much better equipped for this than it was: notebook
09's design machinery would say where to throw to *estimate* a per-region `Σ`,
and the same c-optimality argument applies per region.

* **Feasibility:** high to build, medium to calibrate — you need enough data per
  region, which means practice-session logging rather than a single sitting.
* **Applicability:** very high. Every player already believes this about
  themselves; the model would tell them what to do about it.

### 1.2 Darts within a visit are not independent

The model treats the three darts of a visit as i.i.d. draws around whatever the
player aims at. In reality, a player who sees dart 1 land 15 mm high adjusts.
That correlation is exactly what the 3-dart state space is built to represent,
and adding it needs no new solver: extend the within-visit state with a coarse
summary of the previous dart's error (say a 3×3 grid of "where the last dart
went"), and let the transition matrix depend on it.

Cost: multiplies the within-visit state count by the number of error bins, which
is affordable — the within-visit states are a small fraction of the work.

* **Feasibility:** medium. The solver change is contained; the data to estimate
  "how much do players correct" is the hard part.
* **Applicability:** high, and the most *scientifically* interesting of these,
  because nobody has quantified the value of in-visit correction.

### 1.3 A drifting player

Everything assumes the player is constant while the posterior sharpens forever.
A pull is exactly what appears when someone tires. `ParticleThrowPosterior`
already takes a per-parameter drift and tracks a moving pull well (notebook 15) —
but the drift magnitude is a hook, not a calibration. What a real player's pull
does over an evening is unmeasured, and only a real session can supply it.

* **Feasibility:** high to run, blocked on data. **Applicability:** high.

### 1.4 Bounce-outs, wired darts, and the dart already in the board

Small effects, easy to add as a fixed probability of scoring zero (bounce-out) or
as a reduced effective area for a bed that already holds two darts. The second is
a genuine reason not to aim at the same treble three times, and the model
currently cannot express it.

* **Feasibility:** high. **Applicability:** low-to-medium; worth a footnote
  rather than a project.

---

## 2. Making the computation bigger

### 2.1 The likelihood of a missed dart — **a three-hour bug in a fourteen-hour rebuild**

The per-dart likelihood sums Gaussian density over every pixel carrying the
observed score. A scoring bed is ~5,000 pixels at 512; a **miss** is the entire
non-scoring board, ~145,000. So a scoring dart costs 27 ms and a missed one
**3.4 seconds**, and 11.6% of simulated match darts are misses. This alone is
roughly 2.8 hours of notebook 17's 3.4, and it is what makes live play cost half
a second a dart instead of 22 ms.

Almost all of those pixels are many standard deviations from the dart and
contribute nothing, so the sum should be truncated. The obvious version does not
work: the pixel set is shared across the whole particle cloud, which spans `σ`
from 3.5 mm to 46 mm, so a single cutoff has to be sized for the widest particle
and at eight standard deviations that is wider than the board. (This was tried
and reverted — exact to 3e-15, and slower.) The fix is to give each band of
particles its own pixel set, or to compute the zero-score probability as the
complement of the scoring ones.

* **Feasibility:** high, and self-contained. **Applicability:** indirect but it
  pays for itself immediately in every experiment that simulates play.

### 2.2 Reducing the set of aiming points

Every Q-value in every one of these models is a linear functional of a point's
score distribution, so only points on the convex hull of those distributions can
ever be optimal. A quick experiment (7,573 points, 512-pixel board, σ = 15 mm)
found ~1,950 points that are the argmax for some isotropic random direction, so
the hull is not tiny — but isotropic directions are far broader than the value
vectors that actually arise, which are monotone in score. Taking the union of the
points that are optimal *somewhere* in a family of single-player problems gives a
much smaller set: `mdp_2player.candidate_points` cuts a 1,893-point grid to 195
(10%), and re-solving the single-player 501 problem on the reduced grid changes
the answer by less than 1e-6 darts.

Worth pushing further, because a 10× reduction in the action set is a 10×
reduction in every solve — and the covariance grids of notebooks 12–17 multiply
that saving. Two routes:

* **Rigorous:** an LP per point testing whether its distribution lies inside the
  convex hull of the retained set. ~7,500 small LPs; certifies that discarding it
  cannot lose anything for *any* value function.
* **Practical:** the policy-union heuristic above, then verify by recomputing the
  Bellman maximum over the full set at a sample of states and reporting the gap.

* **Feasibility:** high. **Applicability:** indirect but large.

### 2.3 The low-score region of the two-player game

In the two-player game, every state whose score is below 182 needs its own
within-visit sweep, and there are `O(180 × game_start)` of them — this is what
dominates the cost. There is a way out that mirrors the single-player trick: for
a fixed opponent score, every within-visit value is a **piecewise-linear convex
function of one scalar** (the value of the visit ending back where it started).
Storing the upper envelope of those lines — usually a handful of segments —
instead of recomputing the maximum over thousands of aiming points would collapse
the low-score work by orders of magnitude.

* **Feasibility:** medium; a self-contained piece of geometry (2-D convex hulls
  of (slope, intercept) pairs). **Applicability:** indirect.

### 2.4 Continuous aiming points

The aiming grid is a discretisation, and notebook 17 found the cost of that
directly: the value loss from a parameter error is a **staircase**, flat below
half a grid step, because an error too small to move any recommendation costs
exactly nothing. There is a precision beyond which measuring a player better buys
nothing at all, and it is set by the aiming grid rather than by the data.

Since the score-probability functions are smooth in the aim location, the optimum
could be refined by local continuous optimisation from the grid argmax. Nobody
has asked what the grid *should* be, which is the more interesting half of this.

* **Feasibility:** high. **Applicability:** low for players, medium for the
  measurement work.

---

## 3. Two-player and match play

### 3.1 Asymmetric abilities

`W[u, v]` for two *different* players is the same computation with two transition
matrices (`darts/mdp_2player_asym.py`). It answers "how should I play differently
against a better player?" — the classic intuition is that an underdog should take
more risk, and this model can say exactly where and how much.

* **Feasibility:** high (it is a parameter change). **Applicability:** high.

### 3.2 The opponent, under the richer throw model

Notebook 04 measured the value of knowing the opponent's score under an isotropic
throw. Both the two-player game and the anisotropic model are solved; nobody has
combined them. The endgame effects should be larger for a player whose group is
stretched, since which double they want depends on the shape.

* **Feasibility:** high, but the solve grids multiply. **Applicability:** medium.

---

## 4. Outputs a real player could use

### 4.1 Calibration against real match data — **now the most valuable thing on the list**

The model maps `σ` to a 3-dart average. That mapping should be checked the other
way round: take published professional statistics (3-dart averages, checkout
percentages by score, first-nine averages) and ask which `σ` reproduces them, and
whether *one* `σ` can reproduce all of them at once. If it cannot — for instance
if pros' real checkout percentages are worse than the model predicts at their
scoring `σ` — that is direct evidence for the aim-dependent accuracy of §1.1, and
a genuinely interesting result.

* **Feasibility:** medium; the data is public but needs scraping and cleaning.
* **Applicability:** high, and it is the thing that would make the whole project
  credible to a darts audience rather than a statistics one.

### 4.2 A real player, measured

Everything in this repository is simulated. The measurement protocol is designed,
certified, priced and tested against synthetic players; it has never been pointed
at a human. The protocol is short — a couple of hundred darts at the bull —
and the machinery to fit, advise and track in real time already exists.

The interesting question is not whether it works but where the Gaussian model
*fails*: a real thrower may be skewed, heavy-tailed, or genuinely different at
different targets, and only a real session will say.

* **Feasibility:** high; it needs a player and an evening. **Applicability:** the
  highest of anything here, because every other result is conditional on it.

### 4.3 The prior on a lean is a guess

Notebook 16 found the lean easier to measure than the spread, and worth 0.61
visits per leg across orientations. But its prior is centred on no lean with a
standard deviation of 0.3, chosen by hand, and at a few hundred darts that prior
is doing real work. What real players' leans actually look like is unmeasured.

* **Feasibility:** falls out of §4.2. **Applicability:** medium.

---

## 5. Board design

The Quadro board work already in the repo extends naturally: with a full-game
solver rather than a single-dart one, you can ask whether the Quadro board makes
*legs* shorter, whether it widens or narrows the gap between strong and weak
players, and whether it makes the endgame more or less interesting.

Notebook 16 sharpened the interesting version of this question. The board's
*geometry* is very nearly rotationally symmetric; its **numbering** is not, and
that asymmetry is worth 0.61 visits per leg to a leaning player — the 20 is
flanked by the 1 and the 5 precisely so that sideways error is punished. So "what
would a better dartboard numbering look like?" now has a measurable objective:
which arrangement maximises or minimises the skill gap, and which is most
forgiving of the way real throws are actually shaped.

* **Feasibility:** high (a board is just a different pixel array).
* **Applicability:** low for players, high for a good post.

---

## Suggested order

1. **Measure a real player** (§4.2) — everything else is conditional on it, and
   it is an evening's work plus a willing thrower.
2. **Fix the missed-dart likelihood** (§2.1) — three hours of every rebuild, and
   the difference between 22 ms and half a second a dart in live play.
3. **Calibration against professional statistics** (§4.1) — the credibility item,
   and it doubles as evidence for or against §1.1.
4. **Aim-dependent accuracy** (§1.1) — the biggest remaining modelling gap.
5. **Action-set reduction** (§2.2), which makes the covariance grids of §3.2 and
   everything downstream affordable.
6. **Asymmetric abilities** (§3.1) — cheap, and answers a question players ask.
7. In-visit correlation (§1.2), board numbering (§5), everything else.
