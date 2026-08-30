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
| Calibration against scores (`darts/calibration.py`) | machinery built and validated; **run against 300,985 real darts** (notebook 19) | exact visit likelihood, milliseconds |
| What couples a visit's darts (`darts/dependence.py`) | aim rule + throw coupling, fitted per player (notebook 20) | quadrature over the visit latent; seconds a player |
| The shape of one dart (`darts/throw_families.py`) | eight families compared held-out; **a dart is Student-t, `ν ≈ 2.25`** (notebook 21), and survives an elliptical core and a bounded-moment rival (notebook 24) | whole-board integral, ~4 ms an evaluation |
| Real match data (`darts/real_data.py`) | one loader, one cleaning rule, contamination report | seconds |
| Pooling across players (`darts/hierarchical.py`) | observed information with a step check, REML, shrinkage, and the multivariate version (notebook 26) | a 7x7 Hessian is ~100 likelihood evaluations, seconds a player |
| Throwing the dart the data says (`darts/transitions.py`) | Student-t kernel, matched on the three-dart average, solved at every band (notebook 22) | zero-padded FFT, ~4x the Gaussian; 15 min for three `nu` at seven bands |
| Fitting and measuring one (`darts/fitting.py`, `darts/design.py`) | scale-mixture EM with `nu` profiled; the t score function for the Fisher information (notebook 23) | one extra weight on the same pixel sum; seconds a fit |

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

A fourth has since been added by the arrival of real data, and it outranks the
other three.

**A visit is not three independent darts — but the reason is the aim, not the
throw.** Notebook 19 measured +18 to +22 points of lift on the next dart's
treble-20 chance. Notebook 20 found about half of that is players *moving target*
after a miss, stepping down 20 → 19 → 18, which the solver has no state for.
Notebook 21 found most of what was left is a per-dart heavy tail rather than any
coupling at all.

The pattern across all three is worth carrying forward on its own: **what looked
like the darts of a visit influencing each other keeps turning out to be one dart
being described badly.** Each time the per-dart model improved, the apparent
dependence shrank — the per-visit coupling is now worth +0.002 a visit on top of
a Student-t, against +0.51 on top of a Gaussian.

**And check the data before believing a tail.** Notebooks 19 and 20 both filtered
the scoring phase the same way and both inherited the same defect: the 2017 feed
leaks checkout darts across leg boundaries, into the one visit that filter
selects. It manufactured the entire far tail. `darts/real_data.py` now holds one
definition of the cleaning, and `tests/test_throw_families.py` asserts the defect
so a rebuild cannot quietly restore it.

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

### 1.2 The aim depends on the last dart, and the darts are not independent — **measured, and the top of this list**

This was speculation when it was written. Notebook 19 measured it on 300,985
professional darts and it is not a small effect: hitting the treble 20 raises the
next dart's chance of hitting it from 22.3% to 40.3%, **+18 to +22 points, z above
35**. It is not player pooling (34 of 35 individuals show it), not form drift (the
lift is *negative*, −6.8, across the gap between visits), and not the selection
filter (removing it makes the effect *larger*). Real visits are over-dispersed:
2.31× too many three-treble visits and a 31% shortfall of one-treble visits
against any i.i.d. model at the same marginal rate.

Every transition matrix in this project assumes the opposite. Notebook 19's
verdict section works through which published results survive: per-dart
quantities do, anything reading the spread of a visit total does not, and every
confidence interval in the fitting notebooks is too narrow.

**Notebook 20 then took that measurement apart, and the obvious fix was the
wrong one.** Fitting a family of models to bed sequences — 19 players, five
nested models each, trained on half the legs and scored on the other half —
splits the effect into three, and only the last is about the throw.

**The aim moves, and that is most of it.** Professionals use four scoring
targets and work down them after a miss. From the 20: after hitting the treble
they stay 95.1% of the time; after missing, 24.8% move to the 19. From the 19,
a miss moves to the 18 35.7% of the time. Dart 1 is at the 20 96.6% of the time,
dart 3 only 66.7%. So "missed with dart 1" is substantially "was not aiming at
the treble 20 with dart 2", and on a target-invariant statistic the coupling
falls from +22.3 to +13.1. This replicates across both data feeds, five years
apart, and it costs the players nothing measurable — **−0.45 ± 0.47 points** —
because the 19 is flanked by the 3 and the 7 where the 20 is flanked by the 1
and the 5.

This is not a missing parameter but a missing **state variable**. The solver's
aim point is a function of the score and the dart index; there is nowhere to
put "where did my last dart land". The current model gives essentially zero
probability to a quarter of the darts thrown after the first of a visit.

**Two ways it could have been wrong, both tested (notebook 24).** The elliptical
rival in notebook 21 was priced with a *Gaussian* core, and a Gaussian core reads
a round heavy-tailed group as 1.15:1 and a real 1.60:1 group as 2.27:1 — so an
**elliptical Student-t** was fitted, which recovers 1.008 and 1.580 on the same
simulations. The null holds: +0.0023 a visit, 11 wins from 17, `p = 0.33`. And
because five of the seventeen players sit on the `ν = 2` clip that keeps a t's
variance finite, a **normal-inverse-Gaussian** was fitted — a Gaussian scale
mixture like the t, but with exponential rather than polynomial tails, so every
moment exists and its scale is exactly the per-axis SD. It beats exp-power 16–17
and the Gaussian by 3,730 log units, and loses to the t by 133 — *worst of all on
the five players it was built for*. So `ν → 2` is not a parameterisation artefact
to engineer around: real darts want a tail that is **polynomial**, heavy enough
that the variance may genuinely not exist.

**A player's lean, measured at last (notebook 25).** Notebook 16 gave the throw a
tilt, found it worth 0.61 visits a leg between the best and worst orientation of
an identical ellipse, and ended by saying its prior on `ρ` was a guess and nobody
knew what real players looked like. Seventeen professionals now do. There is **no
population lean** — `μ(e₂) = −0.011 ± 0.039` — so the prior was centred right, but
its width of 0.3 should be about **0.14** once the noise in the estimates is
subtracted, which roughly doubles how hard a short session gets shrunk.
Individually a lean is barely measurable: split-half reliability **0.45** on
disjoint halves of a player's own legs, 0.62 for a whole record, and a 95%
interval about ±0.25 wide. Notebook 16's "93 darts" was for the *bull*, where
every segment boundary meets; competition darts are thrown at the treble 20, and
the factor is roughly twenty. Players differ in elongation (`I²` 0.81) but not
detectably in lean (`I²` 0.36, p = 0.079) — which refines notebook 24, whose
"the ratios scatter, so it is noise" conflated no population *mean* with no
*variation*. Only the first is true.

**Fitting players independently is the right choice, and now it is checked
(notebook 26).** On two independent splits no variant of partial pooling beats no
pooling, and complete pooling loses by 365 log units — 0.067 a visit, a tenth of
the entire Gaussian-to-Student-t gain. Players differ too much (`I²` 0.53–0.95 on
6 of 7 parameters) and are measured too well for shrinkage to add anything. Two
things fell out that generalise beyond this project: coordinate-wise shrinkage
failed *worst on the players with the least data*, backwards from everything
shrinkage is for, because the parameters correlate strongly across players (core
scale against elongation, −0.745) and pulling each toward its own mean drags the
vector off the ridge; and the one positive variant was selected by reading the
held-out score across five candidates and did not replicate, which is a reminder
that held-out likelihood is no defence against overfitting if it is used to pick
the model.

**A dart is Student-t, not Gaussian.** Notebook 21 fitted six candidate
distributions per player on held-out legs. The Gaussian loses to five of them for
all 17 players, and the winner is a **Student-t with `ν ≈ 2.25`** at +0.62
log-likelihood units a visit — beating a two-component mixture with one parameter
where the mixture spends two. The rival explanation, that the group is an ellipse
rather than the tail heavy, was priced identically and gains **nothing** (−0.02,
worse than the Gaussian it contains).

Two corrections come with it, both to notebook 20.

*The tail it measured was mostly a data defect.* The 2017 feed leaks the previous
leg's checkout darts into the next leg's opening visit — and that opening visit is
78% of the pure-scoring sample. About half of notebook 20's non-Gaussianity was
that; see `darts/real_data.py`. The Gaussian still loses on clean data, by about
half as much.

*And the shared scale was a per-dart tail in disguise.* Adding a per-visit scale
to a **Student-t** is worth +0.002 a visit; adding the same coupling to a Gaussian
is worth +0.51. What looked like the darts of a visit influencing each other was
one dart being described badly.

**Held-out likelihood is the wrong yardstick for whether it matters.** On beds
the scale is worth a rounding error. On the visit total — which is what the
transition matrix is built from — independent darts give a standard deviation of
35.9 against an observed 40.5 and a maximum **41% too rarely**; with the scale
the spread is 41.5 and the 180 rate lands within 7%. A bed sequence and a leg of
darts do not reward the same model.

**The shared scale is still not the true mechanism.** It reproduces the
treble-20 lift by overshooting the magnitude-coupling signature by five standard
errors (0.067 against an observed 0.021 ± 0.009), where the offset model — which
loses everywhere else — gets that one nearly right. Neither reaches the observed
lift of 25.1. The likeliest culprit is the aim rule remaining too crude: it
reduces the previous dart to hit-or-miss, when a player surely responds to *how*
they missed. That is the next experiment and it needs no new data.

**A by-product worth keeping.** With the bias properly fitted, professionals show
a systematic sideways pull: a median of 1.3 mm and up to 5 mm, with 14 of 19
pulling toward the 5. The per-player estimates track the raw asymmetry between
the beds either side of the 20 at r = 0.87. Notebook 13 made a pull the most
expensive thing to get wrong and notebook 18 found the sideways component the
only one a scoresheet measures well; this is that measurement, on real players.

* **Feasibility:** high. `darts/dependence.py` holds the model family and
  `results/dependence/` the fits; the remaining work is a richer aim rule.
* **Applicability:** very high, and it is a *correctness* item. The solver has no
  state for the previous dart at all, so it cannot represent the largest of the
  three effects even in principle.

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

### 1.5 Everything was built from the wrong distribution — **done, with one number that did not survive**

Notebook 21 fitted six candidate landing distributions to seventeen professionals
on held-out legs. A dart is a **Student-t with `ν ≈ 2.25`** at a core scale near
6 mm, not a Gaussian, and every transition matrix in this repository was built
from a Gaussian.

**The solver half is done** (notebook 22). `darts/transitions.py` gained
`student_t_kernel` and a `nu` argument that runs the whole way through
`transition_arrays`; `nu=None` is the Gaussian and reproduces every existing
result unchanged, and `nu=inf` reproduces the full 512-pixel MDP solve to
`1e-14`, which is the check that the new path is the distribution and not the
plumbing. Two things had to change with it, both silent when wrong: the kernel is
normalised over the plane rather than over the window (a t at `ν = 2.25` leaves
one dart in a thousand off a 512-pixel board, and the old code would have
renormalised it back on), and the correlation is zero-padded so a dart that
leaves the array is booked as a miss instead of wrapping round to the far side of
the board.

What it moves, matched on the three-dart average via `matched_scale`:

* **The scoring phase does not move.** The treble 20 / treble 19 crossover goes
  from 16.80 mm to between 16.52 and 16.91, not even monotonically in `ν`, and
  five of seven bands aim at the same pixel. The project's most-quoted answer does
  not depend on the assumption that failed.
* **The checkout phase does, and the sign depends on the player.** The whole
  difference in leg length sits below 170. Elite and pro finish 0.11 darts sooner
  as a t, the middle bands 0.12–0.23 later, a pub player 1.61 sooner — because a
  matched t is up to 15% better at an 8 mm bed and 2–12% worse at a whole sector,
  and legs at different standards are made of those in different proportions.
* **`σ` is not what the project has been calling it.** With `ν` near 2 a throw's
  variance barely exists. The familiar "elite ≈ 6.5 mm" matches the Student-t's
  *core scale* (median 5.98 mm), not a standard deviation; a Gaussian fitted to
  the same players returns 11.47 mm, splitting the difference between a core and
  a tail and describing neither. Matching on the three-dart average — rather than
  on variance, which hands a "pro" a 2.67 mm core and a 147 average — lands
  independently on the cores notebook 21 fitted.

**The fitting half is done too** (notebook 23), and it turned out to be the same
change twice. A Student-t is a Gaussian whose width is redrawn every dart, so
that width is a second latent variable alongside the landing point, and its
conditional expectation is one weight:

    u(z) = E[W | Z = z] = (ν + 2) / (ν + q(z))

* `darts/fitting.py`'s E step applies `u` pixel by pixel inside the sum it
  already did; the M step is the *same* weighted Gaussian one. `ν` is not
  estimated inside the EM but **profiled** by `profile_nu`, because notebook 21
  found it correlated with the core scale at +0.62 and a point estimate would hide
  the ridge. From 750 darts of an 8.0mm-core t, the t fit returns 8.09mm and a
  Gaussian returns 12.45mm — the core-versus-tail compromise, reproduced on
  demand. It does **not** improve the bias, which was being estimated badly for an
  unrelated reason.
* **`darts/design.py` built its score function from a Gaussian likelihood**, not
  merely for a Gaussian throw. The t's score function is the Gaussian's multiplied
  by the same `u` — so the same weight, arrived at independently. Two modules that
  share no code agree on the derivative to 2e-12.

What that cost, and it is the part worth carrying forward:

* **The design *rankings* survive.** Bull for tight players, the big single at
  ~136mm for loose ones, the treble ring for the very loose — the structure of
  notebook 09's answer is intact. The threshold moves: the bull stays optimal at
  σ = 16mm under a t where the Gaussian had already switched.
* **The dart counts do not.** A heavy tail carries **less** information about the
  core, in every band — which was the open question, and it could have gone the
  other way. Matched through `matched_scale` so the improvement being detected is
  the same one, notebook 10's **"233 darts to prove a millimetre" becomes 506**,
  and a pub player's 4,857 becomes **23,634**. Notebook 17's criteria are on the
  same footing and have not been re-run.
* Every one of those numbers is conditional on knowing `ν`, and notebook 23 shows
  a few hundred darts of *scores* pin down that a tail exists (+57 log-units) far
  better than how heavy it is (the profile peaks at 3 when the truth is 2.25). A
  session that must learn `ν` too is longer still.
* `darts/bayes.py` and `darts/throw_shape.py` are Gaussian throughout and have not
  been touched, so the live belief updates and the anisotropy work still assume a
  Gaussian throw.

* **Feasibility:** done, in `transitions.py`, `fitting.py` and `design.py`;
  `bayes.py` and `throw_shape.py` remain. **Applicability:** measured. The
  checkout phase and the length of a measurement session both move materially;
  the scoring advice and the design rankings do not.

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

### 4.1 Calibration against real match data — **started; one player done, the answer is "no"**

The model maps `σ` to a 3-dart average. That mapping should be checked the other
way round: take published professional statistics and ask whether *one* `σ` can
reproduce all of them at once.

Notebook 19 ran this for one player with 1,367 clean scoring visits. **One `σ`
cannot.** Matched on his three-dart average (σ ≈ 7.2 mm) the model reproduces his
180 rate (7.5% against 8.05 ± 0.74 observed) and his rate of poor visits
(19.4% against 19.97 ± 1.08), then predicts an exact 60 nearly twice as often as
it happens (17.3% against 10.53 ± 0.83). No `σ` repairs it in either direction.

The diagnosis is §1.2, not §1.1: the failure is over-dispersion of the visit
total, which a per-visit random aim offset explains and an aim-dependent `Σ` does
not. What remains here is to run it across many players rather than one, and to
add the checkout half — the per-double rates are §4.3 and are already done, but
checkout percentage *by score* is untouched and is the branch that would speak to
§1.1.

* **Feasibility:** high now — `data/real/` holds the cleaned data and
  `darts/calibration.py` the estimator.
* **Applicability:** high, and it is the thing that makes the project credible to
  a darts audience rather than a statistics one.

### 4.2 A real player, measured — **now blocked only on data**

Everything in this repository is simulated. The measurement protocol is designed,
certified, priced and tested against synthetic players; it has never been pointed
at a human. The protocol is short — a couple of hundred darts at the bull —
and the machinery to fit, advise and track in real time already exists.

Notebook 18 has since built the other half: fitting from *competition* scores, which
carry no aim point. Above a remaining score of 250 the aim is known to be the treble
20, so a visit total is an exact three-fold convolution of one dart's score
distribution, and about 2,000 scoring darts measure an elite player to +/-0.2mm. The
aim must be held at the bed centre to get there — letting it float reproduces
notebook 09's confounding and returns a sigma 30% low at small samples.

So there are two independent routes now, testing different things: a session with a
willing player measures one player *well*, and published match data measures many
players *badly but at scale*.

The interesting question is not whether it works but where the Gaussian model
*fails*: a real thrower may be skewed, heavy-tailed, or genuinely different at
different targets, and only a real session will say.

* **Feasibility:** high; it needs a player and an evening. **Applicability:** the
  highest of anything here, because every other result is conditional on it.

### 4.3 Per-double checkout rates — **done, with a twist worth following**

An isotropic throw aimed at the centre of a double bed hits it with a probability that
depends only on the bed's size, and every double bed is the same size. So the model
claims **all twenty doubles are equally hard**, to within 0.22 percentage points. A
throw with a 1.5:1 axis ratio instead says they vary by 14 points — 31% at the
double 20 against 45% at the double 6 — because a bed is 8mm deep and 52mm long
and which dimension your error runs into depends on where the bed sits.

Notebook 19 ran it on 15,874 attempts by 16 professionals. **The flat prediction
survives**: chi² = 20.8 on 19 df, p = 0.35, in a test with 80% power against a
5-point peak-to-trough spread.

The twist is underneath. The sixteen players do not agree with each other —
Q = 28.3 on 15 df, p = 0.020, I² = 47%, running from Whitlock at +9.0 points on
the side-versus-top contrast to Anderson at −13.0 — and the isotropic model says
every one of those numbers is exactly zero. The pooled null is an average over
players who individually deviate in both directions, which is what notebook 12
predicts if their groups have different aspect ratios. So the contrast is an
**estimator of a player's group shape, readable from a scoresheet**, and the
follow-up is to fit it per player and check it against their scoring `Σ`.

The confound flagged when this was written is still open: a player better at the
double 20 than the double 3 may have a tall group or may simply have thrown at
the 20 ten thousand more times, which is §1.1 wearing a disguise. The attempt
counts by double are in `data/real/double_attempts.csv`; matching on volume is
the next step and notebook 19 does not do it.

* **Feasibility:** done. **Applicability:** high — and the per-player version is
  the cheapest shape measurement in the project.

### 4.4 The prior on a lean is a guess

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

1. **Give the solver a state for the previous dart** (§1.2) — the largest of the
   three measured failures, and the only one the current state space cannot
   represent even in principle. Real players move target after a miss; the model's
   aim depends on the score alone.
   Nothing else on this list corrects a result that is already published.
2. **Measure a real player, now that it is worth more than it was** (§4.2) —
   everything else is conditional on it, and it is an evening's work plus a
   willing thrower. §1.5 has sharpened why. Notebook 23 can now fit a `ν` and a
   core from nothing but a scoresheet, but showed that scores pin down *that*
   there is a tail far better than *how heavy* it is; coordinates would settle in
   one evening what a scoresheet cannot settle at all. And since the same work
   roughly doubled the darts a measurement session needs, designing that session
   correctly matters more than it did.
3. **Calibration across many players** (§4.1) — notebook 19 fitted one. The estimator
   and the cleaned data are both in the repo, so this is now cheap, and checkout
   percentage by score is the branch that would speak to §1.1.
4. **Fix the missed-dart likelihood** (§2.1) — three hours of every rebuild, and the
   difference between 22 ms and half a second a dart in live play.
5. **Per-player group shape from the doubles** (§4.3, the open half) — matched on
   attempt volume, to separate shape from practice frequency.
6. **Aim-dependent accuracy** (§1.1) — the biggest remaining modelling gap.
7. **Action-set reduction** (§2.2), which makes the covariance grids of §3.2 and
   everything downstream affordable.
8. **Asymmetric abilities** (§3.1) — cheap, and answers a question players ask.
9. Board numbering (§5), everything else.
