# Where to take the darts MDP work next

A ranked guide to the directions that look most promising, judged on two axes:
**feasibility** (can it be computed, and is the data available?) and
**applicability** (would it change what a real darts player does?).

---

## Where things stand

| Model | Status | Cost at full resolution |
|---|---|---|
| Single dart, expected score | published | FFT, milliseconds |
| Single player, memoryless MDP (`darts/mdp.py`) | published | minutes |
| Single player, 3-dart visits (`darts/mdp_3turn.py`) | solved exactly, values + policy | ~4 s for 501 at 7.4k aiming points, ~15 s at 30k |
| Two players, 1 dart per turn (`darts/mdp_2player.py`) | solved exactly | one GEMM per diagonal |
| Two players, 3 darts per turn (`darts/mdp_2player.py`) | solved exactly | dominated by the low-score region; needs a reduced aiming grid |
| Sets and legs | not started | trivial once leg values exist — see §3.1 |

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

---

## 1. Modelling fidelity — the cheapest way to change the answers

The solvers are exact given the model. Everything below changes the *model*,
costs almost nothing extra to compute, and would change real recommendations.
This is where the best return is.

### 1.1 A per-player covariance matrix, estimated from real throws — **do this first**

The repo already has the machinery: `darts/stats.py` supports a general `Sigma`,
and there is an EM routine for fitting a throwing distribution
(`ec83499 EM algorithm for throwing dist`). What is missing is joining it to the
MDP: fit `Sigma` from a player's own darts, then solve *their* MDP.

This matters because the spherical assumption is wrong in a specific direction.
Right-handed players throw with more variance along one axis and a slight tilt,
and the earlier post already showed that a tilted `Sigma` can move the best
single-dart aim from the treble 20 to the treble 15. The same asymmetry will
change which double a player should leave themselves on — a player whose spread
is vertical should prefer the doubles on the side of the board, where the bed is
"wide" in the direction they miss.

* **Feasibility:** high. `ThreeDartMDP` already takes an arbitrary transition
  matrix, and `transitions.transition_arrays(..., Sigma_mm=...)` already accepts
  a full covariance. Data collection is the only real work: a phone camera and
  the existing board-registration code, or a few hundred manually recorded darts.
* **Applicability:** very high. This is the difference between "here is a
  checkout chart" and "here is *your* checkout chart".

### 1.2 Accuracy that depends on where you aim — **the biggest single modelling gap**

The model assumes one `Sigma` everywhere on the board. Real players are far more
accurate at the treble 20 than at the treble 11, because that is what they have
practised, and many are measurably worse throwing at a double under pressure.

Both are one-line changes to the transition builder — `Sigma` becomes a function
of the aiming point — and neither costs the solver anything, because the solver
only ever sees the resulting `(n_points, n_scores)` matrix.

The strategic consequences are large and testable:

* A player who is 20% worse at doubles than at trebles should leave themselves on
  a *different* number than the model currently says.
* A player with a strong 19 and a weak 20 may have a completely different scoring
  route, and the MDP will find it automatically.

* **Feasibility:** high to build, medium to calibrate — you need enough data per
  region, which means practice-session logging rather than a single sitting.
* **Applicability:** very high. Every player already believes this about
  themselves; the model would tell them what to do about it.

### 1.3 Darts within a visit are not independent

The model treats the three darts of a visit as i.i.d. draws around whatever the
player aims at. In reality, a player who sees dart 1 land 15mm high adjusts. That
correlation is exactly what the 3-dart state space is built to represent, and
adding it needs no new solver: extend the within-visit state with a coarse
summary of the previous dart's error (say a 3×3 grid of "where the last dart
went"), and let the transition matrix depend on it.

Cost: multiplies the within-visit state count by the number of error bins, which
is affordable — the within-visit states are a small fraction of the work.

* **Feasibility:** medium. The solver change is contained; the data to estimate
  "how much do players correct" is the hard part.
* **Applicability:** high, and it is the most *scientifically* interesting of
  these, because nobody has quantified the value of in-visit correction.

### 1.4 Bounce-outs, wired darts, and the dart already in the board

Small effects, easy to add as a fixed probability of scoring zero (bounce-out) or
as a reduced effective area for a bed that already holds two darts. The second
one is a genuine reason not to aim at the same treble three times, and the model
currently cannot express it.

* **Feasibility:** high. **Applicability:** low-to-medium; worth a footnote
  rather than a project.

---

## 2. Making the computation bigger

### 2.1 Reducing the set of aiming points — **the enabler for everything else**

Every Q-value in every one of these models is a linear functional of a point's
score distribution, so only points on the convex hull of those distributions can
ever be optimal. A quick experiment (7,573 points, 512-pixel board, σ = 15mm)
found ~1,950 points that are the argmax for some isotropic random direction, so
the hull is not tiny — but isotropic directions are far broader than the value
vectors that actually arise, which are monotone in score. Taking the union of the
points that are optimal *somewhere* in a family of single-player problems gives a
much smaller set: `mdp_2player.candidate_points` cuts a 1,893-point grid to 195
(10%), and re-solving the single-player 501 problem on the reduced grid changes
the answer by less than 1e-6 darts.

Worth pushing further, because a 10× reduction in the action set is a 10×
reduction in every solve, and it turns the full 501×501 three-dart two-player
game from "overnight" into "over lunch". Two routes:

* **Rigorous:** an LP per point testing whether its distribution lies inside the
  convex hull of the retained set. ~7,500 small LPs; certifies that discarding it
  cannot lose anything for *any* value function.
* **Practical:** the policy-union heuristic above, then verify by recomputing the
  Bellman maximum over the full set at a sample of states and reporting the gap.

* **Feasibility:** high. **Applicability:** indirect but large.

### 2.2 The low-score region of the two-player game

In the two-player game, every state whose score is below 182 needs its own
within-visit sweep, and there are `O(180 × game_start)` of them — this is what
dominates the cost. There is a way out that mirrors the single-player trick:
for a fixed opponent score, every within-visit value is a **piecewise-linear
convex function of one scalar** (the value of the visit ending back where it
started). Storing the upper envelope of those lines — usually a handful of
segments — instead of recomputing the maximum over thousands of aiming points
would collapse the low-score work by orders of magnitude.

* **Feasibility:** medium; a self-contained piece of geometry (2-D convex hulls
  of (slope, intercept) pairs). **Applicability:** indirect.

### 2.3 Continuous aiming points

The aiming grid is currently a discretisation. Since the score-probability
functions are smooth in the aim location, the optimum could be refined by local
continuous optimisation from the grid argmax. This mostly matters when reporting
*where* to aim to sub-millimetre precision, which is beyond what a player can
act on — so it is a nicety, not a priority.

* **Feasibility:** high. **Applicability:** low.

---

## 3. Two-player and match play

### 3.1 Sets and legs — **the best effort-to-payoff ratio in this whole list**

Once `W[u, v]` is known, the leg is a black box: two players of given abilities,
one throwing first, produce a single number `p = W[501, 501]`. A match is then a
tiny MDP over `(legs won by A, legs won by B, sets, who throws first)` — a few
thousand states, milliseconds to solve.

That immediately answers questions people actually argue about:

* How much is winning the bull-up worth in a best-of-11? (It is worth much more
  in short formats, and the model quantifies it exactly.)
* How much better does player B have to be to overcome throwing second?
* Is the "sets" format more or less favourable to the underdog than "legs"?

Nothing new has to be invented; it is an afternoon's work on top of the leg
solver.

* **Feasibility:** very high. **Applicability:** high, and highly publishable as
  a post.

### 3.2 Where two-player strategy actually differs from single-player

The interesting output of the leg game is not the win probability, it is the
*difference* between the win-maximising policy and the darts-minimising one. The
places to look:

* **Opponent on a finish.** When the opponent will probably check out next visit,
  your last dart should go at a double even from a poor position, because
  "leaving a good number" is worth nothing if you never throw again. The
  single-player model can never produce this.
* **Big lead.** When far ahead, the win-maximising policy should get *more*
  conservative around busts than the darts-minimising one.
* **How large are these effects?** Now measured, for the one-dart game: ignoring
  the opponent entirely costs **at most 0.5 percentage points of win probability
  anywhere on the board**, and the worst cases are all at low scores (0.0049
  below 60, against 0.0014 above 120). That justifies using the much cheaper
  single-player policy for the whole scoring phase and solving the two-player
  game only near the finish — which is also the region where it is cheapest.
  Worth repeating for the three-dart game, where the endgame effects should be
  larger, because a visit gives three chances to react to the opponent.

  Beware the obvious-looking measure here: counting states where the two
  *argmaxes* differ reports ~17% at high scores as well as low, purely because
  many aiming points are near-ties there and the argmax flips between
  neighbouring pixels for reasons worth a millionth of a win. Measure the value
  given up, not the label.

Note also that the *single-player* objective matters here: minimising darts
thrown treats a bust on the first dart as costing one dart, since the other two
are never thrown, while minimising visits charges it a whole visit. The second is
the right proxy for a race. Both are supported (`dart_cost` / `turn_cost`), and
they disagree on several dozen first-dart aims.

* **Feasibility:** high for the 1-dart game, medium for 3-dart at full
  resolution (see §2.1, §2.2). **Applicability:** medium — the answers are
  interesting but affect only the endgame.

### 3.3 Asymmetric abilities

`W[u, v]` for two *different* players is the same computation with two transition
matrices. It answers "how should I play differently against a better player?"
— the classic intuition is that an underdog should take more risk, and this model
can say exactly where and how much.

* **Feasibility:** high (it is a parameter change). **Applicability:** high.

---

## 4. Outputs a real player could use

### 4.1 An ability-tailored checkout chart

Every checkout chart in every pub is the same chart, computed for a notional
perfect player. The 3-dart model produces the correct chart for *any* ability,
and the analysis notebook shows they genuinely differ: strong players should go
at the double on every dart of a visit, while weak players should use the last
dart to protect the number instead of chasing the double.

Deliverable: a table, per ability band, of what to aim at for every score from
2 to 170 and for each dart of the visit. This is a genuinely new artefact.

* **Feasibility:** done, essentially — it falls out of the solved policy.
* **Applicability:** the highest of anything here.

### 4.2 A practice-value calculator

Differentiate the answer with respect to the model inputs:

* What is 1mm of `σ` worth, in darts per leg? (From the notebook: around 1.5
  darts per leg per mm for a club player — a startlingly large number, and a good
  way to communicate why accuracy matters more than knowing checkouts.)
* What would getting better *only at doubles* be worth, versus only at trebles?
  This is the practice-allocation question every player faces, and it is a
  straightforward comparison of two perturbed models.

* **Feasibility:** high. **Applicability:** very high.

### 4.3 Calibration against real match data

The model currently maps `σ` to a 3-dart average. That mapping should be checked
the other way round: take published professional statistics (3-dart averages,
checkout percentages by score, first-nine averages) and ask which `σ` reproduces
them, and whether *one* `σ` can reproduce all of them at once. If it cannot —
for instance if pros' real checkout percentages are worse than the model predicts
at their scoring `σ` — that is direct evidence for the aim-dependent accuracy of
§1.2, and a genuinely interesting result.

* **Feasibility:** medium; the data is public but needs scraping and cleaning.
* **Applicability:** high, and it is the thing that would make the whole project
  credible to a darts audience rather than a statistics one.

---

## 5. Board design

The Quadro board work already in the repo extends naturally: with a full-game
solver rather than a single-dart one, you can ask whether the Quadro board makes
*legs* shorter, whether it widens or narrows the gap between strong and weak
players, and whether it makes the endgame more or less interesting. The same
machinery answers "what would a better dartboard numbering look like?" — the
standard arrangement is designed to punish inaccuracy, and one can ask which
arrangement maximises or minimises the skill gap.

* **Feasibility:** high (a board is just a different pixel array).
* **Applicability:** low for players, high for a good post.

---

## Suggested order

1. **Ability-tailored checkout charts** (§4.1) — already computable, best payoff.
2. **Sets and legs on top of the leg solver** (§3.1) — an afternoon, lots of
   quotable results.
3. **Fit `Sigma` from real throws and solve a personal MDP** (§1.1).
4. **Practice-value calculator** (§4.2) — falls out of 3.
5. **Aim-dependent accuracy** (§1.2) — the biggest modelling gap.
6. **Action-set reduction** (§2.1), then the full two-player 3-dart game.
7. **Calibration against professional statistics** (§4.3).
8. In-visit correlation (§1.3), board design (§5), everything else.
