# The experiments

Twenty-three notebooks, in six phases. Each phase exists because the previous one
exposed a gap, so they read best in order — but every notebook states its own
question and answers it, and the verdict sections are self-contained.

Notebooks 01–17 are **simulated**: no real player is measured in them. Notebook
18 builds the machinery for meeting real data and **19, 20 and 21 use it** on
300,985 darts of professional competition. Read those three together before
trusting anything in phases 2 and 5 too literally — and read them **in order**,
because each one substantially revises the last.

19 refutes the independence assumption every transition matrix is built on. 20
finds that about half of what 19 measured is the *aim* moving between darts,
which the solver has no state for at all. 21 finds a **data defect** underneath
20's other conclusion, and that once it is removed a dart is a **Student-t**
rather than a Gaussian — and that most of the dependence 19 and 20 were chasing
was one dart being described badly. **22 acts on it**: it solves 501 with the
Student-t and reports which of the project's answers move. **23** does the same
for the other half of the project -- fitting a player and designing the session
that measures them.

All notebooks run on a **512-pixel board** with a **3.52 mm aiming grid**
(`point_stride=4`). Notebook 02 is where that choice is justified: an 8 mm
scoring bed is 9.1 pixels across at 512 and only 4.5 at 256, and the coarser
board misjudges any target defined by a bed.

---

## 1 · Trust the machinery

| | asks | finds |
|---|---|---|
| **01** Is the 3-dart value function right? | Does the fast solver match a reference that assumes nothing? | Yes, to 1e-13. The speedup rests on a board fact: a dart thrown at 62+ cannot bust, so visit-start independence begins at 62 / 122 / 182 |
| **02** Millimetres, pixels, and the board | Is the mm → pixel → score chain correct? | Correct, checked against exact geometry with no pixels involved. **Work at 512 pixels or finer** |
| **03** Did the geometry fixes change anything? | Two bugs were found; did any published result move? | No. One number amended: the T20/T19 crossover, 17 mm → 16.8 mm |

## 2 · What the model says about playing

Strategy results, assuming the player's `σ` is already known.

| | asks | finds |
|---|---|---|
| **04** The two-player leg | How much is the opponent worth? | Throwing first ≈ **7 points of win probability**. Knowing the opponent's score is worth ≤ 1.4 points and only below 120. Also: never measure policy differences by counting argmax changes — it overstates by ~5× |
| **05** Checkout charts for *your* ability | Should everyone use the same chart? | No. The scoring half is common to everyone, the checkout half is not. Standard charts are actively wrong for weak players, usually by saying "go at the double" where the model says protect the number |
| **06** Legs, sets, and the bull-up | What is the throw worth, by format? | **7.4 points** in a best of 3, **1.0** in a best of 13 sets. "Two clear legs" makes the throw worth exactly nothing between equals |
| **08** What is practice worth? | Trebles or doubles? | A millimetre is worth **0.43–0.58 visits per leg**, remarkably flat across ability. **Trebles for strong players, doubles for weak ones** — the bottleneck is scoring at the top and finishing at the bottom |

## 3 · Measuring a real player

Everything above needs a `σ`. These ask where it comes from.

| | asks | finds |
|---|---|---|
| **07** Fitting a throw from scores | Can `σ` be recovered from a scoresheet? | Yes, by exact EM. But 200 darts is **16–29% biased low**, and at 1000 the likelihood is genuinely **bimodal** — "ordinary throw aimed where we thought" against "tighter throw aimed lower" |
| **09** Where should you throw to be measured? | Which target, and is one enough? | **Bull if σ ≲ 13 mm; the big single at ~136 mm for 14–22 mm; the treble ring beyond.** T20 — where everyone would practise — costs 3.1× (league) to 5.5× (elite) in variance. Splitting the session is worth 9–16% if you know `σ`, and decisive if you do not (28% → 63% worst-case efficiency) |
| **10** Can you tell you've improved? | Power analysis on 08 + 09 | An elite player proves a millimetre in **233 darts** a session; a pub player needs **4,857**. The players with most to gain are the ones who cannot verify it |

## 4 · Learning while playing, and a richer player

09 needs a scheduled session. 11 asks whether the match will do instead — and
then the model starts growing.

| | asks | finds |
|---|---|---|
| **11** Learning the player as they play | Prior from an ability band, updated every dart | Works. A fixed chart for the wrong standard costs **0.39 visits per leg forever**; the same wrong belief costs 0.077 *once* if allowed to update |
| **12** The shape of a throw | Is a tall group different from a round one? | Yes. Doubles swing **1.37×** across the board at the same overall spread, and the favourite doubles rotate 90° between a tall and a wide player. Ignoring the shape costs 0.11 visits/leg |
| **13** Aiming off | And a systematic pull? | **Bias does not change the board** — the game is translated, so it is free to compute. Ignoring a 10 mm pull costs **0.89 visits per leg, the largest modelling error in the project.** Match play identifies a pull where a single-target drill cannot |
| **14** Shape and pull together | Do they interact? | Mildly sub-additive. The real finding: **the shape decides which pull hurts, by 4.5×.** "How bad is my pull?" has no answer without the shape of the group it sits in |
| **16** The lean of a throw | Does a tilted group matter? | **0.61 visits per leg** between the best and worst orientation of an *identical* ellipse. It is the board's **numbering**, not its geometry — the 20 is flanked by 1 and 5, so sideways error is punished. Measured at the bull, where every segment boundary meets |

## 5 · Making it usable, and closing the loop

| | asks | finds |
|---|---|---|
| **15** Fast enough to play with | The 4-parameter grid costs ~4 s a dart | A Liu–West particle filter gets a scoring dart to ~22 ms. Two traps documented: static parameters need **rejuvenation** or the filter is confidently wrong, and "drift" is meaningless as a scalar across parameters in different units |
| **17** Measuring all of it | 09 answered the design question for *one* parameter; redo it for all five | **Throw at the bull; 200 darts is plenty.** 09's ring is only 51% efficient for scoring, because it optimises the coordinate worth least — bias is worth ~30× the shape. Plain D-optimality reaches 97% of the decision-optimal design. A match learns bias and spread nearly as well but the lean 2.5× worse, and deliberate mid-match measuring takes ~900 legs to repay |

---

## 6 · Meeting reality

| | asks | finds |
|---|---|---|
| **18** Calibrating against real scores | A scoresheet has no aim point. Can the model be fitted to one anyway -- and can it be *refuted*? | **Yes to both.** Above a remaining score of 250 the aim is known to be the treble 20, so a visit total is an exact three-fold convolution and ~2,000 scoring darts measure an elite player to ±0.2mm. More useful: one σ must explain the average, the 180 rate *and* the checkout rate at once. **An isotropic throw says all twenty doubles are equally hard to within 0.22 points; a 1.5:1 tall throw says they vary by 14** -- so ~200 attempts at each of two doubles tests the model's oldest untested assumption |
| **19** What real darts says | 300,985 professional darts. Three predictions, tested | **Doubles: flat survives** (p = 0.35, with 80% power against a 5-point spread) — but the sixteen players disagree with each other in both directions (I² = 47%), which is a shape signal, not noise. **Visit totals: right at the extremes, wrong in the middle** — no `σ` reproduces the 180 rate and P(exactly 60) at once. **Independence: refuted.** Hitting T20 lifts the next dart's chance by **18–22 points** (z > 35). Not player pooling (34 of 35 individuals), not form drift (the lift is *negative* across the gap between visits), not the filter — but about half of it turns out to be the *aim* moving, which none of those three checks could catch. See 20 |
| **20** What couples the darts | 19 says the darts are not independent. What replaces the assumption? | **Mostly not the throw.** Professionals use **four** scoring targets and step down them after a miss — from the 20 a miss moves to the 19 24.8% of the time, from the 19 to the 18 35.7%. That is about half of what 19 measured, it replicates across both feeds, and it costs them nothing (**−0.45 ± 0.47** points) because the 19's neighbours pay more than the 20's. The model has no *state* for it. **Its second conclusion — that a dart's tails are too thin for a Gaussian — is superseded by 21**, which found the tail was a data defect and the fix a different distribution rather than a wider one |
| **21** Is a throw Gaussian? | A mixture patched the tail. Is that what a throw *is*? | **No, and the tail was not a throw.** The 2017 feed leaks the previous leg's checkout darts into the next leg's opening visit — 6.7% of player-legs against 0.13% in the clean feed — and that visit is **78%** of the pure-scoring sample. 100% of the double 20s sat at exactly one score. Cleaned, the Gaussian still loses to five of six candidates for **all 17 players**, and the winner is a **Student-t, `ν ≈ 2.25`**, beating a two-component mixture with **one** parameter against two. The rival explanation — an elliptical group — was priced identically, shown to be detectable on simulated data, and gains **nothing** (−0.02). Two knock-ons: the per-visit coupling 20 selected is worth **+0.002** on top of a Student-t against +0.51 on a Gaussian, and `σ` is not a standard deviation — the familiar 6.5 mm is a heavy-tailed **core scale** |
| **22** What changes if the dart is Student-t? | 21 says a dart is a t. Solve 501 with one and see what moves | **The scoring phase does not care; the checkout phase does.** Matched on the three-dart average — the thing the ability bands already mean, and not on σ or on variance, which give a "pro" who averages 147 — the T20/T19 crossover moves from 16.80 mm to 16.5–16.9, five of seven bands aim at the same pixel, and **the whole difference in the leg sits below 170**. The sign depends on who throws: elite and pro finish **0.11 darts sooner**, the middle bands 0.12–0.23 later, a pub player **1.61 sooner**. The mechanism is measured, not assumed — a matched t is up to 15% better at an 8 mm bed and 2–12% worse at a whole sector. Sharpest consequence: with **50 left and one dart, a Gaussian pro sets up for 32 and a t pro throws at the bull** |
| **23** What does it cost to measure a Student-t player? | 22 taught the solvers to throw a t. Nothing could yet *estimate* one | **Fitting one costs a single weight; measuring one costs twice the darts.** A t is a Gaussian whose width is redrawn each throw, so the dart's width is a second latent and the E step gains `u = (ν+2)/(ν+q)` -- the M step is the same weighted Gaussian one, and the design side's score function is the Gaussian's times *the same* `u`. From 750 darts of an 8.0mm-core t the t fit returns 8.09mm and a Gaussian returns **12.45mm**. `ν` is profiled, not estimated: the profile separates a real tail (**+57** log-units) from none (**+0.75**) decisively but pins `ν` itself only loosely from scores. Where to throw survives -- bull, then big single, then treble ring -- but the bull stays optimal further out. How long does not: **233 darts to prove a millimetre becomes 506**, and a pub player's 4,857 becomes **23,634**. The roadmap asked which way heavy tails move the information; the answer is *less*, in every band. The normalisation `transitions.py` had to rebuild was worth only **0.2–5.2%** here, and in the safe direction — the score function was what mattered |

---

## Run times

Measured on 4 cores. Notebooks marked ◦ ran with two or three jobs sharing the
machine, so those are upper bounds.

| | notebook | time |
|---|---|---|
| 06 | Legs, sets, and the bull-up | 4 s |
| 19 | What real darts says | 6 s |
| 04 | The two-player leg | 10 s |
| 01 | Is the value function right? | 1 m 13 |
| 02 | Millimetres, pixels, and the board | 1 m 16 |
| 18 | Calibrating against real scores | 1 m 36 |
| 05 | Checkout charts | 2 m 38 |
| 10 | Can you tell you've improved? | 2 m 55 |
| 09 | Where to throw to be measured | 3 m 14 |
| 11 | Learning the player as they play | 3 m 20 |
| 07 | Fitting a throw from scores | 6 m 25 ◦ |
| 12 | The shape of a throw | 6 m 56 ◦ |
| 08 | What is practice worth? | 8 m 19 |
| 03 | Did the geometry fixes change anything? | 8 m 33 |
| 22 | What changes if the dart is Student-t? | 3 m 04 |
| 23 | Measuring a Student-t player | 13 m 22 |
| 21 | Is a throw Gaussian? | 11 m 14 |
| 15 | Fast enough to play with | 15 m 16 |
| 20 | What couples the darts | 22 m 24 |
| 16 | The lean of a throw | 60 m ◦ |
| 14 | Shape and pull together | 68 m ◦ |
| 13 | Aiming off | 69 m |
| 17 | Measuring all of it | **3 h 25** ◦ |

**Total ≈ 8 h 34.**

Notebook 19 is six seconds because all the work is upstream: it reads the files
`scripts/build_real_data.py` produces. That build is not in the total — it
downloads from pinned upstream commits and is bounded by the network, not by
this machine. See `data/real/README.md`.

### Why the slow ones are slow

Four cost drivers, in order of how much damage they do.

**A missed dart costs a hundred times what a hit does.** The per-dart likelihood
sums Gaussian density over every pixel carrying the observed score. A 20 is one
bed — 5,464 pixels at 512. A **miss** is the entire non-scoring board — 145,191.
So a scoring dart costs 27 ms and a missed one **3.4 s**. In simulated match play
**11.6% of darts score zero**, which is ~250 s per 600-dart session. Notebook 17
runs 40 such sessions, so misses alone account for roughly **2.8 of its 3.4
hours**. This is an implementation artefact, not anything about the model; see
the open item in the roadmap.

**The grid posterior pays that tax at every grid point.** Notebook 14 carries an
8,019-point joint grid over `(Σ, b)` and reports 3,383 s for 810 darts — about
four seconds a dart, and 83% of the notebook. Notebook 13 carries 2,431 points
across five sessions of two protocols: 1,855 s. Notebook 15 exists because of
exactly this, and replaces the grid with a particle filter.

**A tilt multiplies the solve grid.** Anisotropy needs one MDP solve per
covariance; a correlation adds an axis to that grid. Notebook 16 solves 45 MDPs
(333 s) where the untilted notebooks solve 25. Bias, by contrast, is free — it
relabels actions rather than changing the board, which is notebook 13's opening
theorem.

**512 pixels is ~1.2× on MDP solving but 3–4× on anything per-pixel** — which is
most of the above. 13, 14, 16 and 17 were all considerably quicker on the coarser
board, and considerably less trustworthy.

### The scripts behind 09, 17, 20, 21 and 22

These are not in the notebook times. Their outputs are committed under
`results/`, so the notebooks re-run without re-solving.

| script | time | writes |
|---|---|---|
| `robust_measurement_design.py` | 53 s | `manifest_design_robust.csv`, `manifest_design_cross.csv` |
| `optimal_measurement_design.py` | 4 m 55 | `manifest_design.csv`, `design/design_{band}.npz` |
| `best_target_by_sigma.py` | 6 m 27 | `manifest_best_target.csv` |
| `why_splitting_helps.py` | 22 m | `design/why_splitting.csv` |
| `two_stage_design.py` | 1 h 18 | `design/two_stage.csv` |
| `decision_weight.py` | 1 h 46 | `design/decision_weight.npz` |
| `design_simulation_study.py` | **4 h 02** | `design/simulation_league.csv` |
| `calibration_recovery.py` | 16 m | `calibration/recovery.csv` |
| `dependence_fits.py` | 47 m | `dependence/fits.csv`, `dependence/signatures.csv` |
| `throw_family_fits.py` | 1 h 06 | `throw_family/fits.csv` |
| `solve_single_player.py --nu 2.25 3 5` | 15 m | `manifest_student_t.csv`, `student_t/{band}_nu{nu}_{obj}.npz` |

**Total ≈ 10 h 04**, so a full rebuild from nothing is about **18 hours**.

Run the design scripts in the order listed above: the two-stage study needs the
lookup table and the robust design, and the simulation study needs the per-band
designs.

The assertion-shaped versions of these results — the ones that fail loudly
rather than needing a table read — are in `tests/`.
