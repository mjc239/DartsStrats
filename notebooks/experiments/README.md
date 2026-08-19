# The experiments

Eighteen notebooks, in six phases. Each phase exists because the previous one
exposed a gap, so they read best in order — but every notebook states its own
question and answers it, and the verdict sections are self-contained.

Everything below is **simulated**. No real player has been measured yet, which is
the largest single caveat on the whole project -- notebook 18 is the preparation
for removing it.

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

---

## Run times

Measured on 4 cores. Notebooks marked ◦ ran with two or three jobs sharing the
machine, so those are upper bounds.

| | notebook | time |
|---|---|---|
| 06 | Legs, sets, and the bull-up | 4 s |
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
| 15 | Fast enough to play with | 15 m 16 |
| 16 | The lean of a throw | 60 m ◦ |
| 14 | Shape and pull together | 68 m ◦ |
| 13 | Aiming off | 69 m |
| 17 | Measuring all of it | **3 h 25** ◦ |

**Total ≈ 6 h 22.**

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

### The scripts behind 09 and 17

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

**Total ≈ 7 h 56**, so a full rebuild from nothing is about **14 hours**.

Run the design scripts in the order listed above: the two-stage study needs the
lookup table and the robust design, and the simulation study needs the per-band
designs.

The assertion-shaped versions of these results — the ones that fail loudly
rather than needing a table read — are in `tests/`.
