# Darts

Optimise darts strategy with statistics: solve 501 exactly as a Markov decision
process, for a throw measured from a real player rather than assumed.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/mjc239/dartsstrats/main)

## What is here

A dart is modelled as a landing point around wherever the player aimed —
`Z ~ N(t + b, Σ)`, or, since notebook 21 measured real throws and found the
Gaussian beaten by every alternative tried, a **Student-t** of the same core.
Given either, three questions have exact answers:

* **Where should you aim?** Solve the leg as an MDP over `(score, dart in visit,
  score the visit started on)`, minimising **visits** rather than darts, which is
  the right objective for a race. The board itself supplies the structure that
  makes this cheap: a dart thrown at 62 or more cannot bust.
* **How good is this player?** Fit `(b, Σ)` from nothing but the scores they
  wrote down, by exact EM — no camera, no coordinates.
* **Where should they throw while being measured?** Fisher information over every
  target on the board, with an equivalence-theorem certificate proving the
  optimal design rather than merely searching for it. The answer is not where
  anyone would naturally practise.

The throw model has grown over the project from a single isotropic `σ` to a full
`(b, Σ)` — a size, a shape, a lean and a pull — and each addition was justified
by measuring what ignoring it costs, in visits per leg.

## Layout

| | |
|---|---|
| `darts/` | the library: board geometry, transition builders, MDP solvers, fitting, measurement design, online belief, and the models for what couples the darts of a visit |
| `notebooks/experiments/` | the numbered experiments — **start with [its README](notebooks/experiments/README.md)** |
| `scripts/` | the long-running computations whose outputs are committed under `results/` |
| `results/` | solved policies, design manifests, simulation studies |
| `tests/` | the notebook claims restated as assertions |
| `docs/research-roadmap.md` | what is done, what is open, and what to do next |

## Getting started

```bash
pip install -r requirements.txt
pytest -q                        # ~5 minutes
```

Then read `notebooks/experiments/README.md`, which maps the twenty-three
experiments, states what each one found, and lists what everything costs to
re-run.

## Three things to know before trusting a number

**Resolution.** Everything runs on a **512-pixel board**. An 8 mm scoring bed is
9.1 pixels across at 512 and 4.5 at 256, and the coarser board silently
misjudges any target defined by a bed — by a factor of six for a tight player at
the treble 20. Notebook 02 establishes this and notebook 09's appendix
demonstrates the failure.

**Almost all of it is simulated.** Notebooks 01–17 compute against a modelled
player. No real thrower has been measured with coordinates, and until one is, the
shape of a single dart's distribution is an assumption.

**A visit is not three independent darts, and the biggest reason is that the aim
moves.** Notebook 19 measured the coupling on 300,985 darts of professional
competition: hitting the treble 20 raises the next dart's chance of doing the
same from 22% to 40%. Notebook 20 took it apart. Professionals use **four**
scoring targets and step down them after a miss — from the 20, a miss goes to the
19 a quarter of the time — and the solver has no state for that at all, because
its aim point depends on the score and never on where the last dart landed.

**A dart is not Gaussian — it is Student-t.** Notebook 21 fitted six candidate
distributions to every player, held out. The Gaussian loses to every one of them
for every player, and the winner is a **Student-t with `ν ≈ 2.25`**, beating a
two-component mixture with one parameter where the mixture needs two. It has a
reading, not just a fit: a Student-t *is* a Gaussian whose width is redrawn for
every dart. The rival explanation — that the group is an ellipse rather than the
tail heavy — was fitted at the same price and gains nothing at all.

That has a consequence for every `σ` quoted anywhere in this project. With `ν`
near 2 a throw's variance barely exists, so `σ` is not a summary of one. The
familiar "elite ≈ 6.5 mm" is the **core scale** of a heavy-tailed throw (median
5.98 mm across professionals), not a standard deviation; fitting a Gaussian
instead returns 11.47 mm, a compromise between core and tail that describes
neither.

**Solving with the Student-t moves the checkout phase and nothing else.**
Notebook 22 puts the t into the transition builder — the only place in the
project that knows what a throw is — and re-solves. Matched on the three-dart
average, which is what the ability bands mean, the treble 20 / treble 19
crossover barely moves (16.80 mm → 16.5–16.9) and five of seven bands aim at the
same pixel, but the whole leg-length difference sits below 170. Its sign depends
on the player: elite and pro finish 0.11 darts sooner as a t, the middle bands
0.12–0.23 later, a pub player 1.61 sooner — because a t is better at an 8 mm bed
and worse at a whole sector, and legs at different standards are made of those in
different proportions. With 50 left and one dart, a Gaussian pro sets up for 32
and a Student-t pro throws at the bull.

**Measuring a Student-t player takes about twice as many darts.** Notebook 23
puts the t into the fitting and measurement-design machinery, where it costs a
single weight: `u = (ν+2)/(ν+q)` in the EM's E step, and the same factor
multiplying the Gaussian score function in the Fisher information. That weight is
the whole story — a dart a long way from the aim point is powerful evidence of a
wide *player* under a Gaussian and is discounted under a t, because a wide *dart*
explains it. So a Gaussian fitted to Student-t darts returns a core-versus-tail
compromise (12.45mm for a thrower whose core is 8.0mm), and a heavy tail carries
*less* information about the core than a Gaussian does. Where to throw to be
measured survives; how long does not — **233 darts to prove a millimetre becomes
506**, and a pub player's 4,857 becomes 23,634.

**And check the data before you trust a tail.** Notebook 21 exists because
notebook 20 reached the opposite conclusion from the same question. The 2017 feed
carries the previous leg's finishing darts into the next leg's opening visit, in
6.7% of player-legs — and since every visit with 430 or more remaining sits near a
leg start, that visit was most of the sample. It manufactured a far tail that
about half of notebook 20's non-Gaussianity was fitting.
