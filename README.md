# Darts

Optimise darts strategy with statistics: solve 501 exactly as a Markov decision
process, for a throw measured from a real player rather than assumed.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/mjc239/dartsstrats/main)

## What is here

A dart is modelled as a Gaussian landing point, `Z ~ N(t + b, Σ)`, around
wherever the player aimed. Given that, three questions have exact answers:

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

Then read `notebooks/experiments/README.md`, which maps the twenty
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

**And check the data before you trust a tail.** Notebook 21 exists because
notebook 20 reached the opposite conclusion from the same question. The 2017 feed
carries the previous leg's finishing darts into the next leg's opening visit, in
6.7% of player-legs — and since every visit with 430 or more remaining sits near a
leg start, that visit was most of the sample. It manufactured a far tail that
about half of notebook 20's non-Gaussianity was fitting.
