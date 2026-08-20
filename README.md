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
its aim point depends on the score and never on where the last dart landed. A
second failure came out of the same work: a single dart's Gaussian tails are far
too thin to produce the doubles and misses real players actually throw.

What that costs: per-dart results stand, but anything reading the *spread* of a
visit total (checkout probability, bust risk, the value of the throw) rests on a
distribution that is too thin, and every confidence interval in the fitting
notebooks is too narrow. Notebook 20's verdict works through which is which.
