# `data/real/` — handoff

Real observed darts data, for calibrating the throw model against actual competition
results rather than a simulated player. This note is written for whoever picks up the
coding work next. `PROVENANCE.md` alongside it is the audit trail — sources, retrieval
dates, definitions, discrepancies. Read this first, then that.

## Get the data

```bash
pip install openpyxl pyreadr
python scripts/build_real_data.py        # ~1 min
```

The CSVs are **gitignored on purpose** — ~35 MB, and none of the four upstream repos
carries a licence, so we regenerate rather than redistribute. Upstreams are pinned to
commit SHAs and output is byte-for-byte deterministic, checked against `CHECKSUMS.txt`
on every run. `--verify-only` re-checks without rebuilding.

## What is here

| file | rows | what it is |
|---|---|---|
| `target_outcomes.csv` | 1,489 | **intended target known.** Per player and target region, the distribution of beds actually hit. 2019, top 16. |
| `double_attempts.csv` | 321 | player × double × attempts × hits. A rollup of the above. |
| `per_dart.csv` | 300,985 | every dart: player, leg, visit, dart index, `score_before`, bed hit, value. Target **not** known. |
| `per_visit.csv` | 103,158 | three-dart totals with `score_before`. Derived from `per_dart.csv`. |
| `match_aggregates.csv` | 188 | per player per match. Has a real `doubles_attempted`, not a bare percentage. |

## Start here: `target_outcomes.csv` ↔ `fit_multi_target`

This is the direct wiring, and it is why `target_outcomes.csv` is the most valuable file
despite being the smallest. `darts.fitting.fit_multi_target` takes `(target_mm, scores)`
pairs and fits `(b, Σ)` — five parameters whatever the number of targets. That is
precisely the shape of `target_outcomes.csv`: a known target region, and the scores that
resulted. 16 professionals, ~170k darts of it, at T20/T19/T18/T17, every double, and bull.

Fitting this gives a `Σ` measured from elite play, against which every simulated ability
band in `results/` can be sanity-checked. As of now the README's own caveat stands —
"it is all simulated… that has not been done". This data is what closes that gap.

### Three things that will bite you

**1. Label vocabulary does not match.** `darts.utils.region_label` emits singles as a bare
number and misses as lowercase: `'T20'`, `'D16'`, `'19'`, `'25'`, `'BULL'`, `'miss'`.
These CSVs use `S19` and `MISS`, and `target_outcomes.csv` additionally uses `DB`/`SB` for
the bull *as a target*. Convert before touching anything in `darts/`. A tested adapter is
below.

**2. The aim point within a region is unknown.** We know a dart was aimed at "T20", not
where in the T20 bed. The source paper (arXiv:2302.10750 §3) treats this as latent and
fits it by EM; `fit_multi_target`'s `b` will absorb part of it, and `shared_bias=False`
gives each target its own mean if you suspect it varies. Do not silently assume the bed
centre is the aim point — that assumption is a modelling choice worth an experiment, not
a detail.

**3. Outcome beds are censored to the target's neighbours.** A T20 attempt can only resolve
to T20/S20/T5/S5/T1/S1 — the source records nothing else, so a wild dart is absent rather
than recorded as a miss. The likelihood must condition on this support, or `Σ` will come
out too tight. This is the single most likely way to get a wrong answer from this file.

### Adapter (tested against the current `region_label` vocabulary)

```python
def to_repo_label(bed):
    """CSV bed -> darts.utils.region_label vocabulary."""
    if bed in ("MISS", "miss"):        return "miss"
    if bed in ("BULL", "DB"):          return "BULL"
    if bed in ("25", "SB"):            return "25"
    if bed.startswith("S"):            return bed[1:]      # S19 -> '19'
    return bed                                             # T20, D16 unchanged
```

Then `darts.checkout._label_to_score` turns the label into points, so
`target_outcomes.csv` expands to the `scores` sequence `fit_multi_target` wants by
repeating each outcome `count` times.

## `per_dart.csv` / `per_visit.csv`

300,985 darts of match play. Useful for the MDP side — empirical visit-score
distributions, real checkout behaviour, how professionals actually route finishes — and
for validating policies against what players did.

**Do not feed these to `fit_from_scores`.** Its docstring is explicit: it assumes every
throw is aimed at the same unknown point, "a practice session at one target, not a bag of
match darts". Match darts are aimed all over the board. Recovering `Σ` from them needs the
target inferred from `score_before`, which is a real modelling problem and not currently
solved anywhere in this repo. `score_before` is retained on every row precisely so that
work is possible.

Gotchas, all documented at length in `PROVENANCE.md`:

- **`post_bust_visit`** — the 2017 feed keeps subtracting after a bust instead of
  reverting, so `score_before` goes negative. 56 visits, flagged not deleted. Filter
  `post_bust_visit == 0` for a clean state variable.
- **`visit_index` is global within the leg**, alternating between players — it is not a
  per-player counter. Group by `(leg_id, player, visit_index)`.
- **Two eras, five years apart.** `dartsviz_pdc_2017` and `sportradar_pdc_wc_2022`. Peter
  Wright's 18,376 darts are two different Peter Wrights. Split on `source` unless you have
  a reason not to.
- **`raw_segment` distinguishes inner from outer singles** in the 2022 source (10 vs 11) —
  radial information that `bed` throws away, and directly relevant to a 2-D landing model.
  Worth using.
- **42 rows name a bed but score zero** (2022 source only) — bounce-outs or voided darts
  where the feed logged the segment anyway. `value` is authoritative and is what the leg
  replay used; `bed` is wrong on exactly these rows. Filter with
  `value == 0 and bed != 'MISS'`. A build gate asserts the count stays at 42.
- `match_id` is empty for 39,871 rows; no leg→match mapping exists upstream.

## Ground rules

These are the constraints the data was collected under. Please keep them.

1. **Never invent, interpolate, or estimate a value.** Leave the cell empty. A missing
   field is fine; a plausible fabricated one silently corrupts the calibration. In
   particular, do not back-compute `doubles_attempted` from a checkout percentage — a
   percentage without its denominator carries no weight. The 188 rows that have one got it
   as the sum of two separately published counts.
2. **Do not re-scrape the upstreams.** They are commercial feeds (Sportradar, Flashscore)
   reached via already-published GitHub repos. Adding a new source means checking its
   robots.txt and terms first, and recording the result in `PROVENANCE.md` either way.
3. **Do not unpin the SHAs in `scripts/build_real_data.py`.** Floating to HEAD makes the
   calibration inputs silently mutable.
4. **The six validation gates are assertions, not logging.** If one fires the build stops.
   Fix the cause; do not downgrade the gate. They are what stands between this and
   quietly-wrong data.

## Open

- **No amateur or club-level data exists publicly, at any granularity.** Every player here
  is an elite professional, so the pub end of the pub-to-elite span is unsupported and any
  `Σ` for a club player remains an assumption. Club scoring lives inside proprietary apps
  (DartConnect, n01); DartConnect's league infrastructure holds exactly this and would have
  to be asked. This is the biggest substantive gap.
- **dartsorakel.com** has the by-double breakdown at far greater scale and recency. Left
  alone deliberately: commercial product, JS-rendered, robots.txt unverifiable. Asking them
  is the highest-value next step — they publish research blogs and may cooperate.
- Nothing links the four eras (2017, 2019, 2022, 2025) beyond player names.
