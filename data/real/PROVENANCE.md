# PROVENANCE — real darts scoring data

> **The CSVs described here are not committed to this repository.** They are
> regenerated from pinned open upstreams by `scripts/build_real_data.py`, which takes
> about a minute and verifies its output against `data/real/CHECKSUMS.txt`:
>
> ```
> pip install openpyxl pyreadr
> python scripts/build_real_data.py
> ```
>
> Two reasons for rebuilding rather than committing: the files total ~35 MB and would
> sit in git history permanently, and none of the four upstream repositories carries a
> licence, so regenerating avoids redistributing their contents. Every upstream is
> pinned to a commit SHA and the rebuild is byte-for-byte deterministic.
>
> The validation gates recorded below run as assertions on every rebuild — if an
> invariant ever breaks the script fails rather than emitting quietly-wrong data.
>
> **Archival risk.** This trades disk for a dependency on four third-party GitHub
> repos staying up. They are unmaintained (last pushed 2019–2025). If that risk is
> unacceptable, `git add -f data/real/target_outcomes.csv data/real/double_attempts.csv
> data/real/match_aggregates.csv` pins the three small, highest-value files for 110 KB.

Compiled 2026-08-20. All figures are as published by the cited sources. **Nothing in these
files is invented, interpolated, estimated, or back-computed from a percentage.** Where a
field in your requested schema could not be sourced, the cell is empty.

Two derivations are performed, both purely mechanical and both flagged per-row:
1. `score_before` for the 2022 source is **reconstructed** by replaying the dart sequence
   (column `score_before_origin`). See validation below.
2. `per_visit.csv` is **derived by aggregation** from `per_dart.csv`. No independent
   per-visit source was used.

---

## Headline answer on availability

**Per-dart data (Schema A) IS publicly available** — contrary to what you might expect.
300,985 darts are in `per_dart.csv`. This is the main finding of this exercise. Both
sources ultimately originate from professional live-scoring feeds that log every
individual dart, and both have been republished on GitHub by third parties.

Per-visit data (Schema B) has no separate public source I could find; it is derived from
per-dart data, which is strictly better anyway.

The scale target (~10,000 darts each for a handful of named professionals) is **partially**
met: 3 players clear 10,000 darts, 20 clear 5,000, 64 clear 1,000. See coverage table.

**No amateur or club-level data was found.** See "Gaps" at the end.

---

## FILE: `per_dart.csv` — Schema A

300,985 rows. Columns: `source, player, match_id, leg_id, visit_index, dart_index,
score_before, bed, value, raw_segment, raw_ring, score_before_origin, post_bust_visit`.

`bed` uses your vocabulary: `T20`, `S5`, `D16`, `25` (outer bull), `BULL` (inner bull, 50),
`MISS`. `value` is the points scored by that dart. `raw_segment` / `raw_ring` preserve the
source's own encoding so you can re-derive `bed` yourself if you disagree with my mapping.

### Source 1 — `dartsviz_pdc_2017` (254,108 darts)

- **Source:** GitHub repo `dmorgan26/dartsviz`, file `plot-app/data/throws.rda`
- **URL:** https://github.com/dmorgan26/dartsviz
- **Retrieved:** 2026-08-20
- **Coverage:** 2017 PDC season. 9,208 legs, 682 matches, 100 named players.
- **`score_before` is supplied natively by the source** — not reconstructed.
- **Field definitions (from the repo's own R code, not my assumption):**
  `create_double_success_per_match_dataset.R` computes a winning shot as
  `ring_id == 2 & score_before == segment*2`, confirming `score = segment * ring_id`,
  `ring_id` 1/2/3 = single/double/treble, `segment` = the bed number, `segment == 50` = bull.
- **Validation performed:** for all 167,022 consecutive within-visit dart pairs,
  `score_before[n+1] == score_before[n] − segment[n]*ring_id[n]`. **100.0000% consistent,
  zero exceptions.** This is a strong internal check that the field semantics above are correct.
- **`visit_index` is global within the leg**, alternating between the two players
  (player A visits 1,3,5…; player B visits 2,4,6…). It is *not* a per-player counter.
  This differs from a naive reading of your schema — worth handling explicitly.

### Source 2 — `sportradar_pdc_wc_2022` (46,877 darts)

- **Source:** GitHub repo `wonderkiduk/darts_data`, directory `data/`
- **URL:** https://github.com/wonderkiduk/darts_data
- **Retrieved:** 2026-08-20
- **Upstream:** the repo's `darts_scraper.py` shows the data came from Sportradar's PDC
  widget feed (`lmt.fn.sportradar.com/pdcsrcardiff/.../gismo/match_timeline/{id}`),
  taking events of type `single_throw_dart`. That is an official live-scoring feed.
- **Coverage:** PDC World Championship 2022 (played Dec 2021 – Jan 2022). 81 matches,
  1,556 legs. The repo contains 92 matches; **11 were excluded** — see below.
- **`score_before` is RECONSTRUCTED by me**, because the source CSVs contain only the
  ordered dart list (`team, points, segment, event_points, double_attempt`) with no
  running score. Reconstruction replays each leg from 501 applying standard bust rules
  (bust if remaining < 0, remaining == 1, or remaining == 0 without a double/inner bull;
  on bust the score reverts to the start of the visit).
- **Validation performed:** all 1,556 retained legs start at exactly 501 and terminate at
  exactly 0 on a double or inner bull. A desynchronised reconstruction cannot satisfy this
  invariant across thousands of legs, so I regard the retained legs as sound.
- **11 matches excluded** because reconstruction did not close cleanly (darts left over
  after the final checkout, i.e. the replay desynchronised). Rather than patch them I
  dropped them entirely. Excluded: Hunt v Krcmar, Wattimena v Koltsov, Murnan v Lim,
  Bunting v R. Smith, Price v Huybrechts, Labanauskas v De Decker, Hempel v R. Smith,
  Noppert v Heaver, Anderson v Cross, Wright v Rydz, Rodriguez v Robb. I do not know the
  cause; most likely missing events in the feed for those matches.
- **Segment encoding** (decoded from the data and confirmed by the arithmetic
  `points = event_points × multiplier`): `3`=treble, `2`=double, `10`=inner single
  (between bull and treble), `11`=outer single (between treble and double), `5`=inner bull
  (50), `4`=outer bull (25), `0` and `6`=no score. Note this source **distinguishes inner
  from outer singles**, which is finer than a `S20` label and is directly informative for a
  2-D landing model. That distinction is preserved in `raw_segment` but collapsed in `bed`.
- **Player names are parsed from the filenames** (`..._<home>_v_<away>.csv`) and
  title-cased, so they are lower-fidelity than Source 1 (e.g. "Michael Van Gerwen").
  Match the two sources on names with care.

### 42 rows where `bed` and `value` disagree (2022 source)

42 darts are recorded with a named bed (`S1`, `T20`, `D18`…) but `points = 0` — bounce-outs
or voided darts where the feed logged the segment anyway. `value` is authoritative: the leg
replay used it, and the 501→0 invariant holds because of it. Treat `bed` as unreliable on
these rows only; filter with `value == 0 and bed != 'MISS'`. A build gate asserts the count.

### `post_bust_visit` flag — read this before modelling

The 2017 feed **continues subtracting after a bust rather than reverting the score**. A
player on 5 who hits S10 is recorded with `score_before = −5` for the remaining darts of
that visit. This affects 56 visits / 148 darts (0.05% of the file). I have **not** deleted
or corrected these — every dart in an affected visit carries `post_bust_visit = 1`.
Filter them out if you want a clean state variable. All rows with
`post_bust_visit = 0` have `2 ≤ score_before ≤ 501`.

### Known gaps in this file
- `match_id` is empty for 39,871 rows (2017 source): those `leg_id`s are present in
  `throws.rda` but absent from `legs.rda`, so no leg→match mapping exists. Left empty.
- No dates, events, or opponents are attached to per-dart rows in either source.
- **No aiming/intended-target information exists in either source.** You observe where the
  dart landed and what the player's remaining score was; you do not observe what they aimed
  at. For scoring visits this is usually inferable (T20), for finishing visits much less so.

---

## FILE: `per_visit.csv` — Schema B (DERIVED)

103,158 rows. Columns: `source, player, match_id, leg_id, visit_index, score_before,
visit_score, darts_used, checkout, bust, post_bust_visit`.

- **Derived by aggregating `per_dart.csv`.** Not an independent source.
- `score_before` is the first dart's `score_before` in the visit — the field you flagged as
  essential, retained throughout.
- `visit_score` is the **raw arithmetic sum of the darts thrown**, including on busted
  visits. On a bust the points do not actually come off the score. I chose the raw sum
  rather than 0 because it is the observable, and added an explicit `bust` column so you can
  apply whichever convention your model wants. 161 busts flagged.
- `darts_used` is 3 except on winning visits (2,443 one-dart and 3,603 two-dart visits).
- Sanity: `visit_score` ranges 0–180; 5,018 maximums (180s); 10,238 checkouts.

---

## FILE: `match_aggregates.csv` — Schema C

188 rows (one per player per match), 2025 PDC World Darts Championship.

- **Source:** GitHub repo `henlewis/Darts-Web-Scraping-PowerBi-Dashboard`, file `Darts Data.xlsx`
- **URL:** https://github.com/henlewis/Darts-Web-Scraping-PowerBi-Dashboard
- **Retrieved:** 2026-08-20
- **Upstream:** the repo's README states the figures were scraped from **Flashscore**.
- **`doubles_attempted` IS PRESENT AND IS REAL.** The source publishes
  `successful_checkouts` and `failed_checkouts` as two separate integer counts.
  `doubles_attempted = successful + failed`. This is a sum of two published counts, **not**
  a back-computation from the percentage.
- **Validation performed:** for all 188 rows, `successful / (successful + failed)` reproduces
  the separately-published `checkout_percentage` to within 0.02pp. **Zero disagreements.**
  The published percentage is retained in its own column so you can check this yourself.
- **Empty fields:** `legs_played`, `darts_thrown`, `points_scored`, `first9_average` are not
  published by this source and are left empty. `legs_won`, `sets_won`, `sets_lost` and
  `stage` are supplied as extras.
- **Definitional caveat — UNRESOLVED.** The source does not state whether
  `average_3_darts` includes the checkout visit, nor whether it is per-leg or per-match.
  From the Flashscore convention it is a **per-match** average, and standard PDC practice
  (see Wikipedia, "Three-dart average") is total points ÷ darts thrown × 3, counting a busted
  visit as three darts and counting only the darts actually thrown on a winning visit — which
  **includes** the checkout visit. I could not confirm this for Flashscore specifically, so
  treat it as probable, not established. If it matters to your calibration, note that you can
  compute averages yourself from `per_dart.csv` under whatever definition you prefer, which is
  the more reliable route.

---

## FILE: `double_attempts.csv` — the by-double breakdown you asked for

321 rows. Columns: `source, player, double, attempts, hits`.

## FILE: `target_outcomes.csv` — the same data, unaggregated (richer)

1,489 rows. Columns: `source, player, target, outcome_bed, count`. For each player and each
*intended target region*, the full distribution of what was actually hit. This is a superset
of `double_attempts.csv` and is the single most useful file here for fitting a 2-D landing
model, because it is the only data in this delivery where **the intended target is known**.

- **Source:** GitHub repo `wangchunsem/OptimalDarts`, file `Raw_Data.xlsx`
- **URL:** https://github.com/wangchunsem/OptimalDarts
- **Retrieved:** 2026-08-20
- **Paper:** Haugh & Wang, *An Empirical Bayes Approach for Estimating Skill Models for
  Professional Darts Players*, arXiv:2302.10750v2 (Apr 2024), CC BY 4.0. Section 3 documents
  the data. The repo is linked from the paper as the official data release.
- **Coverage:** 2019 season, the then-top-16 professionals, by name. Targets present:
  T20, T19, T18, T17, all of D1–D20, and the inner bull. Only these are included because,
  as the paper notes, other regions are targeted too rarely.
- **Definitions:** a row is (target region aimed at, bed actually hit, count of darts).
  `MISS` = landed outside any scoring region. Outcome beds are limited to the target and its
  board-adjacent neighbours, which is why e.g. a T20 attempt can only resolve to
  T20/S20/T5/S5/T1/S1.
- **How the intended target is known:** the paper does not say how the target was
  determined, and this is the main thing I would treat with suspicion. It is presumably
  inferred from game state by whoever collected it, not observed. The paper is explicit that
  it knows the target *region* but **not the aim point within the region** (their §3
  "Limitations"), which is exactly the nuisance parameter your Gaussian will have to absorb.
- **Validation performed** — I reproduced the paper's published summary statistics from my
  extraction, as an end-to-end check:

  | quantity | paper | my extraction |
  |---|---|---|
  | T20 attempts / success | 117,600 / 41.2% | 117,600 / 41.2% ✓ |
  | T19 attempts / success | 27,709 / 41.7% | 27,709 / 41.7% ✓ |
  | T18 attempts / success | 7,717 / 36.9% | 7,717 / 36.9% ✓ |
  | T17 attempts / success | 2,461 / 33.5% | 2,461 / 33.5% ✓ |
  | total attempts, 21 double regions | 16,777 | 16,777 ✓ |
  | odd doubles D1–D19 total | 1,866 | 1,866 ✓ |
  | overall double success | 40.2% | 40.2% ✓ |
  | van Gerwen T20 / T17 | 45.3% / 30.2% | 45.3% / 30.2% ✓ |
  | **D20 attempts** | **4,399** | **4,339** ✗ |

- **DISCREPANCY — flagged, not corrected.** The paper states 4,399 attempts at D20; the
  workbook gives 4,339. Every other published figure matches exactly, including the grand
  total of 16,777 — and 16,777 is only consistent with **4,339** (using 4,399 would give
  16,837). I therefore believe the paper contains a digit transposition and the workbook is
  right. I have left the workbook figure as-is and not adjusted anything.

---

## Sources checked and NOT used

- **dartsorakel.com** — has exactly the per-double attempt/hit breakdown you want, at far
  greater scale and recency (their blog cites e.g. "202 attempts at Double 18" for Van den
  Bergh). **Not scraped.** It is a commercial data product ("No.1 collector and provider of
  data", sells subscriptions); the substantive pages are JavaScript-rendered and return no
  data to a plain fetch; and I could not retrieve and verify their robots.txt. Under your
  rule I stopped. **This is the highest-value next step and is best pursued by asking them
  for access** — they publish research blogs and may well cooperate with a modelling project.
- **Flashscore** — the upstream of `match_aggregates.csv`. Not accessed by me; I used a
  third party's already-published extract. Their terms very likely restrict automated access.
- **Sportradar PDC widget feed** — the upstream of the 2022 per-dart data. A commercial
  feed. Not accessed by me; again I used a published extract.
- **PDC official site (pdc.tv)** — publishes a checkout-success table by checkout value for
  the then-top-10 players (2020 article). Percentages, mostly without denominators. Not
  incorporated; low marginal value next to `target_outcomes.csv`.

### Redistribution status — please read

None of `dmorgan26/dartsviz`, `wonderkiduk/darts_data`, or
`henlewis/Darts-Web-Scraping-PowerBi-Dashboard` carries a licence file. The *paper* behind
`OptimalDarts` is CC BY 4.0 but the repo itself has no licence. The underlying figures
originate from commercial feeds (Sportradar, Flashscore). Factual sports scores are thin
copyright material in most jurisdictions, but **this data is fine for private research and
should not be republished** without checking. I did not scrape any of these upstreams; I
used publicly published GitHub repositories.

---

## Coverage — darts per player (`per_dart.csv`)

3 players ≥10,000 darts · 20 players ≥5,000 · 64 players ≥1,000 · 158 named players total.

| darts | player |
|---:|---|
| 18,376 | Peter Wright |
| 12,996 | Michael van Gerwen |
| 11,088 | Daryl Gurney |
| 9,863 | Raymond van Barneveld |
| 9,752 | Gary Anderson |
| 9,216 | Michael Smith |
| 8,091 | Phil Taylor |
| 8,022 | Simon Whitlock |
| 7,966 | James Wade |
| 7,939 | Gerwyn Price |
| 7,709 | Mensur Suljovic |
| 7,646 | Dave Chisnall |
| 6,570 | Adrian Lewis |
| 6,222 | Jelle Klaasen |
| 5,644 | Joe Cullen |
| 5,475 | Rob Cross |
| 5,474 | Kim Huybrechts |
| 5,453 | Alan Norris |
| 5,232 | Mervyn King |
| 5,085 | Ian White |

Note the two per-dart sources are five years apart (2017 and 2021/22) and player skill is
not stationary over that gap. Wright's 18,376 is 16,285 darts from 2017 plus 2,091 from
2022; pooling them treats a 2017 Wright and a 2022 Wright as the same thrower. Splitting by
`source` is available in every row.

---

## Gaps and things I could not get

1. **No amateur or club-level data at any granularity.** I found none in public form. Club
   scoring is captured almost entirely inside proprietary phone apps (DartConnect, n01,
   Dart Tracker, DartVision — several of which advertise per-dart heat-map recording), and
   none of them publish an open dataset. If the pub-to-elite span matters to your model,
   the realistic routes are (a) asking DartConnect, whose league infrastructure holds
   enormous amounts of exactly this, or (b) collecting it yourself with a per-dart scoring
   app. Every player in this delivery is an elite professional. This is the biggest
   substantive gap.
2. **`first9_average`, `darts_thrown`, `points_scored`, `legs_played`** are absent from the
   aggregate source. `darts_thrown` and `points_scored` are computable exactly from
   `per_dart.csv` for the 2017/2022 matches, but those are different tournaments from the
   2025 aggregates, so no join is possible.
3. **Nothing links the three eras.** 2017 per-dart, 2019 target-outcomes, 2022 per-dart and
   2025 aggregates are four disjoint datasets sharing only player names.
4. **No aim points anywhere.** Only `target_outcomes.csv` knows the intended *region*, and
   even there the aim point within the region is unknown — the paper's authors treat this as
   a latent variable and fit it via EM.
