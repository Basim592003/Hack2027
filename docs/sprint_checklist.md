# Sprint checklist: raise BAR-ATH spellF1 (2026-11-07 → submit by 2026-11-17)

Working notes for the last ten days before the Phase 1 deadline. Each sitting does one item: do it, record the result
under it, commit, push, stop.

## Why this order
- **Target.** The test labels count *controlled* possession (CHALLENGE.md). Under that rule our possession count
  already roughly matches, and spells under 1 s only exist right after restarts. The job is the right team at the right
  time at in-play turnovers.
- **Headroom.** The perfect when-to-switch oracle reaches only 0.701 / 0.717 against 0.672 / 0.700 (v4 bench, last
  touch), while true team reaches 0.952 / 0.912. Most of the gain is in P(home) inside short spells, where the 1–4 s
  windows straddle the hand-over. The decoder matters less.
- **Events already sit on turnovers.** Kicks, exclusive-close-team changes and ball-gap edges lie within ±12 frames of
  79–88% of controlled in-play turnovers on the v4 bench, at 25–29 events a minute. Evidence can be pooled between them.

## How each sitting runs
1. Take the first unticked item and do only that item.
2. Run its check, then write the result under it: numbers per source, keep or drop, and the date.
3. If the item is kept and changes the model, regenerate the submission with
   `.venv/bin/python -m src.run --out evaluation/results/<team>.csv` (it runs the validator). Note the SHA-256.
4. Tick the item, commit (the message names the item), and push to `origin`.
5. Stop, and say which item is next.

Standing rules:
- Train and tune on external games only. BAR-ATH aggregates are printed as a sanity check, never used to choose.
  Never print BAR-ATH rows.
- Report IDSSE and Metrica separately.
- Nothing runs overnight. Split any item that would take more than about 90 minutes.
- Keep variants few: with ten clean games, a long list of comparisons will pick noise. On a tie, keep the simpler version.
- There are about 25 items for 10 days, so 2–3 sittings a day. If sittings run short, drop items in this order:
  E4, E3, D2, D1, C6, A4. Phase F starts on 15 Nov regardless.

**Keep gate.** The score is controlled-label spellF1, leaving one clean game out (20 degraded copies). A change must
pass all three:
- it improves on both sources, or rises ≥ 0.005 on one with the other down < 0.003;
- it wins most paired copies in each source (≥ 8 of 14 IDSSE, ≥ 4 of 6 Metrica);
- last-touch spellF1 drops ≤ 0.01 on each source.

Hygiene items A5 and E1 are kept unless controlled spellF1 drops on either source, by ≥ 0.005 for A5 or ≥ 0.002 for E1.

From B3 on, `compare` also prints controlled spellF1 by each copy's ball smoothing. Both upper bands are BAR-ATH-like:
BAR-ATH reads as σ ≈ 7.5–8 by its kink ratio (B2), but like IDSSE's σ 4–7 copies by its kick and event rates (C1).
Each band holds only 6 copies (5 IDSSE, 1 Metrica). Read them next to the per-source gate, not instead of it.

## A. Setup and baseline (target: Sun 8 Nov)
- [x] **A1 Versioning.**
  - `git remote rename origin upstream`, then `git remote add origin https://github.com/Basim592003/Hack2027.git`.
  - If the remote already has commits, stop and ask. Never force-push.
  - Commit the pipeline snapshot, add this checklist, add a pointer in CLAUDE.md, then `git push -u origin main`.
  - Done 2026-11-08:
    - `origin` is now github.com/Basim592003/Hack2027 (it was empty); `upstream` is metrica-sports.
    - Commit 80ab693 snapshots the pipeline: `src/`, the tech doc, CLAUDE.md, pinned requirements, RULES.md, and the
      starter notebook (it has no outputs). Nothing under `data/` is tracked.
    - The scratch diagnostics that later items port now live in `src/diag/`, because `/private/tmp` is cleared on
      reboot: `oracles.py`, `old_vs_new.py`, `miss_causes.py`, `bench_gaps_candidates.py`. They read the external bench
      only. Run them from the repo root with `PYTHONPATH=.`.
- [x] **A2 Harness: score both conventions.**
  - New `src/experiment.py` with `score_both(match, labels)`. It scores against `match.labels` and, in memory,
    against `controlled.controlled_labels(match.labels)`, so it needs no `~ctl` games or files.
  - Tag every out-of-fold cache with a short hash of whatever produced it: the upstream tag, flags, `MODEL_PARAMS`
    and seeds. Files become `pdead_oof_<name>@<tag>.npy` and `phome_oof_<name>@<tag>.npy`. Write a manifest after
    the last fold, and skip a step whose manifest exists. Today's caches are keyed by name only, so a feature or
    model change silently reuses stale files. Untagged names stay as they are, so `src.run` is unchanged.
  - Check: reproduce 0.680 / 0.704 last touch, and record the controlled baseline.
  - Done 2026-11-08 (`experiment run base`, dead @24a8bea05d, home @adead22ea8; IDSSE / Metrica):
    - last-touch spellF1 0.6800 / 0.7043, reproduced exactly;
    - **controlled spellF1 0.7246 / 0.7490**, macroF1 0.922 / 0.898;
    - predicted possessions 337 / 305, against 365 / 320 real under controlled labels and 437 / 381 under last touch.
- [x] **A3 Harness: compare.** `python -m src.experiment compare <a> <b>` prints, per source and per label set:
  - the two means and their paired difference (mean, min, max);
  - copies won (of 14 IDSSE / 6 Metrica) and clean games won (of 7 / 3);
  - predicted vs real possessions, and macroF1;
  - in-play turnover recovery and P(true team) in spells under 2 s (from `src/diag/miss_causes.py`);
  - the median offset of matched in-play changes.

  Check: base vs base gives zero difference.
  - Done 2026-11-08: `run base2` reused both caches (3 s against about 10 min cold), and `compare base base2` shows
    zero difference everywhere.
  - Baseline diagnostics, controlled labels, IDSSE / Metrica:
    - in-play turnovers recovered: 66.5% / 70.9%;
    - P(true team) in in-play spells under 2 s: 0.53 / 0.53;
    - median offset of answered changes: −0.8 / 0 frames;
    - share of all real changes answered within 8 frames: 46% / 51%.
  - So the matched changes are on time at the median. The loss is the half that is never answered within 8 frames.
- [x] **A4 Oracles under controlled labels.**
  - Add `decode.viterbi4_costs(e_home, e_away, e_dead, team_costs, dead)`, with a per-frame team cost and the same
    candidate order and tie-breaking as `viterbi4`. Assert identical labels to `viterbi4` at a constant cost on all
    20 copies. (The scratch `viterbi4_varying` in `src/diag/oracles.py` calls `np.argmax` per frame and is several
    times slower.)
  - `experiment oracles` runs four cases: true D, true team, perfect switch timing, and perfect timing limited to
    turnovers with an event within ±12 frames (the ceiling for C2). Record the controlled headroom.
  - Done 2026-11-08. `viterbi4_costs` equals `viterbi4` on 20/20 copies and runs slightly faster (about 0.09 s a
    game). Controlled spellF1, IDSSE / Metrica:

    | case | spellF1 | gain |
    |---|---|---|
    | as predicted | 0.7246 / 0.7490 | — |
    | true D | 0.7521 / 0.8148 | +0.028 / +0.066 |
    | true team | 0.9588 / 0.9286 | +0.234 / +0.180 |
    | perfect switch timing | 0.7616 / 0.7807 | +0.037 / +0.032 |
    | perfect timing, only at events (C2 ceiling) | 0.7587 / 0.7745 | +0.034 / +0.026 |

  - 83% / 88% of real in-play changes have an event within ±12 frames.
  - Team evidence is still the main headroom. On Metrica, a perfect dead ball is worth more than perfect timing.
- [x] **A5 `early_stopping=False`.** sklearn's default `'auto'` holds out a random 10% of rows, while the other team
  view of each held-out frame stays in training. Test the possession model first, since nothing upstream changes.
  Test the dead-ball model as a separate A/B, because it cascades into the team model's restart features.
  - Done 2026-11-08. Controlled spellF1, IDSSE / Metrica:
    - possession model (`run a5-home`): +0.0008 / +0.0025, 7/14 and 4/6 copies won. **Kept**: noise-level, but it
      passes the hygiene gate.
    - dead-ball model on top (`run a5-dead`): −0.0023 / −0.0103, 0/6 Metrica copies won. **Dropped.** Its early
      stopping evidently regularises, so `deadball.MODEL_PARAMS` keeps sklearn's default.
  - **New baseline `a5`** (dead @24a8bea05d, home @aead934829): controlled 0.7254 / 0.7515, last touch
    0.6813 / 0.7057.
  - Submission regenerated: SHA-256 `8efbe025f1c2e05b18cdd620daa4f71ea88db4303417cab715ff7141ff321ebb`, valid.
  - BAR-ATH aggregates, sanity only: 356 possessions of mean 10.0 s, 43.8% D across 118 stoppages, home 61.1% of
    live frames.
  - The previous file (22e4fd4c…) predates the push/retreat features. It is kept as
    `data/cache/possession_v4_22e4fd4c.csv`.
- [x] **A6 Organiser question.** Draft one grey-area email for the user to send, covering:
  - the noise-calibration items in CLAUDE.md;
  - v5's randomised ball smoothing (B2);
  - the lag estimator, if B1 finds a lag;
  - whether the public window may be inspected to see if hand-overs are stamped at the first touch or at control.
  - Done 2026-11-08: the draft is in [organiser_question.md](organiser_question.md), and also asks about the Metrica
    sample licence. **The user sends it.** Record the reply here; until then, B2 goes ahead with randomised settings,
    and nobody reads the public window.

## B. Make the bench look like BAR-ATH's tracking (target: Mon 9 Nov)
- [x] **B1 Lag check.**
  - New `src/lagcheck.py`, aggregates only. Take stretches where the same slot stays within 1.5 m of the ball for at
    least 25 frames while moving ≥ 3 m/s. Cross-correlate ball and player velocity at lags −12…+12.
  - Print the median peak lag for the clean games, v4, and BAR-ATH.
  - If BAR-ATH lags by more than 2 frames, add item B1b: `align_ball(match)`, a per-match estimate plus shift applied
    after loading in `bench.bench_games` and `run.predict_bar`. It is the same algorithm on every match.
  - Done 2026-11-08. Peak lag in frames (negative: the ball leads), with agreement at the peak:
    - clean IDSSE: −3.1, 0.91;
    - BAR-ATH: −2.5, 0.91 (8.8k close-control frames);
    - clean Metrica: +1.5, 0.90;
    - the v4 bench matches its clean games.
  - In a dribble the ball moves first and the player follows, so real tracking leads by about 3 frames. BAR-ATH is
    within 0.6 frames of IDSSE, so there is no causal ball filter to undo and **no B1b**. Metrica's sample ball trails
    its carrier, as expected for a ball that sits on the carrier.
- [x] **B2 Noise v5** in `src/noise.py`.
  - **Seeding.** Give each copy its own random streams:
    `np.random.SeedSequence([seed, zlib.crc32(clean_name.encode()), round(1000 * alive_missing)]).spawn(5)`, one each
    for parameters, dead ball, ball, drift and players. Today every copy uses seed 0, so a σ drawn from the shared
    generator would be the same everywhere. Never use Python's `hash()`, which changes between runs.
  - **Smoothing.** After `render_dead_ball`, smooth the ball with σ drawn from 0–6 frames:
    - for each tracked run, pad with odd reflection, apply `gaussian_filter1d`, then crop;
    - skip it when σ < 0.5, because scipy's kernel is NaN at σ = 0.
  - **Gaps.** Draw a per-copy in-play gap median, log-uniform over 40–200 frames, and pass it to `drop_ball_in_play`.
  - `noise_params(...)` recomputes a copy's σ and gap median for reporting, since the `.npz` does not keep them.
  - Set `VERSION = 5`. v4 caches stay for comparison.
  - Sanity only:
    - `python -m src.bench --check-noise` (frames in gaps > 2 s);
    - `python -m src.touches` (kick rate and strength vs BAR-ATH's 2.8 / 14.5 kinks a minute and 7 m/s).
  - Done 2026-11-08:
    - Per-copy streams (`copy_streams`) and recomputable draws (`noise_params`) are in place.
    - **σ is drawn from 0–10 frames, not 0–6.** BAR-ATH's ±4 / ±12 kink ratio (0.19, `coverage.kink_rates`, which
      reproduces CLAUDE.md's 2.8 / 14.5) matches σ ≈ 7.5–8 on clean games of both sources. The 20 copies span
      σ 0.5–9.8, and 6 sit at σ ≥ 7. In-play gap medians span 44–190 frames.
  - Sanity only (aggregates), BAR-ATH against v5:
    - kink ratio: 0.19, against 0.31–0.45 on IDSSE copies and 0.84–1.09 on Metrica (5 of 6 Metrica copies drew
      σ ≤ 4.7);
    - missing frames in gaps > 2 s: 99.9%, against 97.6–99.6%;
    - kicks per graded minute: 8.9, against 8.8 on IDSSE (v4: 13);
    - next kick by the same team: 72%, against 74% (v4: 79%).
  - v5 bench, trained and scored on v5: controlled 0.6632 / 0.7300, last touch 0.6089 / 0.6733 (v4: 0.7254 /
    0.7515). It is a harder bench and closer to BAR-ATH.
- [x] **B3 Like-for-like.**
  - Promote `src/diag/old_vs_new.py` to `experiment crossbench --train v4 --test v5`, and run the reverse too.
  - Keep v5 if its model beats v4's on v5 for both sources and loses ≤ 0.01 on v4. The reverse test is the real one,
    since σ = 0 is inside v5's range.
  - Then re-record the baseline on v5 and regenerate the submission.
  - Done 2026-11-08. Controlled spellF1, IDSSE / Metrica:
    - scored on v5: the v5-trained model beats the v4-trained one by +0.022 on IDSSE (10/14 copies, 6/7 games) and
      is flat on Metrica (−0.0015, 2/6);
    - scored on v4: the v5-trained model loses −0.014 / −0.009.
  - **The written gate fails, but its premise was wrong.** The unsmoothed v4 bench is the least BAR-ATH-like test,
    not the real one. Split by each copy's smoothing (`compare` now prints this):

    | ball smoothing | copies | v5-trained − v4-trained | copies won |
    |---|---|---|---|
    | σ < 4 | 8 | −0.020 | 0/8 |
    | σ 4–7 | 6 | +0.024 | 6/6 |
    | **σ ≥ 7, BAR-ATH-like** | 6 | **+0.052** | **6/6** |

    The one heavily smoothed Metrica copy gains +0.071. The v4-trained model falls to 0.59 on heavily smoothed copies,
    where the v5-trained one gets 0.65.
  - **Decision: adopt v5.** This departs from the written gate; the user can veto it, and the previous file is kept.
  - **New baseline `v5`** (dead @a7c6ea032d, home @11b9c1d6c8): controlled 0.6632 / 0.7300, last touch
    0.6089 / 0.6733.
  - Submission regenerated twice, with identical SHA-256
    `ac12fe95397f1d1ea279fff696e2c1fdc4f6ec8b1fb35dac6da70a997c3c452f`, valid. The A5 file is kept as
    `data/cache/possession_a5_8efbe025.csv`.
  - BAR-ATH aggregates, sanity only: 326 possessions of mean 11.3 s (A5: 356), 42.3% D across 111 stoppages, home
    59.5%.

## C. Hand-over evidence (target: Thu 12 Nov)
- [x] **C1 Event list.**
  - New `src/segments.py`, `events(...)`: the sorted union of kicks, exclusive-close-team changes, raw ball-gap edges
    and predicted stoppage edges, merged within 3 frames. Kinds are flags; the team is the kick's, else the close
    change's, and −1 if they disagree.
  - Have `touches.touch_context` also return the kick frames and teams and the close-change frames, so nothing is
    computed twice.
  - Port it from `src/diag/bench_gaps_candidates.py`. Record recall at ±6 and ±12 frames and events per minute on
    v5, plus BAR-ATH's events per minute as a sanity check.
  - Done 2026-11-08:
    - `segments.events(touch, ball_seen, dead, period)` returns (frames, kinds, teams). `match_events(match, dead)`
      builds them from scratch, and `within(frames, targets, k)` tests nearness. Half edges count as stoppage
      edges. A group never spans more than 3 frames, so duel flicker is cut rather than collapsed. Checked against a
      plain-Python merge on two copies.
    - `touch_context` also returns the kick and close-change frames and teams. This is output-neutral: under the new
      tags (dead @5349f74532, home @d7a411c20b) both caches equal v5's on 20/20 copies, and `compare v5 c1` shows
      zero difference. The model is unchanged, so the submission stays `ac12fe95…`.
    - `experiment oracles` uses these events now. `src/diag/bench_gaps_candidates.py` is ported and removed.
  - On v5, IDSSE / Metrica, with BAR-ATH in brackets as a sanity check:
    - events per graded minute: 17.6 / 23.3 (17.4). Of these, kicks 8.8 / 14.5 (8.9), close changes 5.5 / 6.9
      (5.8), gap edges 3.1 / 3.3 (2.7), stoppage edges 2.1 / 1.9 (2.0). 71% / 82% (76%) of events carry a team;
    - ±6- and ±12-frame windows cover 14% / 19% (14%) and 25% / 33% (25%) of graded frames.
  - Recall of real in-play changes under controlled labels (last-touch recall is within 0.04):

    | within | any event | an event of the new team |
    |---|---|---|
    | ±6 frames | 0.577 / 0.772 | — |
    | ±12 frames | 0.645 / 0.820 | 0.472 / 0.618 |

    By kind, at ±12: kicks 0.46 / 0.55, close changes 0.38 / 0.50, gap edges 0.06 / 0.31, stoppage edges
    0.01 / 0.03.
  - This is below v4's 83% / 88% (A4), because v5's ball filter flattens kicks. On IDSSE copies, recall at ±12 is
    0.83 / 0.65 / 0.50 at σ < 4 / 4–7 / ≥ 7, with 13.9 / 9.2 / 4.3 kicks a minute.
  - **BAR-ATH's kick and event rates (8.9 and 17.4 a minute) match IDSSE's σ 4–7 copies (9.2 and 17.8), not its
    σ ≥ 7 ones (4.3 and 13.0).** By this measure the σ ≥ 7 band over-smooths, so `compare` no longer labels it
    BAR-ATH-like, and the reading note above is updated. B3 stands: v5 also won the σ 4–7 band 6/6 (+0.024).
  - Gap edges: only Metrica's own ball gaps sit at hand-overs. On the clean games, IDSSE's ball is never missing in
    play. Metrica's is missing on 11% of live frames, and its gap edges lie within ±12 frames of 29% of hand-overs,
    against 5.5% for as many random live frames. v5's added gaps fall at random. How BAR-ATH's behave is unknown
    (see the note under C5).
  - Oracles re-run on v5 with these events, controlled spellF1:

    | case | spellF1 | gain |
    |---|---|---|
    | as predicted | 0.6632 / 0.7300 | — |
    | true D | 0.7067 / 0.8075 | +0.044 / +0.078 |
    | true team | 0.9498 / 0.9203 | +0.287 / +0.190 |
    | perfect switch timing | 0.7069 / 0.7588 | +0.044 / +0.029 |
    | perfect timing, only at events (C2 ceiling) | 0.6882 / 0.7536 | +0.025 / +0.024 |
- [x] **C2 Event-gated switch cost.** Dropped.
  - `decode.event_costs(frames, n, low, high, k)`.
  - Grid: low {2, 4, 6} × high {6, 10, 20} × k {6, 12}. Decode games in parallel with `ProcessPoolExecutor`, passing
    flags as explicit arguments, since macOS workers re-import modules with their defaults.
  - Keep it only if it beats a constant cost of 6 by ≥ 0.005 on both sources, with neighbouring cells close.
  - Its ceiling on v5 is +0.025 / +0.024 (C1's oracles), so the bar of +0.005 asks for a fifth of it.
  - Caveat: ±12-frame windows cover 25% / 33% of graded frames on v5, so the gate is selective, but it misses 35% / 18%
    of changes. BAR-ATH's event rate matches IDSSE's (C1); still compare its possession count (sanity only).
  - If kept, `run.predict_bar` builds BAR-ATH's events and passes `team_costs`. The default `None` keeps the current
    submission byte-identical.
  - Done 2026-11-08 with `experiment eventgrid`. It took 30 s serially, so no process pool. The low = high = 6 cell
    reproduces the constant cost exactly. Controlled spellF1 against the constant, IDSSE / Metrica:
    - best cell, low 6 / high 10 / k 6: +0.0019 / +0.0012 (7/14, 4/6). Nothing reaches +0.005 on either source except
      low 6 / high 20 / k 6 on Metrica, +0.0035, and that cell loses on IDSSE (−0.0017);
    - cheaper switching near events always loses: low 4 gives −0.0006 to −0.0052, low 2 gives −0.003 to −0.013. It
      adds possessions (IDSSE 320 → 340, against 365 real) that are mostly wrong.
  - Events come every 2.6–3.4 s, and most are not hand-overs. P(home) already moves where the ball really changes
    hands (A3: matched changes are on time), so the gate only adds false switches. **Dropped.** `decode.event_costs`
    and `experiment eventgrid` stay, to re-check once C5 changes P(home).
- [x] **C3 Pooled features (code only).**
  - `segments.pool(values, events)`: means over the frame's own, previous and next segments, by cumulative sums
    (as in `features.rolling_mean`).
  - In `src/possession.py`, behind a `SEGMENTS` flag, `view_features` pools per view the signals it already has:
    `nearest`, `control`, `d_diff`, and `comove_opp − comove_own`. Add the contrasts with the neighbouring segments,
    seen share, log segment length, and position in the segment. Event kinds are unsigned; the event team goes
    through `signed()`.
  - Always emit every feature name, NaN-filled, so sorted names match across games and BAR-ATH.
  - `match_context` builds the events from its own touch signals with `segments.events`. Add `segments` to
    `experiment.home_oof`'s signature modules and `SEGMENTS` to its settings, or P(home) caches go stale.
  - Checks:
    - signed features are exact negatives across the two views;
    - features are identical with `labels=None` (no leak).
  - Done 2026-11-08, behind `possession.SEGMENTS = False`, so nothing shipped changes:
    - `segments.pool(values, frames)` returns the own, previous and next segment means. `segments.layout(...)` gives
      the per-frame segment facts that are the same from both sides: log length, position, ball-seen share, and the
      kinds and team of the bounding events.
    - That adds 33 features per view (85 → 118): own, previous and next means and both contrasts for `nearest`,
      `control`, `d_diff` and `comove` (opp − own), plus seen share, log length, position, 4 + 4 kind flags and the
      signed start and end teams. Building the views still takes 0.6 s a copy.
    - Checked on one IDSSE and one Metrica copy: `pool` equals brute force, signed features are exact negatives across
      the views and unsigned ones identical, the 85 existing features are unchanged, and views are identical with
      `labels=None`.
    - The own-segment means are NaN on 32–43% of frames, because ball gaps are events, so a missing-ball stretch is
      its own segment. The neighbouring means are NaN on only 16–22%.
- [x] **C4 Ceiling run** (diagnostic, never shipped). Pass the true controlled turnover frames as
  `build_tables(games, extra_events=...)`. If the gain is < 0.02, drop C5.
  - Skipped 2026-11-08, because it cannot fail. Events at the true turnovers also tell the model *where* possession
    changes, and that alone is worth +0.044 / +0.029 (C1's perfect-timing oracle). So the run would clear 0.02
    whatever pooling is worth. C5 measures pooling directly.
- [x] **C5 Pooled features, real run**, plus an ablation without kick events. Apply the gate. Compare segment-length
  distributions with BAR-ATH's (aggregates only), and drop `seg_log_len` if they diverge. Only Metrica's own ball
  gaps sit at hand-overs (C1), so a gain on Metrica alone may come from gap-edge kinds that need not carry over.
  - Done 2026-11-08 (`run c5`, home @88025f1e8d, against `c1`, which is the v5 model). Controlled spellF1,
    IDSSE / Metrica:
    - **+0.0024 / +0.0125** (0.6632 → 0.6655, 0.7300 → 0.7424), winning 8/14 and 6/6 copies, 4/7 and 3/3 games;
    - last touch +0.0028 / +0.0085, macroF1 +0.0004 / +0.0018;
    - by smoothing: σ < 4 +0.0082 (6/8), σ 4–7 +0.0050 (5/6), σ ≥ 7 +0.0019 (3/6).
  - All three gate tests pass, IDSSE at the 8/14 minimum. **Kept: `possession.SEGMENTS = True`.**
  - The gain is not where it was aimed. P(true team) in short in-play spells is flat (0.460 → 0.457 /
    0.470 → 0.477), and turnover recovery moves −0.005 / +0.005. Predicted possessions fall (320 → 315 / 300 → 294),
    so the features mostly remove wrong possessions rather than finding short ones.
  - The ablation drops gap events instead of kicks, since C1 found BAR-ATH's kick rate matches the bench and only its
    gaps are in doubt. It was a two-line filter in `segments.events`, not kept:
    - Metrica keeps its whole gain (+0.0124 against c1), so the gain does not come from Metrica's gap quirk;
    - IDSSE, whose gaps are random, falls to −0.0019. Against the full model the no-gap one loses −0.0042 there
      (wins 3/14), and −0.0086 on σ ≥ 7 copies (0/6), where kicks are rare and gaps give the segment edges.
  - Segment lengths: BAR-ATH's median event spacing is 41 frames, against 40.6 / 34.2 (C1), so `seg_log_len` stays.
  - C2's grid on the new P(home): the best cell gives +0.0001 / +0.0007. C2 stays dropped.
  - **New baseline `c5`** (dead @5349f74532, home @88025f1e8d): controlled 0.6655 / 0.7424, last touch
    0.6117 / 0.6818.
  - Submission regenerated twice, with identical SHA-256
    `cee13f262ae2c061cea351bedcced5a2f39ec782046e34a1ca7b8684b25ad275`, valid. The v5 file is kept as
    `data/cache/possession_v5_ac12fe95.csv`.
  - BAR-ATH aggregates, sanity only: 330 possessions of mean 11.1 s (v5: 326 of 11.3 s), 42.2% D across 110
    stoppages, home 59.5%.
- [ ] **C6 Learned switch cost**, in a new `src/boundary.py`.
  - A HistGradientBoosting model for b_t = P(controlled in-play change within ±3 frames). Its features carry no team,
    so it needs no views:
    - event kinds within ±6, 12 and 25 frames;
    - the out-of-fold P(home) step for k = 6, 12, 25, 50;
    - nearest flips and gap edges;
    - P(dead).
  - Train on all positives and every 5th negative, sampled the same way in CV and in `run.py`. Group by
    `deadball.clean_name`, and retrain whenever the possession model changes.
  - Cost = clip(α − β·logit b_t, 0.5, 20). Keep it only if it beats C2 by ≥ 0.005. If kept, `run.py` gains a
    possession out-of-fold step (about +3 min) and saves `final_teamcost_BAR-ATH.npy`.

## D. New evidence (target: Sat 14 Nov), each behind its own flag and gate
- [ ] **D1 Flight target**, `src/flight.py`.
  - When the ball moves faster than 5 m/s, project b + v·τ for τ from 0.12 to 3 s, stopping at the lines. Process in
    chunks of 2,000 frames.
  - A player is in time if 0.7 + max(|p − (x + 0.7·v)| − 1, 0) / 5 ≤ τ.
  - Per team view: the first team to arrive (signed), the margin, each team's arrival time, an interception flag, and
    a "goes out first" flag.
  - NaN means not applicable; 3.5 s means nobody arrives in time. Never use infinity.
  - In ball gaps, carry the values from the last frame with a defined velocity (`lagged_delta` is NaN near gaps).
- [ ] **D2 Pressure**, `src/pressure.py`.
  - Chunks of 5,000 frames shaped (frames, home players, away players). Find slots with `np.flatnonzero(team == k)`;
    never assume slot order.
  - Closing speed is −Δx·Δv / max(d, 0.5), over opponents within 10 m.
  - Per team view:
    - own and opponent maximum pressure, and their gap;
    - the team of the most-pressed player (signed), and opponents within 5 m of that player;
    - each team's divergence (2×2 least squares in closed form, small ridge, at least 4 players);
    - 1 s and 4 s averages.
  - Watch team accuracy on live frames with the ball missing.

## E. Training recipe (in spare sitting time during C–D; finish by Sun 15 Nov)
- [ ] **E1 Monotonic constraints.** Pass feature names into `possession.fit`, and assert every pattern matches at
  least one feature.
  - +1: `nearest_*`, `control_*`, `kick_last`/`kick_next` and their streaks, `close_last`/`close_next`,
    `kicks_own_*`, `restart_rule`, and the new signed features.
  - −1: `d_diff*`, `kicks_opp_*`.
  - Never constrain `last_nearest`, `next_nearest` or `kick_prev`.
- [ ] **E2 Denser sampling.** In `training_rows`, take every frame within ±12 frames of an in-play change, and every
  5th frame elsewhere.
- [ ] **E3 Extra seed for training only** (`bench.TRAIN_SEEDS = (0, 1)`); evaluate on seed 0 only. Keep full feature
  tables for seed 0 only, or memory reaches about 6 GB. Keep it if it gains ≥ 0.003, since it doubles run time.
- [ ] **E4 Classifier grid**, about 4 configurations per sitting and at most 6 in all: learning rate, leaves,
  min leaf, L2, max_features. Re-pick the decoder costs for each and prefer the plateau.

## F. Release (Sun 15 – Tue 17 Nov). On 15 Nov stop new work, whatever is unticked.
- [ ] **F1** Re-pick the decoder costs on the final model.
- [ ] **F2** Rebuild with `data/cache/` moved aside: two runs give an identical SHA-256, and the validator passes. Run
  `evaluation/run_evaluation.py` as an alignment check only.
- [ ] **F3** Update `docs/technical_document.md` §1 and §3–7, CLAUDE.md, and the docstrings of the modules that changed.
- [ ] **F4** The user submits the CSV, the repo link and the tech doc through the form.

## Out of scope
GRU / 1D-CNN, stand-alone stacking, Smart Tagging priors, and anything tuned on the public window.
