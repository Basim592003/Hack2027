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
- [ ] **B2 Noise v5** in `src/noise.py`.
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
- [ ] **B3 Like-for-like.**
  - Promote `src/diag/old_vs_new.py` to `experiment crossbench --train v4 --test v5`, and run the reverse too.
  - Keep v5 if its model beats v4's on v5 for both sources and loses ≤ 0.01 on v4. The reverse test is the real one,
    since σ = 0 is inside v5's range.
  - Then re-record the baseline on v5 and regenerate the submission.

## C. Hand-over evidence (target: Thu 12 Nov)
- [ ] **C1 Event list.**
  - New `src/segments.py`, `events(...)`: the sorted union of kicks, exclusive-close-team changes, raw ball-gap edges
    and predicted stoppage edges, merged within 3 frames. Kinds are flags; the team is the kick's, else the close
    change's, and −1 if they disagree.
  - Have `touches.touch_context` also return the kick frames and teams and the close-change frames, so nothing is
    computed twice.
  - Port it from `src/diag/bench_gaps_candidates.py`. Record recall at ±6 and ±12 frames and events per minute on
    v5, plus BAR-ATH's events per minute as a sanity check.
- [ ] **C2 Event-gated switch cost.**
  - `decode.event_costs(frames, n, low, high, k)`.
  - Grid: low {2, 4, 6} × high {6, 10, 20} × k {6, 12}. Decode games in parallel with `ProcessPoolExecutor`, passing
    flags as explicit arguments, since macOS workers re-import modules with their defaults.
  - Keep it only if it beats a constant cost of 6 by ≥ 0.005 on both sources, with neighbouring cells close.
  - Caveat: ±12-frame windows cover about 40% of live frames, so gating is only weakly selective. BAR-ATH has fewer
    kicks and could end up with too few possessions; compare its events per minute and possession count (sanity only).
  - If kept, `run.predict_bar` builds BAR-ATH's events and passes `team_costs`. The default `None` keeps the current
    submission byte-identical.
- [ ] **C3 Pooled features (code only).**
  - `segments.pool(values, events)`: means over the frame's own, previous and next segments, by cumulative sums
    (as in `features.rolling_mean`).
  - In `src/possession.py`, behind a `SEGMENTS` flag, `view_features` pools per view the signals it already has:
    `nearest`, `control`, `d_diff`, and `comove_opp − comove_own`. Add the contrasts with the neighbouring segments,
    seen share, log segment length, and position in the segment. Event kinds are unsigned; the event team goes
    through `signed()`.
  - Always emit every feature name, NaN-filled, so sorted names match across games and BAR-ATH.
  - Checks:
    - signed features are exact negatives across the two views;
    - features are identical with `labels=None` (no leak).
- [ ] **C4 Ceiling run** (diagnostic, never shipped). Pass the true controlled turnover frames as
  `build_tables(games, extra_events=...)`. If the gain is < 0.02, drop C5.
- [ ] **C5 Pooled features, real run**, plus an ablation without kick events. Apply the gate. Compare segment-length
  distributions with BAR-ATH's (aggregates only), and drop `seg_log_len` if they diverge.
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
