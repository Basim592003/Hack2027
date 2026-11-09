# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A clone of the organisers' starter repo for **More Than A Hack 2027, Phase 1** (Metrica Sports / FC Barcelona). The task is to label every frame of the tactical video of BAR-ATH (Barcelona = `H`, Athletic = `A`) as `H`/`A`/`D`/`X`. Submissions are ranked on **spellF1**, with macroF1 as the tie-break. The deadline is 2026-11-17.

Everything outside `src/` belongs to the organisers. Our pipeline lives in `src/`. The organiser docs (`CHALLENGE.md`, `DATA.md`, `EVALUATION.md`, `RULES.md`, `FAQ.md`, `SUBMISSION.md`) are authoritative. `evaluation/possession_metrics.py` is the metric specification, and where it disagrees with a doc, the code wins.

**Current work, until the deadline: `docs/sprint_checklist.md`.** One item per sitting: do the first unticked item, write its result under it, commit, push to `origin`, and stop. `origin` is our repo (github.com/Basim592003/Hack2027); `upstream` is the organisers' repo, which we only pull from.

## Hard rules (competition terms + RULES.md)

The organisers' clarification email (top of `RULES.md`, recorded 2026-11-03) **overrides everything else**, including older docs and FAQ answers. In short:
- **Training and tuning use external data only.** That means every trained model and every tuned parameter, never the ground truth or the video.
- **Self-supervised or pseudo-label training on BAR-ATH is now out**, even though the original RULES.md allowed it.
- **Never check output against the video.**
- **Allowed:** automatic checks using only the provided data, inspecting the data and video to understand what counts as possession, providing intermediate-step files, and any dataset whose licence permits it.

The rest of this section still applies:

- **Never put raw match data into the conversation.** The terms forbid giving the videos, datasets "or any extracts thereof" to a cloud LLM, and that includes Claude. Do not `cat`, `head` or print rows of `data/tactical/*`, `data/broadcast/*` or `evaluation/gt/*`, and do not open the video. Write code that runs locally and prints **aggregates only**: counts, shares, percentiles, scores. Reading metadata structure, such as XML tag names or field counts, is fine. External games in `data/external/` are not competition data and may be inspected.
- **No hand labels on BAR-ATH**, in any amount or for any purpose. This includes training, validation and threshold tuning. Correcting the tracking by hand also counts as hand-labelling.
- **The public 5 minutes are not a tuning set.** They cover video frames 11185–18684 and 17 possessions. Use them only to check alignment, file format and label meaning. Never use them to choose between two versions; the docs show that this picks the wrong one.
- **No human-made data about BAR-ATH from any source**, such as Opta, StatsBomb or corrected tracking.
- **Every constant and model weight comes from football reasoning or from other games**, meaning the external benchmark below. Do not derive constants from specific frames of BAR-ATH.
- **Grey area, not yet asked of the organisers:** `src/noise.py` takes some simulator settings from aggregate statistics of BAR-ATH's tracking and Smart Tagging:
  - ball missing 40.7%;
  - 20.5 players tracked and 11.3 tracklet starts a minute;
  - `DEAD_MISSING = 0.75` from the restart profile.

  These are not the ground truth or the video, and they shape training data rather than tune the algorithm. But "calibrating" in the email could be read broadly. The options are to ask the organisers, or to train across a range of noise settings so no single BAR-ATH-derived value matters.
- **Same grey area, added 2026-11-06:**
  - noise v3 makes the position error drift smoothly because BAR-ATH's tracking is smooth frame to frame;
  - `touches.KINK_LAG = 12` was picked over 4 after seeing that BAR-ATH's smoothed ball shows almost no kinks at 4 frames.

  Both are aggregates of BAR-ATH's tracking input, never labels or video. Include them in the question to the organisers.
- **Same grey area, added 2026-11-07:** noise v4 (a moving dead ball) was prompted by aggregates of BAR-ATH's tracking around Smart Tagging set pieces. In the 5 s before a tagged set piece, the seen ball is still on only 4% of frames, and the v3 pipeline called D on 0–1% of them. v4's durations and speeds are round football numbers, not fitted to those aggregates. That check is a sanity check only; never use it to choose between versions.
- **Licences:** IDSSE is CC BY 4.0. The Metrica sample repo has **no formal licence**; its README asks only for responsible use and attribution. Confirm with the organisers, who are Metrica themselves, before relying on it.
- **The repo must regenerate the submission byte-for-byte.** `data/` stays gitignored, and the final repo is public.
- **The tech doc must list every labelled dataset used:**
  - the Metrica sample data (credit Metrica Sports);
  - IDSSE: Bassek et al. 2025, *Scientific Data*, doi:10.1038/s41597-025-04505-y, CC BY 4.0.

## Environment and commands

Use `.venv/bin/python`. It is Python 3.9.6 with numpy, pandas, kloppy 3.18, scikit-learn 1.6.1 and scipy. The system `python3` lacks kloppy and scikit-learn.

```bash
# the submission: one command, deterministic (~4 min from an empty cache)
.venv/bin/python -m src.fetch_metrica                       # once: Metrica sample, pinned to commit e706dd5
.venv/bin/python -m src.fetch_idsse                         # once: IDSSE via the figshare API, MD5-checked
.venv/bin/python -m src.run --out evaluation/results/<team>.csv   # trains on external games, writes + validates

# organisers' tooling
.venv/bin/python evaluation/run_evaluation.py              # scores every evaluation/results/*.csv vs the public 5 min
.venv/bin/python evaluation/run_evaluation.py --bootstrap 1000
.venv/bin/python evaluation/validate_submission.py <file>.csv   # run on the exact file before submitting
.venv/bin/python starter/baseline.py                       # regenerates the 4 baseline CSVs

# our pipeline (run from the repo root, as modules)
.venv/bin/python -m src.fetch_idsse          # downloads the 7 IDSSE games via the figshare API (~2.5 GB, MD5-checked)
.venv/bin/python -m src.external             # converts the Metrica sample games, prints label stats
.venv/bin/python -m src.idsse                # converts the IDSSE games, prints label stats
.venv/bin/python -m src.coverage             # tracking-quality aggregates, BAR-ATH vs external
.venv/bin/python -m src.bench                # the starter baseline's method scored on the external bench
.venv/bin/python -m src.bench --check-noise  # degraded games' coverage next to BAR-ATH's
.venv/bin/python -m src.deadball             # dead-ball detector, leave-one-game-out CV (~2–3 min, run in background)
.venv/bin/python -m src.deadball --transfer  # train on one source, test on the other
.venv/bin/python -m src.deadball --bar 40    # train on all external games, apply to BAR-ATH, aggregates only
.venv/bin/python -m src.possession           # team model, leave-one-game-out (~2 min; needs deadball's OOF cache)
.venv/bin/python -m src.possession --bar 6   # train on all external games, apply to BAR-ATH, aggregates only
.venv/bin/python -m src.possession --no-touches   # the same without the touch features, for comparison
.venv/bin/python -m src.touches              # kick agreement with external labels; kick rates on BAR-ATH (aggregates)
.venv/bin/python -m src.segments             # event list: recall of real in-play changes on the bench; rates on BAR-ATH (aggregates)
.venv/bin/python -m src.decode               # joint decoder cost grid on the bench (needs both OOF caches)

# A/B harness (src/experiment.py): every keep/drop decision since 2026-11-08
.venv/bin/python -m src.experiment run NAME       # tagged OOF caches + joint decode, scored on last-touch AND controlled labels (~15 min cold since E3, seconds cached)
.venv/bin/python -m src.experiment compare A B    # B vs A per source: means, paired diffs, copies/games won, diagnostics
.venv/bin/python -m src.experiment oracles        # headroom: true D, true team, perfect switch timing (also at events only)
.venv/bin/python -m src.experiment eventgrid      # switch cost cheap near events vs constant (C2: dropped, ~30 s)
.venv/bin/python -m src.experiment boundarygrid   # learned switch cost, src/boundary.py, vs constant (C6: dropped, ~1 min)
.venv/bin/python -m src.experiment costgrid NAME  # constant team / dead costs on a saved run's predictions (~3 min); re-pick after any P(home) change
# The OOF tags hash the source of features.py, touches.py, deadball.py, possession.py and (P(home) only) segments.py:
# any edit to them, even a docstring, recomputes the OOF steps (~10 min) on the next run. Batch such edits first.
```

There is no unit-test suite. A change is verified in three ways:
1. The external benchmark (`src.bench` / `src.deadball`).
2. `validate_submission.py`.
3. Re-running and diffing the CSV to confirm the output is deterministic.

To clone the Metrica sample data again: `git clone --depth 1 https://github.com/metrica-sports/sample-data.git data/external/metrica-sample`.

## Architecture

**Organiser code.**
- `games.py` holds per-match constants and asset paths. `half_video_blocks("BAR-ATH")` returns the graded halves.
- `evaluation/possession_metrics.py` holds the metrics; `evaluate(gt_codes, pred_codes)` computes everything.
- `starter/baseline.py` is the nearest-player baseline plus a majority vote. `smooth()` and `write_submission()` are reused from there.

**Our pipeline: one shared container, many sources.**
- `src/match.py`: `Match` holds positions in metres with a corner origin.
  - `px, py`: shape (T, P); `bx, by`: shape (T,); `team` per slot.
  - `period`: 1 or 2 inside a half, 0 elsewhere.
  - `labels`: possession_metrics codes `H=0 A=1 D=2 X=3`, or `None` for BAR-ATH.
  - `cached(name, build)` stores matches as `data/cache/<name>.npz`. It is keyed **by name only**, so after changing a loader, delete the relevant `.npz` files.
- Loaders, each producing a `Match`:
  - `src/epts.py` parses FIFA/EPTS metadata and raw files. It handles several `DataFormatSpecification` blocks (Metrica game 3 starts a new block at every substitution). BAR-ATH has one.
  - `src/sources.py` loads BAR-ATH, unlabelled. The ATD has one record per video frame, and `video_frame = frame_id - 1`.
  - `src/external.py` loads Metrica sample games 1–3 and derives labels from their events with `labels_from_events`:
    - possession = the team of the latest on-ball event;
    - challenges don't move possession;
    - D runs from the event that stopped play until the next SET PIECE;
    - since 2026-11-09 (sprint item D3), `unannotated_stoppages` also marks D where the events say live but the ball is missing for ≥ 4 s and no event of any type starts. That covers 31k frames, about 7 min a game, in which the players walk as in a stoppage. The sample blanks its ball through stoppages, but its labels only close one at a SET PIECE. Metrica's D share goes from 25–33% to 32–41%.
  - `src/idsse.py` loads the 7 DFL games. Labels come from the per-frame `BallStatus` and `BallPossession` flags; referees are dropped, and the halves are joined by 250 `X` frames.
- `src/noise.py`: `degrade()` makes clean labelled games look like the BAR-ATH ATD:
  - drops players (mean 20.5 tracked, about 11 tracklet starts a minute) and re-slots them into a 46-slot pool;
  - adds 0.15 m of position error to players and ball. Since v3 this error **drifts smoothly** (Gaussian low-pass, sd 12 frames). v2's independent per-frame jitter made the bench's second differences about 0.4 m, against 0.001 m on BAR-ATH, so velocity signals were noise on the bench and clean on BAR-ATH;
  - re-renders the ball during stoppages, gone in about 75% of them with edges blurred by about ±1 s;
  - since v4, renders the whole path of the ball through every stoppage (`dead_ball_path`). It runs on and slows, is fetched or a new ball appears, is carried by the restart taker (the player nearest the restart spot when play resumes), and is set on the spot. Neither source has a real dead ball to copy: IDSSE freezes it, Metrica blanks it;
  - adds live-ball dropout as a parameter.
  - since v5 (2026-11-08, sprint item B2):
    - each copy has its own random streams (`copy_streams`, keyed on the game, dropout level and seed); up to v4 every copy shared seed 0;
    - the ball goes through a zero-phase Gaussian of σ drawn from 0–10 frames (`smooth_ball`). BAR-ATH's ratio of ±4- to ±12-frame kinks (0.19, `coverage.kink_rates`) matches σ ≈ 7.5–8 on clean games of both sources;
    - in-play ball gaps get a median drawn from 40–200 frames;
    - `noise_params` recomputes a copy's draws, and `experiment compare` uses them to report results by smoothing band.
  - v6 (2026-11-09, sprint item D3), the bench in use, has no simulator change. The copies are rebuilt because the Metrica labels now mark unannotated stoppages, and the dead-ball rendering and in-play dropout follow the labels. Trained on v6 rather than v5 and scored on v6, controlled spellF1 gains +0.005 on IDSSE, whose copies and labels are unchanged, and +0.009 on Metrica.
  - BAR-ATH has no ball gaps of 1–4 s at all, so its tracker evidently fills short losses, while v5/v6 draw many short in-play gaps. On the bench these are decoded correctly (1–2% called D), so the mismatch costs little.
  - Cached names carry `noise.VERSION`; **bump it after any simulator change.**
- `src/bench.py`: `bench_games()` returns 10 clean games × live-dropout levels (0.10, 0.20) = 20 degraded games. `run(predict)` scores any `Match -> codes` function with the official scorer.
  - **Always report results per source, never pooled**, because the sources' label conventions differ.
- `src/features.py`: `frame_features(match)` gives 96 per-frame signals, computed from positions only so they mean the same thing on every source. Since v4 these include `out_not_restarted` (the ball was seen more than 1 m beyond a line after the last kick) and `nearest_speed` (how fast the player nearest the ball moves). Speeds are displacements over windows (about 1 s for players), not frame differences. `team_sides()` tells which goal each team defends in each half.
- `src/deadball.py`: P(dead) from a `HistGradientBoostingClassifier` trained on the external games only, then `viterbi2` (2-state, switch penalty) → stoppages.
  - Out-of-fold probabilities are cached as `data/cache/pdead_oof_<game>.npy` and BAR-ATH's as `pdead_BAR-ATH.npy`, so step 4 can be evaluated honestly.
- `src/possession.py`: P(home has the ball) per live frame, from a classifier trained on external games only. It uses three feature families:
  - **ball:** nearest player per team, players within 1.5 m, whether the nearest player moves with the ball, and their past/next 1–4 s averages;
  - **team shape:** height up the pitch, length, width, forward speed, and the team nearest the ball when it was last and next seen;
  - **restart:** the corner / goal-kick rule, plus the team nearest the ball at the restart;
  - **touches** (`src/touches.py`, since 2026-11-06): the team of the last and next *kick*, kicks in a row by one team, the team of the kick before the last, kicks per team in ±2 s, and the last/next team alone within 1.5 m. A kick is a peak in the change of ball velocity over ±12 frames (≥ 4 m/s) with a player within 2 m.
  - **segments** (`src/segments.py`, `SEGMENTS`, since 2026-11-08, sprint items C3–C5): `nearest`, `control`, `d_diff` and `comove` pooled over the stretch between the events around each frame and the stretches either side. Also the contrasts between them, the stretch's length, the frame's position in it, the seen share, and the bounding events' kinds and team. It added +0.002 / +0.012 controlled spellF1 (IDSSE / Metrica) on the v5 bench, mostly by removing wrong possessions.

  Features are computed from each team's point of view, with the pitch oriented so "own" attacks +x. Training uses both views; inference averages them, so the output is symmetric and cannot learn "home usually has it".
  - `decode_teams` runs a 2-state Viterbi on each live stretch from step 2's D. Each stretch starts free, and a switch costs `penalty`.
  - Requires step 2's out-of-fold cache. Writes `phome_oof_<game>.npy` and `phome_BAR-ATH.npy`.
- `src/segments.py` (since 2026-11-08, sprint item C1): the events where the ball can change hands. `events(touch, ball_seen, dead, period)` returns (frames, kinds, teams) for kicks, changes of the team alone within 1.5 m, raw ball-gap edges, and predicted stoppage or half edges, merged within 3 frames. Kinds are bit flags. The team is the one the kicks and close changes agree on, else −1.
  - `touches.touch_context` returns the kick and close-change frames these are built from, so nothing is computed twice.
  - `match_events(match, dead)` builds them from scratch, and `within(frames, targets, k)` tests nearness.
  - `experiment oracles` uses them.

### Build order (plan)

1. External loaders, noise simulator and benchmark, with the baseline as reference. **Done.**
2. Per-frame features and the dead-ball detector. **Done.**
3. Possession evidence. **Done** (`src/possession.py`); results are below.
   - controlled-touch logic: possession changes only on a *controlled* touch by the other team, i.e. about 1 s on the ball or 2+ consecutive same-team touches;
   - a team-shape model for frames without a ball, **trained on the external games only** (the 2026-11-03 email rules out pseudo-labels on BAR-ATH);
   - restart-team evidence (see the findings below);
   - Smart Tagging phase priors, used as soft evidence only.
4. Joint decoder. **Done** (`src/decode.py`). One Viterbi pass over {H, A, D after H, D after A} with team cost 10 and dead cost 40; restart memory and boundary snapping are off. The team cost was 6 until 2026-11-09. Sprint item E2 (`possession.DENSE`: also train on every live frame within ±12 frames of a change of team) sharpened P(home) at hand-overs and moved the plateau to 10–12, found with `experiment costgrid`. It matches steps 2+3 run in sequence, and both variants were tested and dropped (results below). Per-frame switch costs were also dropped in sprint items C2 and C6 (2026-11-08/09). A cost cut near events and a cost learned from a change detector (AUC 0.88 / 0.94) both lose or tie. Gains have to come from P(home).
5. Final run. **Done**: `src/run.py`, `requirements-pipeline.txt` (pinned, Python 3.9.6), and the draft `docs/technical_document.md`.
   - The output passes `validate_submission.py`.
   - A rebuild from an empty `data/cache/` was byte-identical (SHA-256 `e5a8c7b9…19264`; after step 6, `f4c9fc60…f422`).
   - Public-window alignment check: spellF1 0.484, 21 possessions against 17 real. Pipeline check only; never use it to choose anything.

   Any change to `src/` changes the submission, so re-run `src.run`, the validator and the determinism diff, and update the tech doc's numbers.
6. Noise v3 (smooth position error) and touch features in the team model. **Done 2026-11-06**, giving bench spellF1 0.681 / 0.721 (findings below).
7. Noise v4 (a moving dead ball) plus the out-of-play latch and nearest-player speed in the dead-ball model. **Done 2026-11-07** (findings below, "Visible dead ball").
   - BAR-ATH now gets 368 possessions of mean 9.7 s, 43.8% D across 118 stoppages, and home 60.9% of live time.
   - The rebuild from an empty cache is byte-identical (`22e4fd4c…4c35b4d`, ~5 min). The previous CSV is kept as `data/cache/possession_noise_v3_f4c9fc60.csv`.
   - BAR-ATH now gets 416 possessions of mean 9.8 s, 35.8% D across 125 stoppages, and home 57.0% of live time.
   - The rebuild from an empty cache is byte-identical (`f4c9fc60…f422`, ~5 min). The previous CSV is `e5a8c7b9…`.

## What the metric rewards

- A possession is a maximal H or A run. `H H D D H H` is **two** possessions: a missed stoppage merges them, and a spurious D splits them.
- Predicted and true runs are matched one-to-one, **same team only**, by IoU, and F1 is averaged over IoU 0.3–0.7. A run with the wrong team scores nothing, even when its timing is perfect.
- Timing: a 1-frame lag is free, 3 frames gives 0.98, 8 frames 0.89, and 1 s late 0.65.
- `X` is forward-filled within a half before spellF1 is computed. So on spellF1 an `X` means "repeat the last label": free inside a possession, costly across a hand-over. macroF1 counts every graded `X` as a miss.
- spellF1 differences below about 0.04 are noise on the full match.

## Data facts

- **Frames:** 199,319 at 25 fps. Graded: the 1st half is frames 11185–90651, the 2nd half 110808–190197.
  - The 2nd half kicks off at **113742**, so its first 2,934 graded frames are all `D`.
  - Smart Tagging times are seconds from the start of the video: `round(s * 25)` gives the frame.
- **ATD:** only `x, y, speed` per slot and `x, y, speed` for the ball. There is **no ball-state or owner field**, so both must be inferred.
  - 46 slots are tracklets, not people.
  - Coordinates are normalised 0–1; the pitch is 105.3 × 68.0 m per `games.py`.
  - The ball can be tracked off the pitch: x runs from −7 to 112 m, y up to 91 m.
- **Smart Tagging:** 638 instances. `Team` is `N/A` on 36, including every throw-in. The intervals are phases, not possessions.

## Measured findings (2026-11-01/02)

All BAR-ATH numbers below are aggregates of input data or of our own predictions, never of the ground truth.

**Tracking quality** (`src.coverage`):

| | ball missing | gaps/min | median gap | players tracked (mean / p05) | tracklet starts/min |
|---|---|---|---|---|---|
| BAR-ATH | 40.7% | 1.4 | 268 frames (~11 s) | 20.5 / 14 | 11.3 |
| Metrica sample 1–3 | 32–41% | 1.6–1.8 | 124–182 frames | 22 / 22 | ~0 |

**The ball goes missing in long gaps, not brief flickers**, and 99% of missing frames are in gaps longer than 2 s. On BAR-ATH, players' mean speed is **2.10 m/s with the ball tracked versus 1.14 m/s without**, so a missing ball often coincides with a stoppage.

**Ball missing versus dead ball:**
- In the Metrica sample: P(dead | missing) = 0.76–0.80, P(missing | dead) = 0.91–0.99, P(missing | live) = 0.09–0.14.
- **Leak:** the Metrica sample tracking blanks the ball for exactly every dead stretch, with edges within about 1 frame. IDSSE never loses the ball, but **freezes it exactly where play stopped on every dead frame** (zero movement in all 7 games), then jumps a median 6 m (p90 26 m) to the restart spot on the first live frame. That is a second leak; see "Visible dead ball" below.

Share of frames with the ball missing, by seconds from a restart:

| seconds from restart | −10 | −3 | −1 | 0 | +1 | +3 | +10 |
|---|---|---|---|---|---|---|---|
| BAR-ATH, Smart Tagging set-piece start | 0.66 | 0.74 | 0.76 | 0.69 | 0.66 | 0.60 | 0.19 |
| Metrica, event set-piece start | 0.72 | 0.96 | 0.97 | 0.16 | 0.03 | 0.10 | 0.26 |

So on BAR-ATH a missing ball is a strong hint of a stoppage, not a giveaway, and it has no sharp restart edge. `noise.py` reproduces that with `DEAD_MISSING = 0.75` and `RESTART_EDGE_SD = 25`.

Live-ball dropout on BAR-ATH can't be measured without labels. A mixture bound puts it at about **10–25% of live frames**, which is why the bench uses the 0.10 and 0.20 levels. The 0.10 level best matches BAR-ATH's gap profile.

**Label conventions differ by source.** Keep them separate:

| source | possessions per match | mean possession | dead ball | stoppages |
|---|---|---|---|---|
| IDSSE (DFL flags) | 370–511 | 5.5–9.2 s | 41–52% | 92–113 |
| Metrica sample (event-derived) | 358–405 | ~10 s | 25–33% | 75–88 |
| BAR-ATH public 5 min (published) | 17 in 5 min | 14.2 s | 20% (first 5 min only) | — |

Metrica is likely the closer convention to BAR-ATH's ground truth, since both use the same vendor's tracking. A full match has about **350–400+ possessions**.

**Starter baseline** (nearest player, ±50-frame vote):

| | spellF1 |
|---|---|
| BAR-ATH full match (published) | 0.2433 |
| BAR-ATH public window (`smooth50` / `smooth125`) | 0.3256 / 0.4242 |
| Bench, IDSSE (live 0.10 / 0.20) | 0.300 / 0.277 |
| Bench, Metrica (live 0.10 / 0.20) | 0.350 / 0.330 |

Our BAR-ATH loader reproduces `baseline_smooth50.csv` exactly (199,319/199,319 frames). The baseline over-counts possessions: 492–555 predicted against 381–437 real on the bench, and 131 against 17 for `baseline_raw` on the public window.

**Dead-ball detector** (step 2, leave-one-game-out, switch penalty 40 chosen by CV). "End-to-end" combines our D with the baseline's team guess.

| | alive/dead F1 | dead-stretch F1 (IoU-matched) | stoppages predicted vs real | end-to-end spellF1 |
|---|---|---|---|---|
| IDSSE: baseline D | 0.844 | 0.592 | 91 vs 102 | 0.288 |
| IDSSE: detector | 0.986 | 0.959 | 99 vs 102 | 0.381 |
| Metrica: baseline D | 0.890 | 0.526 | 93 vs 80 | 0.340 |
| Metrica: detector | 0.965 | 0.880 | 80 vs 80 | 0.403 |

- Without the decoder (penalty 0) it finds about 190 stoppages instead of about 100, so the decoder is essential.
- **Cross-source:** trained on IDSSE and tested on Metrica, dead-stretch F1 is 0.81; trained on Metrica and tested on IDSSE, it is 0.86.
- **On BAR-ATH:** 37.6% D across 127 stoppages averaging 18.8 s. That is somewhat high against 80–102 per match on the external games, so some stoppages may be false ones caused by live-ball dropout.
- The remaining end-to-end gap comes from the team guess flickering (455 vs 437 and 471 vs 381 possessions), which is step 3's job.

**Team model** (step 3, leave-one-game-out, step 2's D at penalty 40, team switch cost 6 chosen by CV):

| | end-to-end spellF1 | macroF1 | team accuracy (live) | possessions predicted vs real |
|---|---|---|---|---|
| IDSSE: baseline team guess, same D | 0.381 | — | — | 455 vs 437 |
| IDSSE: **team model** | **0.616** | 0.919 | 0.889 | 305 vs 437 |
| Metrica: baseline team guess, same D | 0.403 | — | — | 471 vs 381 |
| Metrica: **team model** | **0.674** | 0.909 | 0.891 | 286 vs 381 |

- Switch costs 4–10 form a flat plateau, within 0.04 of each other. A cost of 0 gives about 950 possessions and spellF1 around 0.35.
- At the optimum it **under-counts** possessions, merging short real ones. That's a target for step 4 (duration modelling, boundary snapping).
- During live play with the ball missing, "team nearest when last seen" points the **opposite** way on average, because ball gaps cluster around stoppages and about 70% of stoppages hand the ball over. The model learns this; don't hard-code the sign.
- **On BAR-ATH:** 362 possessions, mean 11.0 s, home 57.8% of live time.

**Joint decoder** (step 4, out-of-fold probabilities):

| | spellF1 IDSSE | spellF1 Metrica |
|---|---|---|
| steps 2+3 in sequence | 0.616 | 0.674 |
| joint pass | 0.618 | 0.678 |
| + snapping team changes | 0.580 | 0.659 |
| + snapping stoppage edges | 0.615 | 0.677 |

Restart memory (keep 0.3 / switch 0.7) changes nothing, at most ±0.002.

**Where the misses are.** Real possessions matched at IoU 0.5, by length (IDSSE / Metrica):

| length | share of all | matched |
|---|---|---|
| < 1 s | 13% / 12% | 7% / 2% |
| 1–2 s | 17% / 14% | 24% / 21% |
| 2–4 s | 23% / 18% | 45% / 48% |
| 4–8 s | 20% / 17% | 73% / 80% |
| 8–15 s | 14% / 14% | 84% / 88% |
| > 15 s | 13% / 24% | 90–97% |

- The external conventions give possession to the last toucher, so deflections and clearances out of play create many sub-1 s "possessions". BAR-ATH's answer key counts only *controlled* possession (CHALLENGE.md), so those are probably rarer there.
- Further gains need better evidence for short possessions, not a better decoder.

**Controlled-possession labels** (`src/controlled.py`, `--labels controlled` on `src.possession` / `src.decode`; 2026-11-04):
- **The rule:** the restarting team owns the ball; after that, the other team takes over only with a spell of at least 1 s. The 1 s comes from the CHALLENGE.md definition, not a score. A longer threshold always makes the target easier, so a score can't choose it.
- **Effect on the labels:** it removes 15–25% of external possessions (Metrica 380 → 322, IDSSE 511 → 395) but changes only 0.2–0.7% of frames.
- **Retraining doesn't help.** Against controlled labels, the last-touch-trained model scores 0.680 / 0.741, against 0.676 / 0.733 retrained (IDSSE / Metrica), with the same best switch cost. The switch cost already implements the rule, so `run.py` keeps `CONVENTION = "last_touch"`.
- **The losses are in-play turnovers.** Possessions starting after a restart (25–27% of all) are recovered 83–84% of the time; those starting with a turnover in play (73–75%) only 57–67%. Controlled possessions of 2–4 s are recovered about 50% of the time. **The next lever is pass and interception evidence.**

**Noise v3 and touch features** (2026-11-06). These moved the bench more than anything since step 3.

Oracle swaps (v2 bench, joint decoder) show **the team decision is the bottleneck**, not dead ball:

| | spellF1 IDSSE | spellF1 Metrica |
|---|---|---|
| as predicted | 0.618 | 0.678 |
| true D, predicted team | 0.638 | 0.711 |
| predicted D, true team | 0.941 | 0.926 |

Team errors sit at turnovers. Share of live frames on the wrong team, by distance to the nearest real in-play team change (IDSSE / Metrica):

| distance | wrong |
|---|---|
| < 1 s | 34% / 28% |
| 1–3 s | 10% / 7% |
| 3–10 s | 2% / 2% |
| > 10 s | 1% / 1% |

In-play possessions under 2 s got a mean P(true team) of only 0.36–0.46: the 1–4 s averages blur the hand-over.

Tracking smoothness, which changed the simulator (aggregates of input only):

| | player 2nd diff, p50 | ball kinks/min over ±4 frames | ball kinks/min over ±12 frames | nearest player to ball, p50 |
|---|---|---|---|---|
| BAR-ATH | 0.001 m | 2.8 | 14.5 | 1.9 m |
| clean external | 0.002–0.01 m | 18 | 16 | 0.28 m Metrica, 2.0 m IDSSE |
| bench, v2 noise | 0.43 m | 16–24 | 15–18 | — |

- BAR-ATH's tracking is smooth, and its ball filter spreads a kick over about half a second.
- **Metrica's sample ball sits on the carrier** (median 0.28 m), so Metrica's proximity signals are unrealistically clean. IDSSE matches BAR-ATH's distance profile.

The kick detector (`src.touches`, ±12 frames, ≥ 4 m/s, player within 2 m):

| | IDSSE bench | Metrica bench | BAR-ATH |
|---|---|---|---|
| kicks per graded minute | 13 | 16 | 9 |
| kick team = label just after | 94% | 97% | — |
| in-play turnovers with a kick by the new team within 0.5 s | 52% | 46% | — |
| next kick by the same team (noise would give 50%) | 79% | 85% | 72% |
| graded frames within 1 s of a kick | 34% | 41% | 25% |
| median kick strength | 11 m/s | 12 m/s | 7 m/s |

Kick strength is therefore **not** a feature: it reads softer on BAR-ATH, and dropping it cost nothing on the bench.

Results (leave one game out, joint decoder, team cost 6, dead cost 40):

| | spellF1 IDSSE | spellF1 Metrica | live team accuracy | possessions predicted vs real |
|---|---|---|---|---|
| v2 noise, no touches (previous submission) | 0.618 | 0.678 | 0.889 / 0.891 | 304 / 284 vs 437 / 381 |
| v3 noise, no touches | 0.611 | 0.674 | 0.887 / 0.893 | 304 / 285 |
| **v3 noise + touches** | **0.681** | **0.721** | 0.905 / 0.905 | 331 / 290 |

- The dead-ball detector is unchanged by v3: dead-stretch F1 0.961 / 0.882, against 0.959 / 0.880.
- Team costs 4–10 are still a plateau.

**Who restarts after a stoppage** (937 stoppages, clean external games; restart type taken from where the ball is at the first live frame):

| restart type | other team restarts (IDSSE / Metrica) |
|---|---|
| corner | 0.84 / 1.00 |
| goal kick | 0.83 / (n=2) |
| throw-in | 0.79 / 0.95 |
| kick-off | 0.72 / 0.83 |
| free kick / other | 0.47 / 0.36 |
| all stoppages | 0.70 / 0.72 |

- **Corners are restarted by the team attacking that goal 96–100% of the time; goal kicks by the defending team 100%.** These follow from the Laws of the Game and are near-certain evidence, given `team_sides()` and where the ball reappears.
- When the same team has the ball before and after a stoppage (`H D H`), it was a free kick or foul in 53% (IDSSE) / 84% (Metrica) of cases, a throw-in in 32% / 9%, and a goal kick in 9% / 1%.
- **Caveat:** both external label conventions give possession to the last team to touch the ball before it goes out. BAR-ATH's ground truth follows "controlled possession", where a defender's clearance out does not flip possession. That would lower the switch rates above, but not the corner and goal-kick rules.

**Visible dead ball, noise v4** (2026-11-07).

On v3 the bench's visible dead ball was always still, at 0.26 m/s median with `ball_still_c4` = 1.00. IDSSE froze it and noise v3 placed Metrica's on the spot. So the detector learned "seen and moving = live". BAR-ATH aggregates, 5 s before a Smart Tagging set-piece start, ball seen: median speed 4.0 m/s, still on 4% of frames, off the pitch on 23%.

Dead-ball detector, leave one game out, dead cost 40 (IDSSE / Metrica):

| | dead-stretch F1 | visible moving dead frames called D | missing-ball live frames called D |
|---|---|---|---|
| v3 bench, old features | 0.961 / 0.882 | — (none on the bench) | — |
| v4 bench, old features | 0.918 / 0.836 | 0.880 / 0.764 | 0.079 / 0.381 |
| v4 bench, + latch and nearest speed | 0.926 / 0.832 | 0.887 / 0.759 | 0.082 / 0.387 |

- **The simulator fix is what teaches the model.** The new features are neutral on the bench. They are kept because they don't depend on how the dead ball moves.
- Dead cost 40 and team cost 6 are still the best cells. End to end (joint decoder) gives 0.672 / 0.700, against 0.681 / 0.721 on v3: the bench is harder, the pipeline is not worse.
- **Like for like on the v4 bench:** the v3 pipeline (trained on v3 games with the old features, leave one game out) scores 0.621 / 0.676, against 0.672 / 0.700 for v4. That is +0.050 / +0.024, and v4 wins on 20 of 20 games. Replicating the v3 pipeline on its own bench gives exactly 0.681 / 0.721. Baseline: 0.288 / 0.338. Dead-ball detector plus baseline team: 0.380 / 0.405.
- **The BAR-ATH sanity check** (aggregates only, never for selection) is the share called D around Smart Tagging set-piece starts:

| seconds from tag start | −10…−5 | −5…−2 | −2…0 | 0…2 | +10…+15 |
|---|---|---|---|---|---|
| ball seen, v3 | 0.00 | 0.00 | 0.01 | 0.18 | 0.02 |
| ball seen, v4 | 0.16 | 0.20 | 0.39 | 0.58 | 0.05 |
| ball missing, v4 | 0.95 | 0.97 | 0.94 | 0.96 | 0.69 |

  Better, but most visible dead ball on BAR-ATH is still called live. BAR-ATH players before tags move at 1.67 m/s with 55% walking, against about 1.0 m/s and 82–87% on bench stoppages, so the collective-slowdown signal reads weaker there.
- **Possible next levers:**
  - Smart Tagging set pieces as a rule-based prior. There are only 59 tags, 5 of them free kicks, and they can't be trained on external games.
  - The opposite error: missing-ball live frames are called D on 39% on Metrica, and P(dead) is 0.69 with the ball missing 10–15 s after a tag start on BAR-ATH.

## Gotchas

- Figshare's plain `ndownloader` links answer scripts with an empty HTTP 202 (a bot check). Do not work around it; `src/fetch_idsse.py` uses the official API (`api.figshare.com/v2/articles/28196177`) instead.
- Some IDSSE games track referees with `TeamId="referee"`. The ball FrameSet also has a `PersonId`.
- Metrica sample games 1 and 2 use CSV with 3 header rows; game 3 uses EPTS plus JSON events. The pitch is 105 × 68 m.
- `possession_metrics.spell_overlaps` is an O(n_gt × n_pred) Python loop, which is fine at about 500 spells.
- `evaluation/results/*.csv`, `evaluation/reports/` and `evaluation/scoreboard.csv` are gitignored, apart from the committed `baseline_*.csv` files.
