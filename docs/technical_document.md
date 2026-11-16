# Ball possession from automatic tracking: technical document

**Team:** false 9 dev · More Than A Hack 2027, Phase 1 · Submission: `false 9 dev.csv` · Code: https://github.com/Basim592003/Hack2027

## 1. Approach

The ranking metric scores **whole possessions**: maximal runs of `H` or `A`, matched one-to-one to the true runs by temporal overlap. A per-frame classifier flickers and invents hundreds of possessions; the published baseline predicts 131 on a stretch that has 17. So the pipeline makes the decision about **segments**, not frames.

It has four stages. **Every learned part is trained, and every setting chosen, on labelled games other than BAR-ATH.**

1. **Dead-ball detector.** For each frame it gives a probability that play is stopped. It is a gradient-boosted classifier over 96 signals computed from positions:
   - **ball:** visibility, speed, distance to the lines (negative outside), and whether it is at a corner, the 6-yard box or the centre spot;
   - **out of play:** whether the ball has been seen beyond a line since the last kick, and how long ago it was last off the pitch. Once out, play stays dead until somebody plays it, whatever the ball does in between;
   - **ball gaps:** how long the ball has been missing, and where it was last seen and where it reappears;
   - **players:** collective speed, the share walking or running, how fast the player nearest the ball moves (a ball carried at walking pace is not a dribble), players within 9.15 m of the ball, and kick-off formation;
   - each of these averaged over windows of 1–10 s before and after the frame.
2. **Team model.** For each live frame it gives a probability that the home team has the ball. It is a gradient-boosted classifier over 118 signals in five families:
   - **ball:** each team's nearest player, players within 1.5 m, whether the nearest player moves *with* the ball, and averages of these over the past and next 1–4 s, which separates a controlled spell from a single touch;
   - **team shape:** height up the pitch, length, width and forward speed of each team; how that forward speed changes, since a team that loses the ball stops pushing up; and the team nearest the ball when it was last seen and when it reappears. This carries the decision on the ~41% of graded frames without a ball;
   - **restarts:** where play restarts after each detected stoppage, and whose player is nearest the ball at that moment;
   - **touches:** every *kick* of the ball is credited to a team. A kick is a peak in the change of the ball's velocity over half a second either side, with a player within 2 m; it goes to that player's team. The signals are:
     - the team of the last and the next kick, and how far away each is;
     - how many kicks in a row that team made;
     - the team of the kick before the last, which tells a completed pass from an interception;
     - kicks per team in the last and next 2 s;
     - the last and next team to be alone within 1.5 m of the ball;
   - **segments between events.** An event is any moment the ball can change hands:
     - a kick;
     - a change of the team alone within 1.5 m of the ball;
     - the ball disappearing or reappearing;
     - the start or end of a detected stoppage.

     Events within 3 frames of each other merge. For each frame, the nearest-team, control, distance and co-movement signals are averaged over the stretch between the events around it and over the stretches before and after it, together with the differences between them. The stretch's length, the frame's position in it and the share of it with the ball seen are signals too, as are the kinds and team of the events that bound it. Fixed 1–4 s windows straddle the moment the ball changes hands; these averages stop at it.

   Every signal is computed from each team's point of view, with the pitch oriented so that team attacks the same way. Training uses both views of every frame and inference averages them. The model therefore cannot learn that "home usually has the ball", and its output is symmetric by construction.

   **Training rows** are both views of every 5th live frame, plus every live frame within 12 frames (0.5 s) of a change of team, where the decision is hardest. They come from the 20 degraded external games of §3 plus a second, independently degraded copy of each.
3. **Second stage.** A second gradient-boosted classifier gives P(home) again (`src/stack.py`). Next to the team model's 118 signals, it sees 31 more made from the team model's own output around the frame:
   - its log-odds 1, 2, 4 and 8 s before and after;
   - their mean over seven windows of up to 8 s, and their spread over two;
   - their mean over the segment, the one before and the one after, with the differences;
   - the teams of the events that bound the segment, P(dead), and how far the frame is from a stoppage.

   The decoder adds frames up as if each were separate evidence, but the team model's mistakes last as long as whatever misleads it. The second stage learns from the other games when the team model's guess holds. It is trained only on the team model's out-of-fold predictions, so it learns what the team model is like on a game it has not seen, and it is symmetric in the same way.
4. **Decoder.** One Viterbi pass over four states: home, away, dead after home, and dead after away.
   - Frame evidence: P(D) = p_dead, P(H) = (1 − p_dead)·p_home, P(A) = (1 − p_dead)·(1 − p_home), with p_home from the second stage.
   - Every change of team costs 12 and every entry to or exit from a stoppage costs 40, in log-probability units. Both were chosen on the external games. The team cost was re-chosen twice: when denser training made the team model sharper at hand-overs, and for the second stage, which finds more of them.
   - A team change only happens where the evidence for the other team is sustained enough to pay that cost.
   - Frames outside the two graded halves are written as `X`.

## 2. Assets used and not used

| asset | used? | how / why not |
|---|---|---|
| tactical ATD, players and ball | **yes** | Positions only. Speeds are recomputed from displacements over ~1 s, so a signal means the same on BAR-ATH as on the external games, which carry no speed channel. |
| tactical ATD metadata | **yes** | Team of each track slot. |
| graded half boundaries (`games.py`) | **yes** | To decode each half separately and write `X` outside them. |
| Smart Tagging | **only to shape the training noise and for automatic checks** (see §3, §5) | Not a model input. The external games have no Smart Tagging, so a model using it could not be trained or tuned on external data. |
| tactical and broadcast video | no | Out of scope for the time available. Video-based ball detection is the obvious next step (§6). |
| broadcast ATD, homographies | no | Not synchronised to the tactical timeline, and without video the homographies add nothing. |

## 3. Training data

**No data from BAR-ATH was used to train or tune anything.** The public 5-minute ground truth was used only to confirm frame alignment and file format.

| dataset | games | labelled frames | what the labels are | who labelled it | licence |
|---|---|---|---|---|---|
| Metrica Sports sample data (github.com/metrica-sports/sample-data, commit `e706dd5`) | 3 | ~283 min of play | Frame labels derived **by our code** from Metrica's human-annotated event data (`src/external.py`). Possession is the team of the latest on-ball event, with challenges excluded. A stoppage runs from the event that stopped play to the next set piece. Stretches of 4 s or more with no ball and no event are also stoppages (below). | Events by Metrica Sports' annotators; frame labels by algorithm | No formal licence; README asks for responsible use and attribution |
| IDSSE: Bassek, Weber, Rein & Memmert (2025), *Scientific Data*, doi:10.1038/s41597-025-04505-y | 7 Bundesliga games (2022/23) | ~668 min of play | Per-frame `BallStatus` (alive/dead) and `BallPossession` (team) as delivered in the DFL position data | DFL data collection; see the paper | CC BY 4.0 |

**Stoppages missing from the Metrica labels.** The Metrica sample's tracking blanks the ball through every stoppage, but its event data only closes a stoppage at a set piece. Some stretches have no ball for 4 s or more while the events call play live. There are 47 such stretches, 31,413 frames or about 7 minutes a game. None of them contains an event of any type, and the players walk through them: mean speed 1.26 m/s, under 1.5 m/s on 75% of frames. Annotated stoppages look the same (1.02 m/s and 87%), and live play does not (2.18 m/s and 22%). Shorter gaps contain an event 39% of the time, as live play does. These stretches are labelled dead, which raises Metrica's dead share from 25–33% to 32–41%. Trained on the corrected labels, the pipeline also scores 0.005 higher on the IDSSE games, whose data did not change (§5).

**Making the external games look like the provided tracking** (`src/noise.py`). The external tracking is far cleaner than the provided ATD, so each game is degraded with fixed seeds before training:
- players: missing stretches and re-assignment to a 46-slot tracklet pool;
- positions of players and ball: a 0.15 m error that drifts smoothly over about half a second, as a filtered tracker's does, rather than jumping each frame;
- the ball's track: smoothed by a zero-phase Gaussian whose width is drawn per copy between 0 and 10 frames. The provided ball is filtered: its velocity changes over 4 frames are a fifth as frequent as over 12, which matches a width of about 7.5–8 frames on the external games. The range includes no smoothing, so no single value decides the training data. The provided ball does not lag its players, so the filter has no delay;
- the ball's absence: removed in about 75% of stoppages with blurred edges, and during live play at two levels (10% and 20% of live frames), in gaps whose median length is drawn per copy between 40 and 200 frames;
- the ball while play is stopped: neither source has a realistic one. IDSSE freezes the ball exactly where play stopped for every dead frame, then moves it to the restart spot in one jump. Metrica's sample tracking has no ball at all during stoppages. A model trained on that learns that a dead ball is either missing or perfectly still. So every stoppage is re-rendered from what happens in one:
  - the ball runs on and slows;
  - it is passed back, or a new ball appears;
  - it is carried by the restart taker, the player nearest the restart spot when play resumes, whose real movement is in the data;
  - it is set down on the spot.

  Durations and speeds are round football numbers, drawn afresh for each stoppage over wide ranges.

Each game is used at both dropout levels, giving 20 training games. Each of those is degraded a second time with another seed, and that second copy only adds training rows for the team model; scoring always uses the first. Every copy draws its own random numbers.

The amount of player and ball dropout and the ball's smoothing were set to match **aggregate statistics of the provided tracking and Smart Tagging**:
- the share of frames without a ball;
- mean players tracked and tracklet restarts per minute;
- ball visibility around tagged set pieces;
- how smooth positions are from frame to frame;
- how often the ball's velocity changes over 4 frames against 12.

Re-rendering the dead ball was prompted by one more such statistic: in the 5 s before a tagged set piece, the provided ball is still on only 4% of the frames it is seen. Its durations and speeds were not fitted to that. No ground truth and no video were involved. These are aggregates of the provided input, used to make the training data realistic rather than to tune the algorithm. We read that as within the 2026-11-03 rules clarification, and list each such statistic here so the organisers can judge. Where a value could only be read off the provided game, such as the ball filter's width, training draws it from a wide range around that value instead.

## 4. Possession rules as implemented

- **Dead ball** is learned from the external labels. Signals the model relies on:
  - The ball beyond a line means dead 99% of the time on the IDSSE games.
  - A stoppage lasts until a restart: a still ball being kicked, or the ball reappearing at a restart spot.
  - A ball seen beyond a line stays dead until it is next kicked or thrown, even while it is carried or thrown back.
  - Collective slowing is a stoppage signature. On BAR-ATH, players move at 2.1 m/s with the ball tracked against 1.1 m/s without it.
  - Several seconds with no ball and nobody playing it is a stoppage, which the corrected Metrica labels (§3) now teach.
- **Turnovers in play are anchored on events.** The marks are the new team's kick followed by more of its own, the other team's player becoming the only one near the ball, or the ball reappearing with the other side. The evidence on each side of such an event is pooled separately (§1).
- **A change of possession needs sustained evidence for the other team.** The switch cost means a single touch, a deflection or a lost aerial duel rarely flips the label. That is our reading of "controlled possession" in CHALLENGE.md, and every result is scored against that definition (§5).
- **Contested or loose ball:** the label stays with the last team. The decoder never switches without evidence that pays the cost.
- **Restarts** follow the Laws of the Game:
  - a corner is restarted by the team attacking that goal (96–100% on the external games);
  - a goal kick by the team defending it (100%);
  - for throw-ins, free kicks and kick-offs, the team of the player nearest the ball at the restart is evidence.

  On the external games about 70% of stoppages hand the ball to the other team. Encoding that as an explicit prior in the decoder made no difference, so it is not used.
- **`X`** is used only outside the graded halves.

## 5. Validation on the external games

Each game was scored with the official `evaluation/possession_metrics.py`. Leave-one-game-out: the game being scored was never in training, at any noise level or seed. The two sources are reported separately because their label conventions differ. DFL marks 45% of the match dead across about 102 stoppages a match; the corrected Metrica labels mark 38% across 96.

**Two label conventions.** The external labels give possession to whoever touched the ball last. BAR-ATH's answer key counts *controlled* possession (CHALLENGE.md), so every result is also scored against the external labels relabelled to that definition (`src/controlled.py`). The team that restarts play owns the ball, and the other team takes over only with a spell of at least 1 s. That removes 15–25% of the external possessions while changing under 1% of frames. The controlled score is the primary one. The models are trained on the last-touch labels: training on the relabelled ones did no better (0.676 vs 0.680 IDSSE, 0.733 vs 0.741 Metrica).

**The final pipeline**, on the current bench:

| spellF1 | controlled, IDSSE | controlled, Metrica | last touch, IDSSE | last touch, Metrica |
|---|---|---|---|---|
| starter baseline (nearest player, ±2 s vote) | 0.298 | 0.337 | 0.277 | 0.318 |
| **final pipeline** | **0.690** | **0.770** | **0.637** | **0.710** |

Controlled macroF1 is 0.912 / 0.929.

**How the last steps added up.** Each step was kept only if it gained on both sources, or 0.005 on one with the other flat, and won most paired copies in each. The last step is the one exception (below). Controlled spellF1, IDSSE / Metrica, each row on top of the one before:

| step | IDSSE | Metrica |
|---|---|---|
| pipeline on the bench with the ball filter (§3) | 0.663 | 0.730 |
| + segment features (§1) | 0.666 | 0.742 |
| + corrected Metrica labels (§3); Metrica scored on them from here on | 0.671 | 0.754 |
| + training rows near every change of team, team cost 6 → 10 | 0.677 | 0.761 |
| + a second noise copy of every game for training | 0.684 | 0.770 |
| + second-stage team model, team cost 10 → 12 | **0.690** | **0.770** |

On the corrected labels, the model before that step scores 0.745 on Metrica, so the label fix itself is worth +0.009 there and +0.005 on IDSSE.

The second stage wins 12 of 14 IDSSE copies but only 3 of 6 Metrica copies, where the rule asks for 4. On Metrica it is flat (+0.0004), with 2 of its 3 games up. It was kept because the gain is broad:
- last touch rises by +0.009 / +0.007;
- more turnovers in play are recovered, 59% → 61% on IDSSE and 69.7% → 69.9% on Metrica;
- the true team gets more probability in short possessions: 0.51 → 0.54 and 0.50 → 0.54.

Its evaluation is nested. Each held-out game's second stage is trained on team-model predictions from models that saw neither that game nor the one predicted.

**How the pipeline was built**, on an earlier bench whose dead ball was frozen or missing, last-touch labels:

| | spellF1 IDSSE | spellF1 Metrica |
|---|---|---|
| starter baseline | 0.294 | 0.351 |
| + our dead-ball detector (baseline team guess) | 0.386 | 0.411 |
| + our team model without touches (dead ball and team decoded in sequence) | 0.610 | 0.670 |
| + joint decoder | 0.611 | 0.674 |
| + touch features | 0.681 | 0.721 |

**The bench was made harder twice, closer to the provided tracking.** Each time the pipeline was retrained, and a model trained on the new bench beat the old one on it.
- Re-rendering the dead ball (§3): the earlier dead ball was too easy. Any visible stoppage ball was perfectly still, so the detector learned that a moving ball means play is live. On the re-rendered bench the previous pipeline scores 0.621 / 0.676 against 0.672 / 0.700, and it loses on every one of the 20 games.
- Smoothing the ball (§3): on the most heavily smoothed copies, training with the filter gains +0.05 controlled spellF1, on all 6 copies.

**Where the remaining error is.** On the bench, giving the final pipeline's decoder the true team would raise controlled spellF1 by +0.27 / +0.18, the true dead ball by +0.03 / +0.05, and perfect timing of team changes by +0.05 / +0.03. So the errors left are mostly *which* team has the ball in short possessions (§6).

**Does it carry over to BAR-ATH?** We can't score that without labels, but some checks need none. None of them was used to choose anything.
- **Ball filter.** Kicks are as frequent on BAR-ATH as on the bench: 8.9 per graded minute, against 8.8 on the IDSSE copies.
- **Events.** There are 17.4 events per graded minute on BAR-ATH against 17.6 on IDSSE, and ±12-frame windows around them cover 25% of graded frames on both.
- **Passes arrive.** Consecutive kicks are by the same team on 72% of BAR-ATH's against 74% on the bench. For flights, project the ball in a straight line and give each player 0.7 s to react, then 5 m/s. The kicker's team reaches the ball first on 75% of BAR-ATH's flights, against 74% (IDSSE) and 78% (Metrica).
- **Ball lag.** BAR-ATH's ball leads its carrier by 2.5 frames in a dribble, like IDSSE's 3.1, so there is no filter delay to undo.
- **Dead ball before set pieces.** Play is almost always stopped in the seconds before a Smart Tagging set piece. Measured on 2026-11-07 with the re-rendered-bench pipeline: with the ball visible there, dead ball was called on 16–39% of frames, against 0–1% before; with the ball missing, on 93–97%.

**Tried and dropped:**
- An explicit "who restarts" prior in the decoder: ±0.002.
- Snapping team changes to the first touch of the new team: −0.04 / −0.02. Snapping stoppage edges to ball-out and kick moments: about 0.
- **Detecting kicks over ±4 frames** instead of ±12. It works on the bench but finds only 1 kick per graded minute on BAR-ATH, because the provided ball track is smoothed over several frames. ±12 frames finds 9.
- **Kick strength as a feature.** The same smoothing makes kicks read softer on BAR-ATH (median 7 m/s against 11–12), so the feature would not carry over. Removing it changed the bench by under 0.002.
- Switching off the dead-ball model's built-in early stopping: −0.002 / −0.010. Its validation holdout evidently regularises it. In the team model, where it was neutral, it is off.
- **Per-frame team-switch costs**, either cut near events or learned from a change detector (AUC 0.88 / 0.94): +0.002 at best for the first and below zero everywhere for the second. Events come every 3 s and most are not hand-overs, and the team model's output already moves where the ball does change hands.
- **Who reaches a moving ball first**, as a family of team-model signals: +0.004 / −0.001. The team model already sees who next touches the ball, so the physics only adds where the reception is never seen.
- **Pressure and team spreading**: closing speeds of nearby opponents, and the divergence of each team's velocity field. +0.001 / −0.002.
- **Monotonic constraints** on the team model's signals: −0.004 / −0.013.
- **Classifier settings**: more leaves, slower learning, feature subsampling, larger leaves. None was better on both sources; the closest, 63 leaves, gave +0.005 / −0.000.
- **More training copies**: a third noise copy of every game gave −0.000 / −0.001. Training the dead-ball model on the second copy too gave −0.001 / −0.003.
- **Training the dead-ball detector densely around stoppage edges**, as the team model is around hand-overs: −0.002 / −0.002.
- **Kick and segment signals for the dead-ball detector**: −0.004 / −0.007. These were the time since and until a kick, its strength and how still the ball was before it, kick rates, and six signals averaged over the segments between kicks and ball gaps.

## 6. Limitations

- **Short possessions.** On frames of real possessions under 2 s that start in play, the second stage gives the true team a probability of only about one half on average: 0.54 / 0.54, up from 0.46 / 0.44 before the denser training and the second stage. Possessions that start with a turnover in play are matched 61% / 70% of the time (IoU 0.5), far less often than those that start at a restart. Possessions under 4 s are 42% / 34% of all real ones, and 46% / 52% of them are matched, against 87% / 91% of those over 8 s. Four in five misses are possessions given to the wrong team, or run together with a neighbour when a brief possession between them is missed.
- **Few labelled games.** Trained on 3, 6 and 9 of the other games, the team model scores 0.673 / 0.758, 0.678 / 0.761 and 0.684 / 0.770, so it is still gaining from data. The public candidates were not brought in: PFF's World Cup 2022 games and SkillCorner's A-League games are broadcast tracking from other providers, and there was not time to convert and check them.
- **Convention gap.** The models are trained on last-touch labels and scored on our reading of controlled possession. BAR-ATH's annotators may draw the line differently.
- **Unknown ball-loss pattern.** How often and for how long the ball is lost during live play on BAR-ATH cannot be measured without labels. The bench draws gap lengths over a wide range. BAR-ATH's ball is never missing for 1–4 s, so its tracker evidently fills short losses, while the bench's often is. Short gaps are decoded correctly on the bench (1–2% wrongly called dead), so this probably costs little.
- **The ball filter is not Gaussian.** BAR-ATH's ball matches the heavily smoothed bench copies on short-scale smoothness, but its kick rate matches the moderately smoothed ones. The range of filter widths in training covers both.
- **Stoppages.** On BAR-ATH the detector finds 106 stoppages and 44% dead ball. The bench's real figures are 96–102 per match and 38–45%, so this looks plausible, but a missed stoppage can still split a possession or merge two.
- **Visible dead ball.** The detector leans on the ball going missing, which happens in most stoppages but not all. On BAR-ATH a ball that stays visible while play is stopped is still mostly called live (§5).
- **With more time:** a ball detector run on the tactical video, to recover the missing ball, which is the root cause; and a sequence model in place of windowed features.

## 7. Reproduction

Python 3.9.6, `pip install -r requirements-pipeline.txt`, with the match data unpacked into `data/` as in DATA.md.

```bash
python -m src.fetch_metrica      # external data, pinned to commit e706dd5
python -m src.fetch_idsse        # external data via the figshare API, MD5-checked (~2.5 GB)
python -m src.run --out "evaluation/results/false 9 dev.csv"
python evaluation/validate_submission.py "evaluation/results/false 9 dev.csv"
```

`src.run` takes about 18 minutes from an empty cache on a 10-core laptop with 16 GB of RAM, peaking at 5.3 GB. It is deterministic: fixed seeds, a fixed model `random_state` and pinned inputs. Two rebuilds from an empty `data/cache/` reproduced the file byte for byte (SHA-256 `ce4cdd876e5b8ebc4728b518ad56d476c23130a5d1a638cf06c7a941799926fb`).

The submission reports 326 possessions of mean 10.9 s, 44.0% of graded frames dead ball across 106 stoppages, and the home side on the ball for 60.7% of live time.

To reproduce §5:
- `python -m src.experiment run NAME` gives out-of-fold predictions on the bench, decodes them, and scores both conventions;
- `python -m src.experiment run NAME --stack` does the same with the second stage, in nested folds (about 3.5 hours);
- `python -m src.experiment learncurve` trains the team model on fewer of the games;
- `python -m src.experiment compare A B` pairs two runs copy by copy;
- `python -m src.experiment oracles` gives the headroom;
- `python -m src.diag.gaps NAME` breaks a saved run's missed possessions down by length, cause and how they start;
- `python -m src.experiment costgrid NAME` gives the decoder costs;
- `python -m src.experiment eventgrid` and `boundarygrid` give the per-frame switch costs;
- `python -m src.segments`, `src.flight`, `src.lagcheck`, `src.touches` and `src.coverage` give the carry-over checks, which print BAR-ATH aggregates only.

Flags at the top of these modules switch the signal families and training choices compared above:
- `src/possession.py`: `SEGMENTS`, `FLIGHT`, `PRESSURE`, `MONOTONIC`, `DENSE`, `EXTRA_SEEDS`;
- `src/deadball.py`: `TRAIN_EXTRA`, `DENSE`;
- `src/features.py`: `EVENTS`;
- `src/run.py`: `STACKED`, for the second stage. `docs/sprint_checklist.md` records every comparison, with its numbers and date.
