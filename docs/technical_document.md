# Ball possession from automatic tracking: technical document

**Team:** [team name] · More Than A Hack 2027, Phase 1 · Submission: `[team_name].csv`

## 1. Approach

The ranking metric scores **whole possessions**: maximal runs of `H` or `A`, matched one-to-one to the true runs by temporal overlap. A per-frame classifier flickers and invents hundreds of possessions; the published baseline predicts 131 on a stretch that has 17. So the pipeline makes the decision about **segments**, not frames.

It has three stages. **Every learned part is trained, and every setting chosen, on labelled games other than BAR-ATH.**

1. **Dead-ball detector.** For each frame it gives a probability that play is stopped. It is a gradient-boosted classifier over 96 signals computed from positions:
   - **ball:** visibility, speed, distance to the lines (negative outside), and whether it is at a corner, the 6-yard box or the centre spot;
   - **out of play:** whether the ball has been seen beyond a line since the last kick, and how long ago it was last off the pitch. Once out, play stays dead until somebody plays it, whatever the ball does in between;
   - **ball gaps:** how long the ball has been missing, and where it was last seen and where it reappears;
   - **players:** collective speed, the share walking or running, how fast the player nearest the ball moves (a ball carried at walking pace is not a dribble), players within 9.15 m of the ball, and kick-off formation;
   - each of these averaged over windows of 1–10 s before and after the frame.
2. **Team model.** For each live frame it gives a probability that the home team has the ball. It is a gradient-boosted classifier over 79 signals in four families:
   - **ball:** each team's nearest player, players within 1.5 m, whether the nearest player moves *with* the ball, and averages of these over the past and next 1–4 s, which separates a controlled spell from a single touch;
   - **team shape:** height up the pitch, length, width and forward speed of each team, and the team nearest the ball when it was last seen and when it reappears. This carries the decision on the ~41% of graded frames without a ball;
   - **restarts:** where play restarts after each detected stoppage, and whose player is nearest the ball at that moment;
   - **touches:** every *kick* of the ball is credited to a team. A kick is a peak in the change of the ball's velocity over half a second either side, with a player within 2 m; it goes to that player's team. The signals are:
     - the team of the last and the next kick, and how far away each is;
     - how many kicks in a row that team made;
     - the team of the kick before the last, which tells a completed pass from an interception;
     - kicks per team in the last and next 2 s;
     - the last and next team to be alone within 1.5 m of the ball.

     These are sharp at the moment the ball changes hands, where the averages above blur it.

   Every signal is computed from each team's point of view, with the pitch oriented so that team attacks the same way. Training uses both views of every frame and inference averages them. The model therefore cannot learn that "home usually has the ball", and its output is symmetric by construction.
3. **Decoder.** One Viterbi pass over four states: home, away, dead after home, and dead after away.
   - Frame evidence: P(D) = p_dead, P(H) = (1 − p_dead)·p_home, P(A) = (1 − p_dead)·(1 − p_home).
   - Every change of team costs 6 and every entry to or exit from a stoppage costs 40, in log-probability units. Both were chosen on the external games.
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
| Metrica Sports sample data (github.com/metrica-sports/sample-data, commit `e706dd5`) | 3 | ~283 min of play | Frame labels derived **by our code** from Metrica's human-annotated event data (`src/external.py`). Possession is the team of the latest on-ball event, with challenges excluded. A stoppage runs from the event that stopped play to the next set piece. | Events by Metrica Sports' annotators; frame labels by algorithm | No formal licence; README asks for responsible use and attribution |
| IDSSE: Bassek, Weber, Rein & Memmert (2025), *Scientific Data*, doi:10.1038/s41597-025-04505-y | 7 Bundesliga games (2022/23) | ~668 min of play | Per-frame `BallStatus` (alive/dead) and `BallPossession` (team) as delivered in the DFL position data | DFL data collection; see the paper | CC BY 4.0 |

**Making the external games look like the provided tracking** (`src/noise.py`). The external tracking is far cleaner than the provided ATD, so each game is degraded with fixed seeds before training:
- players: missing stretches and re-assignment to a 46-slot tracklet pool;
- positions of players and ball: a 0.15 m error that drifts smoothly over about half a second, as a filtered tracker's does, rather than jumping each frame;
- the ball: removed in about 75% of stoppages with blurred edges, and removed during live play at two levels (10% and 20% of live frames);
- the ball while play is stopped: neither source has a realistic one. IDSSE freezes the ball exactly where play stopped for every dead frame, then moves it to the restart spot in one jump. Metrica's sample tracking has no ball at all during stoppages. A model trained on that learns that a dead ball is either missing or perfectly still. So every stoppage is re-rendered from what happens in one:
  - the ball runs on and slows;
  - it is passed back, or a new ball appears;
  - it is carried by the restart taker, the player nearest the restart spot when play resumes, whose real movement is in the data;
  - it is set down on the spot.

  Durations and speeds are round football numbers, drawn afresh for each stoppage over wide ranges.

Each game is used at both dropout levels, giving 20 training games. The amount of player and ball dropout was set to match **aggregate statistics of the provided tracking and Smart Tagging**: share of frames without a ball, mean players tracked, tracklet restarts per minute, ball visibility around tagged set pieces, and how smooth positions are from frame to frame. Re-rendering the dead ball was prompted by one more such statistic: in the 5 s before a tagged set piece, the provided ball is still on only 4% of the frames it is seen. Its durations and speeds were not fitted to that. No ground truth and no video were involved. We have asked the organisers to confirm this is within the 2026-11-03 rules clarification. If it is not, the alternative is to train across a range of noise settings chosen without reference to the provided game.

## 4. Possession rules as implemented

- **Dead ball** is learned from the external labels. Signals the model relies on:
  - The ball beyond a line means dead 99% of the time on the IDSSE games.
  - A stoppage lasts until a restart: a still ball being kicked, or the ball reappearing at a restart spot.
  - A ball seen beyond a line stays dead until it is next kicked or thrown, even while it is carried or thrown back.
  - Collective slowing is a stoppage signature. On BAR-ATH, players move at 2.1 m/s with the ball tracked against 1.1 m/s without it.
- **Turnovers in play are anchored on kicks.** The new team's kick, followed by more of its own, marks where possession changes hands.
- **A change of possession needs sustained evidence for the other team.** The switch cost means a single touch, a deflection or a lost aerial duel rarely flips the label. That is our reading of "controlled possession" in CHALLENGE.md.
- **Contested or loose ball:** the label stays with the last team. The decoder never switches without evidence that pays the cost.
- **Restarts** follow the Laws of the Game:
  - a corner is restarted by the team attacking that goal (96–100% on the external games);
  - a goal kick by the team defending it (100%);
  - for throw-ins, free kicks and kick-offs, the team of the player nearest the ball at the restart is evidence.

  On the external games about 70% of stoppages hand the ball to the other team. Encoding that as an explicit prior in the decoder made no difference, so it is not used.
- **`X`** is used only outside the graded halves.

## 5. Validation on the external games

Each game was scored with the official `evaluation/possession_metrics.py`. Leave-one-game-out: the game being scored was never in training, at either noise level. The two sources are reported separately because their label conventions differ. DFL marks 41–52% of the match dead with 5.5–9 s possessions; the Metrica-derived labels give 25–33% dead and ~10 s possessions.

The final pipeline, on the bench with the re-rendered dead ball (§3):

| | spellF1 IDSSE | spellF1 Metrica |
|---|---|---|
| starter baseline (nearest player, ±2 s vote) | 0.288 | 0.338 |
| + our dead-ball detector (baseline team guess) | 0.380 | 0.405 |
| **final pipeline** | **0.672** | **0.700** |

How each part was added, measured on the earlier bench, whose dead ball was frozen or missing:

| | spellF1 IDSSE | spellF1 Metrica |
|---|---|---|
| starter baseline (nearest player, ±2 s vote) | 0.294 | 0.351 |
| + our dead-ball detector (baseline team guess) | 0.386 | 0.411 |
| + our team model without touches (dead ball and team decoded in sequence) | 0.610 | 0.670 |
| + joint decoder | 0.611 | 0.674 |
| + touch features | 0.681 | 0.721 |

The touch features also raise live-frame team accuracy from 0.887 / 0.893 to 0.905 / 0.905. Most of the real possessions they newly recover are short ones (§6).

**The dead ball the earlier bench showed was too easy.** When the ball was visible during a stoppage, it was always perfectly still, so the detector learned that a moving ball means play is live. On the re-rendered bench, a visible moving dead ball is called dead on 89% (IDSSE) and 76% (Metrica) of frames. Whole stoppages, matched like possessions, score F1 0.926 / 0.832, against 0.961 / 0.882 on the earlier bench. On the re-rendered bench, the previous pipeline (trained on the earlier bench) scores 0.621 / 0.676 against 0.672 / 0.700 for the final one, and it loses on every one of the 20 games. Its 0.681 / 0.721 on the earlier bench was flattered by the easy dead ball. The out-of-play and nearest-player-speed signals were neutral on the bench (stoppage F1 0.918 / 0.836 without them). They are kept because they don't depend on how the ball moves while play is stopped, which no bench can reproduce exactly.

**Does it carry over to BAR-ATH?** One check needs no labels. Play is almost always stopped in the seconds before a Smart Tagging set piece. With the ball visible there, the earlier pipeline called dead ball on 0–1% of frames, and the current one calls it on 16–39%. With the ball missing, both pipelines call it on 93–97%. That is better, but most visible dead ball on BAR-ATH is still called live (§6). This check was not used to choose anything.

An even earlier bench had a position error that jumped independently every frame. That made its tracking hundreds of times jerkier frame to frame than the provided tracking, so velocity signals meant different things in training and in use. The smoother error (§3) changed the rows above by under 0.01.

For reference, the starter baseline scores 0.243 on the full BAR-ATH match, so the external bench is somewhat easier than the real task.

**Do the touches carry over to BAR-ATH?** We can't score that without labels, but one check needs none. In real play most passes arrive, so consecutive kicks are mostly by the same team; kicks that were noise would give 50%. The share is 72% on BAR-ATH, against 79% (IDSSE) and 85% (Metrica) on the bench. Kicks are rarer on BAR-ATH, though: 25% of graded frames lie within 1 s of one, against 34–41%.

**Tried and dropped** (the first three and the last were measured before the touch features):
- An explicit "who restarts" prior in the decoder: ±0.002.
- Snapping team changes to the first touch of the new team: −0.04 / −0.02.
- Snapping stoppage edges to ball-out and kick moments: about 0.
- **Detecting kicks over ±4 frames** instead of ±12. It works on the bench but finds only 1 kick per graded minute on BAR-ATH, against 13–16 on the bench, because the provided ball track is smoothed over several frames. ±12 frames finds 9.
- **Kick strength as a feature.** The same smoothing makes kicks read softer on BAR-ATH (median 7 m/s against 11–12), so the feature would not carry over. Removing it changed the bench by under 0.002.
- **Training on "controlled possession" labels.** We relabelled the external games with the CHALLENGE.md definition: the restarting team owns the ball, and the other team takes over only with a spell of at least 1 s. Scored against those labels, a model trained on them did no better than one trained on the original last-touch labels (0.676 vs 0.680 IDSSE, 0.733 vs 0.741 Metrica). The decoder's switch cost already refuses to flip possession on a brief touch, which is the same rule.

## 6. Limitations

- **Short possessions.** Real possessions under 2 s are 28–30% of the external games' possessions. With touch features we recover 8–12% of those under 1 s and 36–38% of those lasting 1–2 s, against 2–7% and 21–24% before.
  - Some of these are artefacts of the external conventions, which give possession to whoever touched last.
  - Possessions that start with a turnover in play (about 77–79% of all) are matched 54–59% of the time, against 84–85% for those starting at a restart.
- **Weaker touch signal on BAR-ATH.** Kicks are rarer and softer there than on the bench, so the touch features probably help less there than the bench suggests.
- **Convention gap.** The external labels hand possession to the last toucher. BAR-ATH's ground truth requires control. Model weights and costs are tuned to the former.
- **Unknown live-ball dropout.** How often the ball is lost during live play on BAR-ATH cannot be measured without labels. We train across two levels, 10% and 20%, but the true value is an assumption.
- **Stoppage count.** On BAR-ATH the detector finds 118 stoppages, against 80–102 per match on the external games. Some may be live-play ball losses mistaken for stoppages.
- **Visible dead ball.** On BAR-ATH, a ball that stays visible while play is stopped is still mostly called live (above). The detector leans on the ball going missing, which happens in most stoppages but not all. Where it misses one, a stoppage can split into two, or two possessions can merge.
- **With more time:**
  - simulating the provided ball track's smoothing in the training noise, so kicks look alike in training and use;
  - a ball detector run on the tactical video, to recover the missing ball, which is the root cause;
  - a sequence model in place of windowed features.

## 7. Reproduction

Python 3.9.6, `pip install -r requirements-pipeline.txt`, with the match data unpacked into `data/` as in DATA.md.

```bash
python -m src.fetch_metrica      # external data, pinned to commit e706dd5
python -m src.fetch_idsse        # external data via the figshare API, MD5-checked (~2.5 GB)
python -m src.run --out evaluation/results/[team_name].csv
python evaluation/validate_submission.py evaluation/results/[team_name].csv
```

`src.run` takes about 5 minutes from an empty cache, or 4 with converted games cached, on a 10-core laptop with 16 GB of RAM. It is deterministic: fixed seeds, a fixed model `random_state` and pinned inputs. A rebuild from an empty `data/cache/` reproduced the file byte for byte (SHA-256 `22e4fd4cf15b39fac2eb3c728679220bfa9b587639c70362602aeea7c4c35b4d`).

The submission reports 368 possessions of mean 9.7 s, 43.8% of graded frames dead ball across 118 stoppages, and the home side on the ball for 60.9% of live time.

To reproduce the validation tables:
- `python -m src.deadball` gives §5 rows 1–2;
- `python -m src.possession` gives the team model;
- `python -m src.possession --no-touches` gives it without touches;
- `python -m src.decode` gives the joint decoder;
- `python -m src.touches` gives the kick statistics.
