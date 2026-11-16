# Ball possession from automatic tracking

**Team:** false 9 dev · More Than A Hack 2027, Phase 1 · Submission: `false 9 dev.csv` · Code: https://github.com/Basim592003/Hack2027

This is the short version; `docs/technical_details.md` in the repository is the full one, with every signal, comparison and check.

## 1. Approach

The metric scores whole possessions: maximal runs of `H` or `A`, matched one-to-one to the true runs by overlap, same team only. A per-frame classifier flickers and invents possessions; the published baseline predicts 131 on a stretch that has 17. So the pipeline estimates probabilities per frame and leaves the decision to a decoder that works on segments. **Every learned part is trained, and every setting chosen, on labelled games other than BAR-ATH.**

1. **Dead-ball detector.** A gradient-boosted classifier gives P(play stopped) for each frame from 96 signals computed from positions. They cover where the ball is and how fast it moves, whether it has crossed a line since the last kick, how long it has been missing and where it reappears, how fast the players move and how many walk, players within 9.15 m of the ball, and kick-off formation, each averaged over windows of 1–10 s.
2. **Team model.** A second classifier gives P(home has the ball) for each live frame from 118 signals:
   - each team's nearest player, players within 1.5 m, and whether the nearest player moves with the ball, over the past and next 1–4 s;
   - team shape: height up the pitch, length, width and forward speed, and the team nearest the ball when it was last seen and when it reappears. These carry the decision on the ~41% of frames without a ball;
   - restarts: where play restarts after a stoppage, and whose player is nearest the ball then;
   - **kicks**: peaks in the change of the ball's velocity with a player within 2 m, credited to that player's team. Signals include the team of the last and next kick and how many kicks in a row it made;
   - the ball signals averaged between **events** (a kick, a change of the team alone near the ball, the ball disappearing or reappearing, a stoppage edge), so that no average straddles the moment the ball changes hands.

   Every signal is computed from each team's point of view, and the two views are averaged. The model therefore cannot learn that "home usually has the ball".
3. **Second stage.** A further classifier sees the team model's signals plus 31 summaries of the team model's own output around the frame: its log-odds 1–8 s before and after, and their means over windows and segments. It is trained only on out-of-fold predictions, so it learns from the other games when the team model's guess holds.
4. **Decoder.** One Viterbi pass over four states: home, away, dead after home, and dead after away. A change of team costs 12 and entering or leaving a stoppage costs 40, in log-probability units; both were chosen on the external games. A team change therefore needs sustained evidence. Frames outside the graded halves are `X`.

## 2. Assets used and dropped

| asset | used? | how / why not |
|---|---|---|
| tactical ATD: players, ball, metadata | **yes** | Positions and each slot's team only. Speeds are recomputed from positions, so every signal means the same as on the external games. |
| graded half boundaries (`games.py`) | **yes** | Each half is decoded separately. |
| Smart Tagging | **not as a model input** | The external games have none. Used only in aggregate, to shape the training noise (§3) and for checks that chose nothing. |
| video, broadcast ATD, homographies | no | Out of scope for the time available. The broadcast data is not synchronised to the tactical timeline. |

**Tried and dropped**, because each lost or tied on the external games:
- per-frame switch costs, cut near events or learned from a change detector;
- snapping team changes to touches;
- a "who restarts" prior;
- kick strength, which reads softer on BAR-ATH's smoothed ball;
- who reaches a moving ball first;
- pressure signals;
- monotonic constraints;
- other classifier settings;
- more noise copies;
- for the dead-ball detector, denser training around stoppage edges, or kick and segment signals.

## 3. Training data

**No data from BAR-ATH was used to train or tune anything.** The public 5 minutes were used only to check frame alignment and file format.

| dataset | games | labels | labelled by | licence |
|---|---|---|---|---|
| Metrica Sports sample data (github.com/metrica-sports/sample-data, commit `e706dd5`) | 3, ~283 min of play | Frame labels derived by our code (`src/external.py`) from the event data. Possession is the team of the latest on-ball event, with challenges excluded. A stoppage runs from the event that stopped play to the next set piece. Stretches of 4 s or more with no ball and no event also count as stoppages. | Events by Metrica Sports' annotators; frame labels by algorithm | No formal licence; the README asks for responsible use and attribution |
| IDSSE: Bassek, Weber, Rein & Memmert (2025), *Scientific Data*, doi:10.1038/s41597-025-04505-y | 7 Bundesliga games, ~668 min of play | Per-frame `BallStatus` and `BallPossession` flags | DFL data collection | CC BY 4.0 |

**Making the external games look like the provided tracking** (`src/noise.py`). Each game is degraded with fixed seeds:
- players go missing and are re-slotted into a 46-slot tracklet pool;
- positions get a smoothly drifting 0.15 m error;
- the ball is smoothed by a filter whose width is drawn per copy between 0 and 10 frames;
- the ball disappears in about 75% of stoppages and on 10% or 20% of live frames;
- the ball's path through every stoppage is re-rendered: it runs on, is fetched, is carried by the restart taker and is set down. This is needed because IDSSE freezes the ball where play stopped and Metrica's sample has no ball during stoppages.

Each game is used at both dropout levels, giving 20 copies for scoring. A second copy of each, made with another seed, only adds training rows.

These amounts were set to match **aggregate statistics of the provided tracking and Smart Tagging**:
- the share of frames without a ball;
- players tracked and tracklet restarts per minute;
- ball visibility around tagged set pieces;
- frame-to-frame smoothness;
- how often the ball's velocity changes over 4 frames against over 12.

The re-rendered stoppage ball was prompted by one more: the provided ball is rarely still before a tagged set piece. No ground truth and no video were involved. These statistics shape the training data rather than tune the algorithm. We read that as within the 2026-11-03 rules clarification, and list them here so the organisers can judge. Where a value could only be read off the provided game, such as the filter width, training draws it from a wide range around that value.

## 4. Possession rules as implemented

- **Controlled possession.** A change of team needs sustained evidence for the other side, which the decoder's switch cost enforces. A single touch, a deflection or a lost aerial duel rarely flips the label. We score against the external labels relabelled to this reading (`src/controlled.py`): the team that restarts play owns the ball, and the other team takes over only with a spell of at least 1 s.
- **Loose and contested balls** stay with the last team.
- **Turnovers in play** are anchored on events: the new team's kick followed by more of its own, its player becoming the only one near the ball, or the ball reappearing with its side.
- **Dead ball** is learned from the external labels. A ball seen beyond a line stays dead until it is next kicked or thrown, and a stoppage lasts until a restart. Several seconds with no ball and nobody playing it also mark a stoppage.
- **Restarts** follow the Laws of the Game. Corners are taken by the attacking team (96–100% on the external games) and goal kicks by the defending team (100%). For throw-ins, free kicks and kick-offs, the team nearest the ball at the restart is evidence.
- **`X`** is written only outside the graded halves.

## 5. Results and limitations

Each copy is scored leave-one-game-out with the official `evaluation/possession_metrics.py`. Results are reported per source, IDSSE / Metrica, because the label conventions differ.

| spellF1 | controlled, IDSSE | controlled, Metrica | last touch, IDSSE | last touch, Metrica |
|---|---|---|---|---|
| starter baseline | 0.298 | 0.337 | 0.277 | 0.318 |
| **final pipeline** | **0.690** | **0.770** | **0.637** | **0.710** |

Controlled macroF1 is 0.912 / 0.929. A change was kept only if it gained on both sources (or 0.005 on one with the other flat) and won most paired copies of each. The second stage is the one exception. It won 12 of 14 IDSSE copies but only 3 of 6 Metrica copies, where the rule asks for 4, and was flat on Metrica. It also raised last-touch spellF1 by +0.009 / +0.007.

- **Short possessions are the main weakness.**
  - Possessions under 4 s are 42% / 34% of the real ones. Only 46% / 52% of them are matched, against 87% / 91% of those over 8 s.
  - On possessions under 2 s that start in play, the true team gets a probability of only 0.54 on average. The tracking often does not show who won a duel or a deflection.
  - Given the true team, controlled spellF1 would rise by +0.27 / +0.18. Given the true dead ball, it would rise by only +0.03 / +0.05.
- **Few labelled games.** Trained on 3, 6 and 9 other games, the team model scores 0.673, 0.678 and 0.684 on IDSSE, so more games would still help a little.
- **Convention gap.** The models learn last-touch labels and are scored on our reading of controlled possession. BAR-ATH's annotators may draw the line elsewhere.
- **Stoppages.** On BAR-ATH the detector finds 106 stoppages and 44% dead ball, close to the external games' 96–102 and 38–45%. A stopped ball that stays visible is still mostly called live.
- **With more time:** a ball detector on the tactical video, to recover the ball that is missing on about 41% of frames, and a sequence model in place of windowed features.

## 6. Reproduction

Install Python 3.9.6, run `pip install -r requirements-pipeline.txt`, and unpack the match data into `data/` as in DATA.md:

```bash
python -m src.fetch_metrica      # external data, pinned to commit e706dd5
python -m src.fetch_idsse        # external data via the figshare API, MD5-checked (~2.5 GB)
python -m src.run --out "evaluation/results/false 9 dev.csv"
python evaluation/validate_submission.py "evaluation/results/false 9 dev.csv"
```

`src.run` takes about 18 minutes from an empty cache on a 10-core laptop with 16 GB of RAM, peaking at 5.3 GB. It is deterministic: two rebuilds from an empty `data/cache/` gave the same file byte for byte (SHA-256 `ce4cdd876e5b8ebc4728b518ad56d476c23130a5d1a638cf06c7a941799926fb`). The experiments behind §5 run through `python -m src.experiment`; §7 of `docs/technical_details.md` lists them.
