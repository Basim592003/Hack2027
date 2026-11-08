# Draft: rules question for the organisers (sprint item A6)

To be sent by the team. Fill in the bracketed parts.

---

**Subject:** MTAH 2027 Phase 1: rules question on simulator settings and label meaning

Hello,

We are [team name]. After your clarification of 3 November, that training and tuning must use external data only,
we would like to check four points that sit close to that line before we submit. In none of them is anything fitted to
the ground truth or the video. The question is only whether these uses of the provided *input* data are acceptable.

**1. Simulator settings taken from aggregates of BAR-ATH's tracking.** We train only on external labelled games: IDSSE
and your Metrica sample games. A simulator degrades their tracking to look like BAR-ATH's. Some of its settings come
from aggregate statistics of BAR-ATH's tracking and Smart Tagging, never from labels or video:
- the ball is missing on 40.7% of graded frames, 20.5 players are tracked on average, and there are 11.3 tracklet
  starts per minute;
- the ball is missing on about 75% of frames around Smart Tagging restart times;
- the simulated position error drifts smoothly, because BAR-ATH's tracking is smooth from frame to frame;
- our kick detector measures over ±12 frames rather than ±4, because BAR-ATH's ball shows few direction changes over
  4 frames;
- the simulated dead ball moves (fetched, carried, placed), because BAR-ATH's visible ball before tagged set pieces is
  rarely still.

We are now also randomising the amount of ball smoothing and the length of in-play ball gaps over wide ranges, so that
no single BAR-ATH value decides the training data. Is this use of input aggregates acceptable?

**2. Automatic time alignment.** If the ball track turns out to lag the player tracks, we would add a step that
estimates the lag from each match's own tracking and shifts the ball. It would be applied the same way to every match,
external games and BAR-ATH alike. Is that acceptable as improving the data by algorithm?

**3. Reading the public five minutes for label meaning.** May we look at the public ground truth to see whether a
change of possession is stamped at the new team's first touch or once control is established? We would use the answer
only to choose how we relabel the external games, as a rule. We would never use it to tune a parameter or to choose
between versions of our model.

**4. Metrica sample data licence.** The sample-data repository has no formal licence. May we use it for training and
validation, with attribution in our technical document?

Thank you,

[name], [team name]
