"""Touches: the frames a player plays the ball, and that player's team.

The team model's ball features are averages over 1-4 s, which blur the moment the ball changes
hands. On the external bench, frames within 1 s of a turnover in play are on the wrong team
28-34% of the time, against under 2% more than 3 s from one, and real possessions under 2 s that
start in play get P(true team) below 0.5 on average. A touch is a sharp event:

  kick    the ball's velocity changes -- a pass, shot, clearance, block or first touch -- by at
          least KINK_SPEED between the KINK_LAG frames before and after, at a local maximum, with
          all three positions tracked (never interpolated) and a player within TOUCH_RADIUS. It
          belongs to the nearest player's team.
  close   exactly one team has a player within src/possession.py's CONTROL_RADIUS of the
          ball: dribbling, shielding, receiving.

From those, per frame: the team of the last and the next kick and how far away each is, how many
kicks in a row that team had (CHALLENGE.md: control is keeping the ball, or two or more touches
by one team), the team of the kick before the last (a completed pass or an interception), kicks
by each team in the last and next 2 s, and the last and next "close" team. src/possession.py
turns them into features from each team's point of view.

Nothing is fitted. The constants are round physical numbers; KINK_SPEED sits above what tracking
drift alone produces. KINK_LAG is half a second because a tracker's ball filter smooths a kick
over several frames: BAR-ATH's ball shows 2.8 velocity kinks per graded minute over 4 frames but
14.5 over 12, against 16-24 and 15-18 on the bench, so a short lag would see touches on the
external games and almost none on BAR-ATH. At 12 frames, on the bench, the kick's team matches
the team in possession just after it 94% (IDSSE) / 97% (Metrica) of the time, and 52% / 46% of
turnovers in play have a kick by the new team within half a second.

Do they carry over to BAR-ATH? Without labels, one check: in real play most passes arrive, so
consecutive kicks are mostly by the same team; noise would give 50%. BAR-ATH 72%, bench 79% /
85%. BAR-ATH's kicks are rarer (25% of graded frames within 1 s of one, against 34% / 41%) and
softer (median 7 m/s, against 11-12): its ball filter flattens them. So the kick's strength is
not a feature -- it would read differently on BAR-ATH -- and dropping it cost nothing on the bench.

What it bought (2026-11-06, noise v3, leave one game out, joint decoder at team cost 6): spellF1
0.611 -> 0.681 IDSSE, 0.674 -> 0.721 Metrica; live team accuracy 0.887 -> 0.905, 0.893 -> 0.905.

    python -m src.touches       # agreement with the external labels; aggregate rates on BAR-ATH
"""

import numpy as np

from .features import CONTEXT_CAP

KINK_LAG = 12            # frames, ~0.5 s either side over which the ball's velocity is measured
KINK_SPEED = 4.0         # m/s change in the ball's velocity that reads as a touch, not drift
TOUCH_RADIUS = 2.0       # metres; positions are body centres, the foot reaches about a metre
COUNT_WINDOW = 50        # frames, 2 s, for kicks per team
MAX_STREAK = 5


def kink_strength(bx, by, fps, lag=KINK_LAG):
    """|v_after - v_before| in m/s, NaN unless the ball is tracked at t - lag, t and t + lag."""
    n = len(bx)
    out = np.full(n, np.nan)
    if n > 2 * lag:
        ax = bx[2 * lag:] - 2 * bx[lag:-lag] + bx[:-2 * lag]
        ay = by[2 * lag:] - 2 * by[lag:-lag] + by[:-2 * lag]
        out[lag:-lag] = np.hypot(ax, ay) * fps / lag
    return out


def kicks(bx, by, dist, team, fps):
    """(frames, teams, strengths) of every kick, frames ascending."""
    strength = kink_strength(bx, by, fps)
    filled = np.where(np.isnan(strength), -np.inf, strength)
    padded = np.pad(filled, KINK_LAG, constant_values=-np.inf)
    local_max = np.lib.stride_tricks.sliding_window_view(padded, 2 * KINK_LAG + 1).max(axis=1)
    frames = np.flatnonzero((filled >= KINK_SPEED) & (filled >= local_max))
    d = dist[frames]
    d = np.where(np.isnan(d), np.inf, d)
    slot = np.argmin(d, axis=1) if len(frames) else np.zeros(0, dtype=np.int64)
    near = d[np.arange(len(frames)), slot] < TOUCH_RADIUS
    frames, slot = frames[near], slot[near]
    return frames, team[slot].astype(np.int8), strength[frames]


def _pick(values, i, ok, fill):
    """values[i] where ok, else fill; safe when there are no events at all."""
    if not len(values):
        return np.full(len(i), fill, dtype=np.float64)
    return np.where(ok, values[np.clip(i, 0, len(values) - 1)], fill)


def last_and_next(frames, values, n):
    """Per frame: the value of the latest event at or before it and of the first one after it,
    the frames since and until (CONTEXT_CAP where there is none), and the frame's position among
    the events (events before it: frames[:pos])."""
    index = np.arange(n)
    pos = np.searchsorted(frames, index, side="right")
    has_last, has_next = pos > 0, pos < len(frames)
    last = _pick(values, pos - 1, has_last, -1).astype(np.int8)
    nxt = _pick(values, pos, has_next, -1).astype(np.int8)
    since = np.minimum(index - _pick(frames, pos - 1, has_last, -CONTEXT_CAP), CONTEXT_CAP)
    until = np.minimum(_pick(frames, pos, has_next, n + CONTEXT_CAP) - index, CONTEXT_CAP)
    return last, since, nxt, until, pos


def streaks(teams):
    """Kicks in a row by the same team, ending at each kick and starting at each kick."""
    back = np.ones(len(teams), dtype=np.int64)
    ahead = np.ones(len(teams), dtype=np.int64)
    for i in range(1, len(teams)):
        if teams[i] == teams[i - 1]:
            back[i] = back[i - 1] + 1
    for i in range(len(teams) - 2, -1, -1):
        if teams[i] == teams[i + 1]:
            ahead[i] = ahead[i + 1] + 1
    return np.minimum(back, MAX_STREAK), np.minimum(ahead, MAX_STREAK)


def window_count(events, before, after):
    """Events in [t - before, t + after] per frame, by cumulative sums."""
    sums = np.concatenate(([0], np.cumsum(events)))
    n = len(events)
    index = np.arange(n)
    return (sums[np.clip(index + after + 1, 0, n)]
            - sums[np.clip(index - before, 0, n)]).astype(np.float64)


def touch_context(bx_raw, by_raw, dist, close, team, fps):
    """Per-frame touch signals in team indices (0 home, 1 away, -1 none).

    bx_raw / by_raw are the tracked ball with its gaps left NaN; dist is the (T, P) player-ball
    distance; close is [home, away] players within the control radius per frame."""
    n = len(bx_raw)
    frames, teams, strength = kicks(bx_raw.astype(np.float64), by_raw.astype(np.float64), dist,
                                    team, fps)
    last, since, nxt, until, pos = last_and_next(frames, teams, n)
    back, ahead = streaks(teams)
    out = {
        "kick_last": last, "kick_since": since, "kick_next": nxt, "kick_until": until,
        "kick_prev": _pick(teams, pos - 2, pos > 1, -1).astype(np.int8),
        "kick_last_streak": _pick(back, pos - 1, pos > 0, 0),
        "kick_next_streak": _pick(ahead, pos, pos < len(frames), 0),
    }
    for side in (0, 1):
        events = np.zeros(n, dtype=np.int64)
        events[frames[teams == side]] = 1
        out["kicks_past_{}".format(side)] = window_count(events, COUNT_WINDOW, 0)
        out["kicks_next_{}".format(side)] = window_count(events, 0, COUNT_WINDOW)

    exclusive = (close[0] > 0) != (close[1] > 0)
    close_frames = np.flatnonzero(exclusive)
    close_team = np.where(close[0][close_frames] > 0, 0, 1).astype(np.int8)
    c_last, c_since, c_next, c_until, _ = last_and_next(close_frames, close_team, n)
    out.update(close_last=c_last, close_since=c_since, close_next=c_next, close_until=c_until)
    return out


if __name__ == "__main__":
    import pandas as pd

    from .bench import bench_games, source_of
    from .features import SHORT_BALL_GAP, fill_short_gaps
    from .match import A, H
    from .possession import CONTROL_RADIUS
    from .sources import bar_ath

    SETTLE = 4   # frames after a kick at which its team is compared with the label
    rows = []
    for match in [bar_ath()] + bench_games():
        bx = fill_short_gaps(match.bx, SHORT_BALL_GAP)
        by = fill_short_gaps(match.by, SHORT_BALL_GAP)
        dist = np.hypot(match.px - bx[:, None], match.py - by[:, None])
        frames, teams, _ = kicks(match.bx.astype(np.float64), match.by.astype(np.float64), dist,
                                 match.team, match.fps)
        graded = match.period > 0
        close = [np.sum((dist < CONTROL_RADIUS) & (match.team == k)[None, :], axis=1)
                 for k in (0, 1)]
        ctx = touch_context(match.bx, match.by, dist, close, match.team, match.fps)
        in_graded = teams[graded[frames]]
        row = {"source": "BAR-ATH" if match.labels is None else source_of(match.name),
               "kicks_per_graded_min": len(in_graded) / (graded.sum() / match.fps / 60),
               "next_kick_same_team": np.mean(in_graded[1:] == in_graded[:-1]),
               "within_1s_of_kick": np.mean(np.minimum(ctx["kick_since"],
                                                       ctx["kick_until"])[graded] < 25)}
        if match.labels is not None:
            y = match.labels
            live = (y == H) | (y == A)
            after = np.clip(frames + SETTLE, 0, len(y) - 1)
            in_play = live[frames] & live[after]
            row["kick_team_agrees"] = np.mean(teams[in_play]
                                              == np.where(y[after[in_play]] == H, 0, 1))
            change = np.flatnonzero(live[1:] & live[:-1] & (y[1:] != y[:-1])) + 1
            row["turnovers_with_kick"] = np.mean(
                [np.any((np.abs(frames - c) <= 12) & (teams == (0 if y[c] == H else 1)))
                 for c in change])
        rows.append(row)
    table = pd.DataFrame(rows)
    print(table.groupby("source").mean(numeric_only=True).to_string(float_format="%.3f"))
