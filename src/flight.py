"""Flight target: while the ball travels, which team gets to it first.

A turnover in play is mostly a pass that the other team reaches first. During the flight the
averaged ball features still describe the passer, and the touch features only learn who got there
once someone touches the ball. This reads it from the flight itself, with round physical numbers
and nothing fitted:

  * while the ball moves faster than MIN_SPEED, project it in a straight line, b + v tau, for tau
    in TAUS, and stop where it would cross a line
  * a player starts REACTION seconds later from where he would be by then at his current velocity,
    runs at RUN_SPEED and reaches REACH metres; he is in time for the ball at tau if
    REACTION + max(|b + v tau - (x + REACTION v_x)| - REACH, 0) / RUN_SPEED <= tau
  * a team's arrival time is the first tau with one of its players in time, NOBODY if none is

The path ignores the ball slowing down and rising, so the arrival times are rough. They are
evidence, not an answer: src/possession.py turns them into each team's view, and the team model
learns from the external games how far to trust them. NaN means the ball is not in flight. In a
ball gap the values of the last frame with a measured ball velocity are carried on, so a pass
that disappears into the air keeps its prediction until the ball is seen again.

    python -m src.flight     # how often the ball is in flight, and where passes go, bench vs BAR-ATH
"""

import numpy as np

MIN_SPEED = 5.0                     # m/s: slower balls are carried or rolling, not passed
TAUS = np.arange(1, 26) * 0.12      # s: 0.12 to 3.0, every 3 frames at 25 fps
REACTION = 0.7                      # s
RUN_SPEED = 5.0                     # m/s
REACH = 1.0                         # m
NOBODY = 3.5                        # s: arrival time when no player arrives within TAUS
CHUNK = 2000                        # frames per block of the (frames, players, taus) arrays


def arrivals(bx, by, bvx, bvy, px, py, pvx, pvy, team, pitch):
    """(times, out): each team's arrival time in seconds, (T, 2) in team indices, and the time the
    projected ball leaves the pitch, (T,), NOBODY if it stays on. NaN where the ball is not in
    flight. Player velocities that are unknown count as standing still."""
    n = len(bx)
    length, width = pitch
    times = np.full((n, 2), np.nan)
    out = np.full(n, np.nan)
    speed = np.hypot(bvx, bvy)
    flying = np.flatnonzero(np.nan_to_num(speed) > MIN_SPEED)
    teams = [team == k for k in (0, 1)]
    for start in range(0, len(flying), CHUNK):
        rows = flying[start:start + CHUNK]
        path_x = bx[rows, None] + bvx[rows, None] * TAUS[None, :]           # (F, K)
        path_y = by[rows, None] + bvy[rows, None] * TAUS[None, :]
        outside = (path_x < 0) | (path_x > length) | (path_y < 0) | (path_y > width)
        first_out = np.where(outside.any(axis=1), np.argmax(outside, axis=1), len(TAUS))
        on_pitch = np.arange(len(TAUS))[None, :] < first_out[:, None]     # (F, K)
        out[rows] = np.where(first_out < len(TAUS), TAUS[np.minimum(first_out, len(TAUS) - 1)],
                             NOBODY)
        start_x = px[rows] + REACTION * np.nan_to_num(pvx[rows])            # (F, P)
        start_y = py[rows] + REACTION * np.nan_to_num(pvy[rows])
        gap = np.hypot(path_x[:, None, :] - start_x[:, :, None],
                       path_y[:, None, :] - start_y[:, :, None])           # (F, P, K)
        need = REACTION + np.maximum(gap - REACH, 0.0) / RUN_SPEED
        in_time = (need <= TAUS[None, None, :]) & on_pitch[:, None, :]    # NaN players: False
        for k in (0, 1):
            reached = in_time[:, teams[k], :].any(axis=1)                  # (F, K)
            times[rows, k] = np.where(reached.any(axis=1), TAUS[np.argmax(reached, axis=1)],
                                      NOBODY)
    defined = ~np.isnan(speed)
    last = np.maximum.accumulate(np.where(defined, np.arange(n), -1))
    carry = ~defined & (last >= 0)
    times[carry] = times[last[carry]]
    out[carry] = out[last[carry]]
    return times, out


if __name__ == "__main__":
    import pandas as pd

    from .bench import bench_games, source_of
    from .features import SHORT_BALL_GAP, fill_short_gaps
    from .possession import VELOCITY_LAG, lagged_delta
    from .sources import bar_ath
    from .touches import kicks

    rows = []
    for match in [bar_ath()] + bench_games():
        fps = match.fps
        bx = fill_short_gaps(match.bx, SHORT_BALL_GAP)
        by = fill_short_gaps(match.by, SHORT_BALL_GAP)
        px, py = match.px.astype(np.float64), match.py.astype(np.float64)
        velocity = [lagged_delta(v, VELOCITY_LAG) * fps / VELOCITY_LAG for v in (bx, by, px, py)]
        times, out = arrivals(bx, by, velocity[0], velocity[1], px, py, velocity[2], velocity[3],
                              match.team, match.pitch)
        graded = match.period > 0
        dist = np.hypot(px - bx[:, None], py - by[:, None])
        kick_frames, kick_teams, _ = kicks(match.bx.astype(np.float64),
                                           match.by.astype(np.float64), dist, match.team, fps)
        # At each kick, after the ball has had VELOCITY_LAG frames to leave the foot: does the kicker's
        # team get there first? In real play most passes arrive, so this should be well above half.
        after = np.minimum(kick_frames + VELOCITY_LAG, match.n_frames - 1)
        decided = ~np.isnan(times[after, 0]) & (times[after, 0] != times[after, 1])
        first = np.where(times[after, 0] < times[after, 1], 0, 1)
        in_flight = ~np.isnan(times[:, 0]) & graded
        rows.append({
            "source": "BAR-ATH" if match.labels is None else source_of(match.name),
            "graded frames in flight": in_flight.mean() / graded.mean(),
            "kicks followed by a flight": decided.mean(),
            "kicker's team first": np.mean(first[decided] == kick_teams[decided]),
            "nobody in time": np.mean((times[in_flight] == NOBODY).all(axis=1)),
            "goes out first": np.mean(out[in_flight] < np.nanmin(times[in_flight], axis=1)),
        })
    print(pd.DataFrame(rows).groupby("source").mean().T.to_string(float_format="%.3f"))
