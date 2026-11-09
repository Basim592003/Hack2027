"""Pressure and team shape: who is being closed down, and which team is spreading out.

Two signals that need no ball, for the frames without one (41% on BAR-ATH):

  pressure    the closing speed of each pair of opponents within RADIUS of each other,
              -dx.dv / max(d, MIN_DISTANCE), counted only while they close in, and summed over a
              player's opponents. The player most closed down is usually on the ball, so his team
              usually has it; also how many opponents are within NEAR of him.
  divergence  each team's velocity field fitted as v = A x + b over its tracked players (at least
              MIN_PLAYERS, a small RIDGE), and trace(A): positive when the team spreads out, as a
              team that has just won the ball does, negative when it contracts.

Slots are found by team, never by position in the array. Arrays of (frames, home players, away
players) are built CHUNK frames at a time. Nothing is fitted; the radii are round numbers.

    python -m src.pressure    # how often the most-pressed team is the one nearest the ball, bench vs BAR-ATH
"""

import numpy as np

RADIUS = 10.0          # m: opponents further apart than this are not pressing each other
NEAR = 5.0             # m
MIN_DISTANCE = 0.5     # m
MIN_PLAYERS = 4
RIDGE = 1.0            # m^2
CHUNK = 5000


def pressure(px, py, pvx, pvy, team):
    """(most, most_team, near): per frame, each team's largest pressure on one of its players
    (T, 2), the team of the player most pressed (-1 when nobody is), and the opponents within NEAR
    of that player. NaN where a team has no tracked player."""
    n = len(px)
    home, away = np.flatnonzero(team == 0), np.flatnonzero(team == 1)
    most = np.full((n, 2), np.nan)
    most_team = np.full(n, -1, dtype=np.int8)
    near = np.full(n, np.nan)
    for start in range(0, n, CHUNK):
        rows = slice(start, min(start + CHUNK, n))
        dx = px[rows][:, None, away] - px[rows][:, home, None]           # (F, home, away)
        dy = py[rows][:, None, away] - py[rows][:, home, None]
        dvx = pvx[rows][:, None, away] - pvx[rows][:, home, None]
        dvy = pvy[rows][:, None, away] - pvy[rows][:, home, None]
        d = np.hypot(dx, dy)
        closing = -(dx * dvx + dy * dvy) / np.maximum(d, MIN_DISTANCE)
        closing = np.where((d < RADIUS) & (closing > 0), closing, 0.0)    # NaN compares False
        on = [closing.sum(axis=2), closing.sum(axis=1)]                    # (F, home), (F, away)
        seen = [~np.isnan(px[rows][:, home]), ~np.isnan(px[rows][:, away])]
        for k in (0, 1):
            most[rows, k] = np.where(seen[k].any(axis=1),
                                     np.where(seen[k], on[k], -1.0).max(axis=1), np.nan)
        top = np.nan_to_num(most[rows], nan=0.0)
        team_rows = np.where(top.max(axis=1) <= 0, -1, np.where(top[:, 0] >= top[:, 1], 0, 1))
        most_team[rows] = team_rows
        frames = np.arange(rows.stop - rows.start)
        close = d < NEAR
        home_near = close[frames, np.argmax(on[0], axis=1), :].sum(axis=1)
        away_near = close[frames, :, np.argmax(on[1], axis=1)].sum(axis=1)
        near[rows] = np.where(team_rows == 0, home_near, np.where(team_rows == 1, away_near, np.nan))
    return most, most_team, near


def divergence(px, py, pvx, pvy, team):
    """(T, 2): trace(A) of each team's least-squares velocity field v = A x + b, NaN with fewer
    than MIN_PLAYERS tracked players with a velocity."""
    out = np.full((len(px), 2), np.nan)
    for k in (0, 1):
        slots = np.flatnonzero(team == k)
        x, y, vx, vy = px[:, slots], py[:, slots], pvx[:, slots], pvy[:, slots]
        ok = ~(np.isnan(x) | np.isnan(vx))
        count = ok.sum(axis=1)

        def centred(values):
            values = np.where(ok, values, 0.0)
            mean = values.sum(axis=1) / np.maximum(count, 1)
            return np.where(ok, values - mean[:, None], 0.0)

        x, y, vx, vy = centred(x), centred(y), centred(vx), centred(vy)
        sxx, syy, sxy = (x * x).sum(1) + RIDGE, (y * y).sum(1) + RIDGE, (x * y).sum(1)
        cxx, cxy, cyx, cyy = (vx * x).sum(1), (vx * y).sum(1), (vy * x).sum(1), (vy * y).sum(1)
        trace = (cxx * syy - (cxy + cyx) * sxy + cyy * sxx) / (sxx * syy - sxy * sxy)
        out[:, k] = np.where(count >= MIN_PLAYERS, trace, np.nan)
    return out


if __name__ == "__main__":
    import pandas as pd

    from .bench import bench_games, nearest_team, source_of
    from .possession import VELOCITY_LAG, lagged_delta
    from .sources import bar_ath

    rows = []
    for match in [bar_ath()] + bench_games():
        fps = match.fps
        px, py = match.px.astype(np.float64), match.py.astype(np.float64)
        pvx = lagged_delta(px, VELOCITY_LAG) * fps / VELOCITY_LAG
        pvy = lagged_delta(py, VELOCITY_LAG) * fps / VELOCITY_LAG
        most, most_team, near = pressure(px, py, pvx, pvy, match.team)
        div = divergence(px, py, pvx, pvy, match.team)
        graded = match.period > 0
        nearest = nearest_team(match)
        both = graded & (most_team >= 0) & (nearest >= 0)
        rows.append({
            "source": "BAR-ATH" if match.labels is None else source_of(match.name),
            "graded frames with a most-pressed player": np.mean(most_team[graded] >= 0),
            "most-pressed team is nearest the ball": np.mean(most_team[both] == nearest[both]),
            "median top pressure (m/s)": np.nanmedian(np.nanmax(most[graded], axis=1)),
            "median opponents within 5 m": np.nanmedian(near[graded]),
            "divergence sd (1/s)": np.nanstd(div[graded]),
        })
    print(pd.DataFrame(rows).groupby("source").mean().T.to_string(float_format="%.3f"))
