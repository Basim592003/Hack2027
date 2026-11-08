"""Is the ball track late? Ball against player velocity during close control, aggregates only.

A tracker that filters the ball causally (a forward-only filter) delays it against the players by the
filter's lag. Every boundary read off the ball would then be that many frames late on BAR-ATH but not on
the bench. During close control -- one player within CONTROL_RADIUS of the ball for at least MIN_STRETCH
frames while moving at MIN_SPEED or more -- ball and player move together, so the lag at which their
velocities agree best is the ball's delay: positive means the ball trails the player. On clean tracking it
should be about 0.

    python -m src.lagcheck        # clean external games, the current bench, BAR-ATH

Prints aggregates only: a peak lag in frames, correlations, and how many frames went into them.
"""

import numpy as np
import pandas as pd

from .bench import bench_games, source_of
from .external import labelled_games
from .idsse import idsse_games
from .noise import VERSION
from .possession import CONTROL_RADIUS
from .sources import bar_ath

MAX_LAG = 12             # frames either way
MIN_STRETCH = 25         # frames of close control by the same slot
MIN_SPEED = 3.0          # m/s: the controlling player is running, so both velocities carry signal
VELOCITY_LAG = 6         # frames, centred, the same for ball and players so neither is shifted


def velocity(x, y, fps, lag=VELOCITY_LAG):
    """Centred (vx, vy) in m/s over `lag` frames, NaN where an end is missing; (T,) or (T, P)."""
    half = lag // 2
    vx = np.full(x.shape, np.nan)
    vy = np.full(y.shape, np.nan)
    n = len(x)
    if n > lag:
        vx[half:half + n - lag] = (x[lag:] - x[:n - lag]) * fps / lag
        vy[half:half + n - lag] = (y[lag:] - y[:n - lag]) * fps / lag
    return vx, vy


def close_control(match, pvx, pvy):
    """(frames, slots) inside close-control stretches: the same slot nearest the ball, within
    CONTROL_RADIUS, moving at least MIN_SPEED, for at least MIN_STRETCH frames in a half."""
    bx, by = match.bx.astype(np.float64), match.by.astype(np.float64)
    dist = np.hypot(match.px - bx[:, None], match.py - by[:, None])
    dist = np.where(np.isnan(dist), np.inf, dist)
    rows = np.arange(match.n_frames)
    slot = np.argmin(dist, axis=1)
    speed = np.hypot(pvx[rows, slot], pvy[rows, slot])
    owner = np.where((dist[rows, slot] < CONTROL_RADIUS) & (speed >= MIN_SPEED)
                     & (match.period > 0), slot, -1)
    breaks = np.flatnonzero(owner[1:] != owner[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(owner)]))
    frames, slots = [], []
    for start, stop in zip(bounds[:-1], bounds[1:]):
        if owner[start] >= 0 and stop - start >= MIN_STRETCH:
            frames.append(np.arange(start, stop))
            slots.append(np.full(stop - start, owner[start]))
    if not frames:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(frames), np.concatenate(slots)


def lag_profile(match):
    """Cosine agreement of the ball's velocity at t with the controlling player's at t - k, pooled
    over close-control frames, for k = -MAX_LAG..MAX_LAG; and the number of frames used."""
    bvx, bvy = velocity(match.bx.astype(np.float64), match.by.astype(np.float64), match.fps)
    pvx, pvy = velocity(match.px.astype(np.float64), match.py.astype(np.float64), match.fps)
    frames, slots = close_control(match, pvx, pvy)
    lags = np.arange(-MAX_LAG, MAX_LAG + 1)
    agreement = np.full(len(lags), np.nan)
    for i, k in enumerate(lags):
        back = frames - k
        ok = (back >= 0) & (back < match.n_frames)
        t, b, s = frames[ok], back[ok], slots[ok]
        ball = np.stack([bvx[t], bvy[t]], axis=1)
        player = np.stack([pvx[b, s], pvy[b, s]], axis=1)
        good = np.isfinite(ball).all(axis=1) & np.isfinite(player).all(axis=1)
        ball, player = ball[good], player[good]
        if len(ball):
            agreement[i] = np.sum(ball * player) / np.sqrt(np.sum(ball ** 2) * np.sum(player ** 2))
    return lags, agreement, len(frames)


def peak(lags, agreement):
    """Peak lag in frames, refined by a parabola through the best lag and its neighbours."""
    i = int(np.nanargmax(agreement))
    if 0 < i < len(lags) - 1:
        left, mid, right = agreement[i - 1:i + 2]
        curvature = left - 2 * mid + right
        if curvature < 0:
            return lags[i] + 0.5 * (left - right) / curvature, mid
    return float(lags[i]), agreement[i]


def row(match, group):
    lags, agreement, used = lag_profile(match)
    lag, best = peak(lags, agreement)
    return {"group": group, "source": "BAR-ATH" if match.labels is None else source_of(match.name),
            "peak_lag_frames": lag, "agreement_at_peak": best,
            "agreement_at_0": agreement[lags == 0][0], "frames_used": used}


if __name__ == "__main__":
    rows = [row(m, "clean") for m in labelled_games() + idsse_games()]
    rows += [row(m, "bench v{}".format(VERSION)) for m in bench_games(levels=(0.10,))]
    rows.append(row(bar_ath(), "BAR-ATH"))
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print(table.groupby(["group", "source"])[["peak_lag_frames", "agreement_at_peak",
                                              "agreement_at_0", "frames_used"]]
          .median().to_string(float_format="%.3f"))
