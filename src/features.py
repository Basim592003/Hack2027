"""Per-frame signals from a Match: the same code for BAR-ATH and for the labelled external games.

Everything is computed from positions alone -- the BAR-ATH ATD carries a speed per slot but the
external games do not, and a feature must mean the same thing on both. Speeds are displacements
over a window rather than frame differences, because frame-level position jitter (~0.15 m) would
otherwise read as several metres per second.

    feats = frame_features(match)      # {name: (T,) float32}, NaN where undefined
    names, matrix = as_matrix(feats)   # (T, F) for a model

Nothing here is fitted. The geometry (6-yard box, 9.15 m, centre spot) is the Laws of the Game;
the window lengths are round numbers of seconds.

A dead ball is not always a missing or a still one: while play is stopped it is fetched, carried
and thrown back (src/noise.py, v4). Two signals hold whatever it does meanwhile:
  out_not_restarted  the ball was seen beyond a line more recently than the last kick
                     (src/touches.py) -- once out, play stays dead until somebody plays it again.
  nearest_speed      how fast the player nearest the ball moves: a ball carried at walking pace
                     is not a dribble.

Behind EVENTS (sprint item G3), event_features adds what a restart looks like: when the last and
next kicks were, how hard they were and how still the ball was in the 2 s before each (a restart
is a kick after a still ball), how many kicks there are around, and the signals in EVENT_POOLED
averaged over the stretch between events (src/segments.py) and the stretches either side, so a
stoppage can end at its restart kick rather than wherever a window average happens to cross.
"""

import numpy as np

# Geometry, Laws of the Game, metres.
GOAL_AREA_DEPTH = 5.5
GOAL_AREA_HALF_WIDTH = 9.16
FREE_KICK_DISTANCE = 9.15
NEAR_CORNER = 2.5
NEAR_TOUCHLINE = 1.5
NEAR_CENTRE = 2.0
OFF_PITCH = 1.0          # metres beyond a line: the whole ball is over, with tracking error to spare

SHORT_BALL_GAP = 12      # frames; gaps this short are interpolated before computing ball speed
BALL_SPEED_LAG = 6       # frames between the two positions a ball speed is measured over
PLAYER_SPEED_LAG = 24    # frames, ~1 s, for player speeds
SLOW, FAST = 1.5, 4.0    # m/s: walking, running
CONTEXT_CAP = 1500       # frames; "how long since the ball was seen" saturates at a minute

# (name, frames before, frames after) context windows, at 25 fps.
WINDOWS = (("c1", 12, 12), ("c4", 50, 50), ("c10", 125, 125), ("past4", 100, 0),
           ("next4", 0, 100))
WINDOWED = ("ball_seen", "ball_speed", "ball_still", "mean_speed", "frac_slow", "frac_fast",
            "n_visible", "boundary_dist", "nearest_dist", "n_within_9m", "own_half_min",
            "nearest_speed")

EVENTS = False           # sprint item G3: kick and segment signals (event_features; lost, off)
RESTART_STILL = 50       # frames, 2 s, before a kick over which the ball's stillness is measured
EVENT_POOLED = ("ball_seen", "ball_still", "ball_speed", "mean_speed", "frac_slow",
                "nearest_speed")


def rolling_mean(values, before, after):
    """NaN-aware mean of values over [t - before, t + after], by cumulative sums."""
    valid = ~np.isnan(values)
    sums = np.concatenate(([0.0], np.cumsum(np.where(valid, values, 0.0), dtype=np.float64)))
    counts = np.concatenate(([0], np.cumsum(valid)))
    n = len(values)
    index = np.arange(n)
    low = np.clip(index - before, 0, n)
    high = np.clip(index + after + 1, 0, n)
    count = counts[high] - counts[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, (sums[high] - sums[low]) / count, np.nan)


def fill_short_gaps(values, max_gap):
    """Linear interpolation across NaN runs no longer than max_gap, leaving longer gaps NaN."""
    out = values.astype(np.float64).copy()
    missing = np.isnan(out)
    edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if stop - start <= max_gap and start > 0 and stop < len(out):
            out[start:stop] = np.interp(np.arange(start, stop), [start - 1, stop],
                                        [out[start - 1], out[stop]])
    return out


def lagged_speed(x, y, lag, fps):
    """|p(t + lag/2) - p(t - lag/2)| / (lag / fps), NaN where either end is missing."""
    half = lag // 2
    out = np.full(x.shape, np.nan, dtype=np.float64)
    n = len(x)
    if n <= lag:
        return out
    dx = x[lag:] - x[:n - lag]
    dy = y[lag:] - y[:n - lag]
    out[half:half + n - lag] = np.hypot(dx, dy) / (lag / fps)
    return out


def nan_row_mean(values):
    valid = ~np.isnan(values)
    count = valid.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, np.where(valid, values, 0).sum(axis=1) / count, np.nan)


def team_sides(match):
    """(T,) per team: -1 where it defends the x=0 goal, +1 where it defends x=length.

    Decided per half from the median x of each team's centroid: the team whose shape sits nearer
    a goal over a whole half is the one defending it. Relative, so it holds when one side
    dominates and both shapes push upfield.
    """
    sides = np.zeros((2, match.n_frames), dtype=np.int8)
    for half in np.unique(match.period[match.period > 0]):
        frames = match.period == half
        centroid = [np.nanmedian(nan_row_mean(np.where(match.team == k, match.px[frames], np.nan)))
                    for k in (0, 1)]
        home_left = centroid[0] < centroid[1]
        sides[0, frames] = -1 if home_left else 1
        sides[1, frames] = 1 if home_left else -1
    # Frames outside a half take the side of the nearest half, so features stay defined there.
    for k in (0, 1):
        known = np.flatnonzero(sides[k] != 0)
        if len(known):
            nearest = known[np.clip(np.searchsorted(known, np.arange(match.n_frames)), 0,
                                    len(known) - 1)]
            sides[k] = sides[k][nearest]
    return sides


def ball_geometry(bx, by, pitch):
    """Distances and flags describing where the ball is relative to the pitch markings."""
    length, width = pitch
    boundary = np.minimum(np.minimum(bx, length - bx), np.minimum(by, width - by))
    corner = np.min([np.hypot(bx - cx, by - cy) for cx in (0, length) for cy in (0, width)],
                    axis=0)
    goal_area = (np.minimum(bx, length - bx) < GOAL_AREA_DEPTH) & \
        (np.abs(by - width / 2) < GOAL_AREA_HALF_WIDTH)
    nan = np.isnan(bx)
    return {
        "boundary_dist": boundary,
        "near_corner": np.where(nan, np.nan, corner < NEAR_CORNER),
        "near_touchline": np.where(nan, np.nan, np.minimum(by, width - by) < NEAR_TOUCHLINE),
        "in_goal_area": np.where(nan, np.nan, goal_area),
        "near_centre": np.where(nan, np.nan,
                                np.hypot(bx - length / 2, by - width / 2) < NEAR_CENTRE),
    }


def at_events(values, i):
    """values[i] where 0 <= i < len(values), NaN elsewhere (also when there are no events)."""
    ok = (i >= 0) & (i < len(values))
    out = np.full(len(i), np.nan)
    out[ok] = values[i[ok]]
    return out


def event_features(feats, kick_frames, kick_teams, kick_strength, seen, period):
    """Kick timing and segment means for the dead-ball model, from frame_features' own signals.
    The events are kicks, ball-gap edges and half edges: close-team changes say nothing about a
    stoppage, and predicted stoppage edges would come from this very model."""
    from . import segments  # segments imports this module
    from .touches import KINK_LAG
    n = len(seen)
    index = np.arange(n)
    out = {}
    pos = np.searchsorted(kick_frames, index, side="right")
    out["log_since_kick"] = np.log1p(np.fmin(index - at_events(kick_frames, pos - 1), CONTEXT_CAP))
    out["log_until_kick"] = np.log1p(np.fmin(at_events(kick_frames, pos) - index, CONTEXT_CAP))
    still = rolling_mean(feats["ball_still"], RESTART_STILL, 0)
    still_before = still[np.maximum(kick_frames - KINK_LAG, 0)]
    for name, values in (("strength", kick_strength), ("still_before", still_before)):
        out["last_kick_" + name] = at_events(values, pos - 1)
        out["next_kick_" + name] = at_events(values, pos)
    marks = np.zeros(n)
    marks[kick_frames] = 1.0
    for tag, before, after in WINDOWS:
        out["kick_rate_" + tag] = rolling_mean(marks, before, after)

    touch = {"kick_frames": kick_frames, "kick_teams": kick_teams,
             "close_change_frames": np.zeros(0, dtype=np.int64),
             "close_change_teams": np.zeros(0, dtype=np.int8)}
    frames, kinds, teams = segments.events(touch, seen, np.zeros(n, dtype=bool), period)
    layout = segments.layout(frames, kinds, teams, seen)
    for name in ["seg_log_len", "seg_pos"] + ["seg_{}_{}".format(edge, kind) for edge in
                                             ("start", "end") for kind in ("kick", "gap", "stop")]:
        out[name] = layout[name]
    for name in EVENT_POOLED:
        own, before, after = segments.pool(np.asarray(feats[name], dtype=np.float64), frames)
        out["seg_" + name] = own
        out["seg_{}_prev".format(name)] = before
        out["seg_{}_next".format(name)] = after
        out["seg_{}_from_prev".format(name)] = own - before
        out["seg_{}_to_next".format(name)] = after - own
    return out


def frame_features(match):
    fps = match.fps
    n = match.n_frames
    length, width = match.pitch
    feats = {}

    # -- ball ------------------------------------------------------------------------------
    seen = ~np.isnan(match.bx)
    bx = fill_short_gaps(match.bx, SHORT_BALL_GAP)
    by = fill_short_gaps(match.by, SHORT_BALL_GAP)
    feats["ball_seen"] = seen.astype(np.float64)
    ball_speed = lagged_speed(bx, by, BALL_SPEED_LAG, fps)
    feats["ball_speed"] = ball_speed
    feats["ball_still"] = np.where(np.isnan(ball_speed), np.nan, ball_speed < 1.0)
    geometry = ball_geometry(bx, by, match.pitch)
    feats.update(geometry)

    # How long the ball has been gone, and where it was last seen and next seen.
    index = np.arange(n)
    last = np.maximum.accumulate(np.where(seen, index, -1))
    following = np.minimum.accumulate(np.where(seen, index, n)[::-1])[::-1]
    since = np.where(last >= 0, index - last, CONTEXT_CAP)
    until = np.where(following < n, following - index, CONTEXT_CAP)
    feats["log_since_seen"] = np.log1p(np.minimum(since, CONTEXT_CAP))
    feats["log_until_seen"] = np.log1p(np.minimum(until, CONTEXT_CAP))
    feats["log_gap_len"] = np.log1p(np.minimum(np.where(seen, 0, since + until - 1), CONTEXT_CAP))
    last_c = np.clip(last, 0, n - 1)
    next_c = np.clip(following, 0, n - 1)
    for name in ("boundary_dist", "near_corner", "near_touchline", "in_goal_area", "near_centre"):
        feats["last_" + name] = np.where(last >= 0, geometry[name][last_c], np.nan)
        feats["next_" + name] = np.where(following < n, geometry[name][next_c], np.nan)
    feats["last_ball_speed"] = np.where(last >= 0, ball_speed[last_c], np.nan)
    feats["next_ball_speed"] = np.where(following < n, ball_speed[next_c], np.nan)

    # -- players ---------------------------------------------------------------------------
    px, py = match.px.astype(np.float64), match.py.astype(np.float64)
    visible = ~np.isnan(px)
    feats["n_visible"] = visible.sum(axis=1).astype(np.float64)
    speed = np.stack([lagged_speed(px[:, k], py[:, k], PLAYER_SPEED_LAG, fps)
                      for k in range(px.shape[1])], axis=1)
    feats["mean_speed"] = nan_row_mean(speed)
    feats["frac_slow"] = nan_row_mean(np.where(np.isnan(speed), np.nan, speed < SLOW))
    feats["frac_fast"] = nan_row_mean(np.where(np.isnan(speed), np.nan, speed > FAST))

    dist = np.hypot(px - bx[:, None], py - by[:, None])
    no_ball = np.isnan(bx) | np.isnan(dist).all(axis=1)
    nearest = np.argmin(np.where(np.isnan(dist), np.inf, dist), axis=1)
    feats["nearest_dist"] = np.where(no_ball, np.nan, dist[index, nearest])
    feats["nearest_speed"] = np.where(no_ball, np.nan, speed[index, nearest])
    feats["n_within_9m"] = np.where(np.isnan(bx), np.nan,
                                    np.sum(dist < FREE_KICK_DISTANCE, axis=1))
    del speed

    # Out of play until restarted. An off-pitch frame within KINK_LAG of the last kick is the
    # restart itself (a throw-in leaves from behind the line), not the ball going out again.
    from .touches import KINK_LAG, kicks  # touches imports this module
    kick_frames, kick_teams, kick_strength = kicks(match.bx.astype(np.float64),
                                                   match.by.astype(np.float64), dist, match.team, fps)
    del dist
    off = np.nan_to_num(geometry["boundary_dist"], nan=np.inf) < -OFF_PITCH
    off &= seen
    last_off = np.maximum.accumulate(np.where(off, index, -1))
    kick_mark = np.full(n, -1)
    kick_mark[kick_frames] = kick_frames
    last_kick = np.maximum.accumulate(kick_mark)
    feats["out_not_restarted"] = ((last_off >= 0) &
                                  ((last_kick < 0) | (last_off > last_kick + KINK_LAG))
                                  ).astype(np.float64)
    feats["log_since_off_pitch"] = np.log1p(np.minimum(
        np.where(last_off >= 0, index - last_off, CONTEXT_CAP), CONTEXT_CAP))
    if EVENTS:
        feats.update(event_features(feats, kick_frames, kick_teams, kick_strength, seen,
                                    match.period))

    # Kick-off shape: every player of both teams in their own half.
    sides = team_sides(match)
    own = []
    for k in (0, 1):
        on_own = np.where(visible & (match.team == k)[None, :],
                          (px - length / 2) * sides[k][:, None] > 0, np.nan)
        own.append(nan_row_mean(on_own))
    feats["own_half_min"] = np.fmin(own[0], own[1])

    # -- context windows -------------------------------------------------------------------
    for name in WINDOWED:
        for tag, before, after in WINDOWS:
            feats["{}_{}".format(name, tag)] = rolling_mean(feats[name], before, after)
    for name in ("mean_speed", "ball_seen", "frac_fast"):
        feats[name + "_next_minus_past"] = feats[name + "_next4"] - feats[name + "_past4"]

    return {k: np.asarray(v, dtype=np.float32) for k, v in feats.items()}


def as_matrix(feats, names=None):
    names = sorted(feats) if names is None else names
    return names, np.stack([feats[k] for k in names], axis=1)
