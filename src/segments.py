"""Events: the frames where the ball can change hands, and the segments of play between them.

The team model's ball features are averages over fixed 1-4 s windows, which straddle a hand-over
and blur it (src/touches.py). But a hand-over happens at a touch, in a duel or out of sight, and
each of those leaves a mark in the tracking, so the evidence for each team can be pooled between
the marks instead. An event is any of

  kick    a kick of the ball, owned by the nearest player's team (src/touches.py)
  close   the team alone within the control radius changes; owned by the new team
  gap     the tracked ball disappears or reappears: raw gaps, before any interpolation
  stop    a predicted stoppage starts or ends (src/deadball.py), or a half does

Events within MERGE frames of a group's first event join that group and become one event at its
first frame. Its kinds are flags, and its team is the team of its kicks and close changes where
they agree, -1 where they disagree or it has neither. A group never spans more than MERGE frames,
so a run of duel flicker is cut every few frames instead of collapsing into its first frame.

What they catch (noise v5 bench, 2026-11-08, out-of-fold stoppages; IDSSE / Metrica, with BAR-ATH
as a sanity check only). There are 17.6 / 23.3 events per graded minute, against 17.4 on BAR-ATH:
kicks 8.8 / 14.5 (8.9), close changes 5.5 / 6.9 (5.8), gap edges 3.1 / 3.3 (2.7) and stoppage
edges 2.1 / 1.9 (2.0). An event lies within 12 frames of 65% / 82% of real controlled in-play
changes (58% / 77% within 6), and one owned by the new team lies within 12 of 47% / 62%. Windows
of +/-12 frames cover 25% / 33% of graded frames (BAR-ATH 25%). Kicks catch the most: 46% / 55%
on their own. Recall falls with ball smoothing, because the filter flattens kicks below
KINK_SPEED: on IDSSE copies it is 83%, 65% and 50% at sd < 4, 4-7 and >= 7 frames. BAR-ATH's kick
and event rates match IDSSE's sd 4-7 copies (9.2 and 17.8 a minute), not its sd >= 7 ones (4.3
and 13.0).

Gap edges catch 6% / 31%. IDSSE's clean ball is never missing in play, and noise v5 drops it at
random. The clean Metrica sample, though, loses its ball on 11% of live frames, and those gaps
start or end within 12 frames of 29% of its hand-overs, against 5.5% for the same number of
random live frames. Whether BAR-ATH's gaps (same vendor) behave like Metrica's or fall at random
is unknown.

    python -m src.segments    # recall of real in-play changes on the bench; rates on BAR-ATH
"""

import numpy as np

from . import touches
from .features import SHORT_BALL_GAP, fill_short_gaps

MERGE = 3                 # frames: an event this close to a group's first event joins the group
KICK, CLOSE, GAP, STOP = 1, 2, 4, 8
KINDS = (("kick", KICK), ("close", CLOSE), ("gap", GAP), ("stop", STOP))


def edges(values):
    """Frames whose value differs from the previous frame's."""
    return np.flatnonzero(np.diff(values.astype(np.int8)) != 0) + 1


def events(touch, ball_seen, dead, period):
    """(frames, kinds, teams) of every event, frames ascending, teams in team indices.

    touch is touches.touch_context's output, ball_seen the tracked ball before interpolation, dead
    the predicted stoppages, period the match's halves."""
    parts = ((touch["kick_frames"], KICK, touch["kick_teams"]),
             (touch["close_change_frames"], CLOSE, touch["close_change_teams"]),
             (edges(ball_seen), GAP, None),
             (np.concatenate((edges(dead), edges(period))), STOP, None))
    frames = np.concatenate([f for f, _, _ in parts]).astype(np.int64)
    kinds = np.concatenate([np.full(len(f), kind, dtype=np.int8) for f, kind, _ in parts])
    teams = np.concatenate([np.full(len(f), -1, dtype=np.int8) if t is None else t.astype(np.int8)
                            for f, _, t in parts])
    if not len(frames):
        return frames, kinds, teams
    order = np.argsort(frames, kind="stable")
    frames, kinds, teams = frames[order], kinds[order], teams[order]
    first, start = [], None
    for i, frame in enumerate(frames.tolist()):
        if start is None or frame - start > MERGE:
            first.append(i)
            start = frame
    first = np.array(first, dtype=np.int64)
    low = np.minimum.reduceat(np.where(teams >= 0, teams, 2), first)
    high = np.maximum.reduceat(teams, first)
    return (frames[first], np.bitwise_or.reduceat(kinds, first),
            np.where(low == high, low, -1).astype(np.int8))


def match_events(match, dead):
    """events() for a match, from the same ball, distances and touch signals as
    src/possession.py's match context. possession is imported here, not at the top, so that it can
    import this module."""
    from .possession import CONTROL_RADIUS
    bx = fill_short_gaps(match.bx, SHORT_BALL_GAP)
    by = fill_short_gaps(match.by, SHORT_BALL_GAP)
    dist = np.hypot(match.px.astype(np.float64) - bx[:, None],
                    match.py.astype(np.float64) - by[:, None])
    close = [np.sum((dist < CONTROL_RADIUS) & (match.team == k)[None, :], axis=1) for k in (0, 1)]
    touch = touches.touch_context(match.bx, match.by, dist, close, match.team, match.fps)
    return events(touch, ~np.isnan(match.bx), dead, match.period)


def within(frames, targets, k):
    """Per target frame, whether an event lies within k frames of it (frames ascending)."""
    if not len(frames):
        return np.zeros(len(targets), dtype=bool)
    at = np.searchsorted(frames, targets - k)
    return (at < len(frames)) & (frames[np.minimum(at, len(frames) - 1)] <= targets + k)


if __name__ == "__main__":
    import os

    import pandas as pd

    from . import deadball, experiment, possession
    from .bench import bench_games, source_of
    from .match import CACHE_DIR, H
    from .sources import bar_ath

    WITHIN = (6, 12)      # frames either side
    games = bench_games()
    dead_tag = experiment.dead_oof(games)
    rows = []
    for match in [bar_ath()] + games:
        bar = match.labels is None
        p_dead = np.load(os.path.join(CACHE_DIR, "final_pdead_BAR-ATH.npy") if bar
                         else deadball.oof_path(match.name, dead_tag))
        frames, kinds, teams = match_events(
            match, deadball.decode(p_dead, match.period, possession.DEAD_PENALTY))
        graded = match.period > 0
        minutes = graded.sum() / match.fps / 60
        inside = graded[frames]
        row = {"source": "BAR-ATH" if bar else source_of(match.name),
               "ball_sd": np.nan if bar else experiment.ball_sd(match.name),
               "events/min": inside.sum() / minutes,
               "median spacing (frames)": np.median(np.diff(frames[inside])),
               "events with a team": np.mean(teams[inside] >= 0)}
        for name, flag in KINDS:
            row[name + "/min"] = np.sum(inside & ((kinds & flag) > 0)) / minutes
        for k in WITHIN:
            row["graded frames within {}".format(k)] = within(frames, np.flatnonzero(graded),
                                                              k).mean()
        if not bar:
            for suffix, relabel in experiment.CONVENTIONS:
                truth = match.labels if relabel is None else relabel(match.labels)
                changes, new = experiment.in_play_changes(truth, experiment.pm.valid_runs(truth))
                new = np.where(new == H, 0, 1)
                for k in WITHIN:
                    row["recall {} {}".format(k, suffix)] = within(frames, changes, k).mean()
                right = np.zeros(len(changes), dtype=bool)
                for team in (0, 1):
                    right[new == team] = within(frames[teams == team], changes[new == team], 12)
                row["recall 12 new team " + suffix] = right.mean()
                if suffix == "ctl":
                    for name, flag in KINDS:
                        row["recall 12 {} ctl".format(name)] = within(
                            frames[(kinds & flag) > 0], changes, 12).mean()
        rows.append(row)

    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("== events per graded minute; recall: share of real in-play changes with an event "
          "within k frames")
    print(table.groupby("source").mean(numeric_only=True).drop(columns="ball_sd").T
          .to_string(float_format="%.3f"))
    bench = table[table.source != "BAR-ATH"]
    band = pd.cut(bench.ball_sd, experiment.SMOOTHING_BANDS, right=False)
    grouped = bench.groupby(["source", band], observed=True)
    by_band = grouped[["events/min", "kick/min", "recall 6 ctl", "recall 12 ctl",
                       "recall 12 new team ctl"]].mean()
    by_band.insert(0, "copies", grouped.size())
    print("== bench by ball smoothing (frames)")
    print(by_band.T.to_string(float_format="%.3f"))
