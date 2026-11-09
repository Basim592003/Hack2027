"""Labelled games other than BAR-ATH: the only place constants and model weights may be set.

Metrica Sports sample data (https://github.com/metrica-sports/sample-data), three anonymised
matches with tracking and human-annotated event data, cloned into data/external/metrica-sample.
The events are human-labelled -- on other games, which RULES.md allows -- and are turned into
frame labels by a fixed algorithm, labels_from_events, written to approximate the challenge's
definitions rather than tuned against anything:

  * alive, owner: the team of the most recent on-ball event (pass, carry, recovery, shot, ball
    lost, ball out, set piece). Challenges are contested and do not move possession, which is the
    CHALLENGE.md rule that a touch without control is not a change of possession.
  * dead: from the event that stopped play (ball out, foul, offside, goal) until the set piece
    that restarts it. Every restart ends a stoppage, so every set piece opens one.
  * dead, too: a stoppage nobody annotated (unannotated_stoppages). The sample's tracking blanks
    the ball through every stoppage. Where the events call play live but the ball is gone for at
    least UNANNOTATED_STOPPAGE frames and no event of any type starts, play has stopped. All 47
    such stretches (31k frames, about 7 minutes a game) have no event at all, and the players walk
    through them (mean 1.26 m/s, under 1.5 m/s on 75% of frames, like annotated stoppages at
    1.02 m/s and 87%, against 2.18 m/s and 22% in live play). Shorter gaps contain an event 39%
    of the time, as live play does. Found 2026-11-09, sprint item D3.
  * X before each half's kick-off and after its last event, as the BAR-ATH ground truth does.

    python -m src.external            # converts, caches and prints label statistics
"""

import json
import os

import numpy as np
import pandas as pd

from . import epts
from .match import A, D, H, X, Match, cached

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICA_DIR = os.path.join(ROOT, "data", "external", "metrica-sample", "data")
METRICA_PITCH = (105.0, 68.0)
METRICA_GAMES = (1, 2, 3)

POSSESSION_TYPES = {"PASS", "CARRY", "RECOVERY", "SHOT", "BALL LOST", "BALL OUT", "SET PIECE"}
# Events whose end frame is where the ball went dead (it travelled there); for the rest -- fouls,
# offsides, handballs -- play stops where the event happens.
TRAVELLING_TYPES = {"PASS", "SHOT", "BALL OUT", "BALL LOST", "CARRY"}
IGNORED_BEFORE_RESTART = {"CARD", "SET PIECE"}
UNANNOTATED_STOPPAGE = 100   # frames (4 s) without ball or event, labelled live, that are a stoppage


# -- events ------------------------------------------------------------------------------------

def _events_csv(game):
    path = os.path.join(METRICA_DIR, "Sample_Game_{0}".format(game),
                        "Sample_Game_{0}_RawEventsData.csv".format(game))
    raw = pd.read_csv(path)
    return pd.DataFrame({
        "team": np.where(raw["Team"].str.strip().str.lower() == "home", 0, 1),
        "type": raw["Type"].str.strip().str.upper(),
        "subtype": raw["Subtype"].fillna("").astype(str).str.upper(),
        "period": raw["Period"].astype(int),
        "start": raw["Start Frame"].astype(int),
        "end": raw["End Frame"].fillna(0).astype(int),
    })


def _events_json(game, team_names):
    path = os.path.join(METRICA_DIR, "Sample_Game_{0}".format(game),
                        "Sample_Game_{0}_events.json".format(game))
    rows = []
    for event in json.load(open(path))["data"]:
        subtypes = event.get("subtypes") or []
        if isinstance(subtypes, dict):
            subtypes = [subtypes]
        rows.append({
            "team": 0 if event["team"]["name"] == team_names[0] else 1,
            "type": event["type"]["name"].strip().upper(),
            "subtype": "-".join(s["name"] for s in subtypes).upper(),
            "period": int(event["period"]),
            "start": int(event["start"]["frame"] or 0),
            "end": int((event.get("end") or {}).get("frame") or 0),
        })
    return pd.DataFrame(rows)


def labels_from_events(events, n_frames):
    """(labels, period) arrays, indexed by tracking frame - 1. See the module docstring."""
    events = events.sort_values(["start", "end"], kind="stable").reset_index(drop=True)
    labels = np.full(n_frames, X, dtype=np.int8)
    period = np.zeros(n_frames, dtype=np.int8)

    for half, group in events.groupby("period"):
        kickoffs = group[(group["type"] == "SET PIECE") & group["subtype"].str.contains("KICK OFF")]
        first = int(kickoffs["start"].min() if len(kickoffs) else group["start"].min()) - 1
        last = int(max(group["start"].max(), group["end"].max())) - 1
        period[first:last + 1] = half

        owner = np.full(n_frames, -1, dtype=np.int8)
        on_ball = group[group["type"].isin(POSSESSION_TYPES)]
        for start, team in zip(on_ball["start"].values - 1, on_ball["team"].values):
            owner[max(start, first)] = team
        # Forward-fill the change points, starting from the kick-off team.
        carry = np.maximum.accumulate(np.where(owner >= 0, np.arange(n_frames), -1))
        span = slice(first, last + 1)
        filled = owner[np.maximum(carry, 0)]
        labels[span] = np.where(filled[span] == 0, H, A)

        rows = group.to_dict("records")
        for index, event in enumerate(rows):
            if event["type"] != "SET PIECE" or event["start"] - 1 <= first:
                continue
            previous = next((e for e in reversed(rows[:index])
                             if e["type"] not in IGNORED_BEFORE_RESTART
                             and e["start"] < event["start"]), None)
            if previous is None:
                continue
            stop = previous["start"]
            if previous["type"] in TRAVELLING_TYPES:
                stop = max(stop, previous["end"])
            dead_from, dead_to = stop - 1, event["start"] - 1
            if dead_from < dead_to:
                labels[dead_from:dead_to] = D
    labels[period == 0] = X
    return labels, period


def unannotated_stoppages(labels, events, bx):
    """labels with D wherever they say live but the tracked ball is missing for at least
    UNANNOTATED_STOPPAGE frames in a row and no event of any type starts (module docstring)."""
    starts = np.sort(events["start"].values - 1)
    out = labels.copy()
    run = ((labels == H) | (labels == A)) & np.isnan(bx)
    edges = np.diff(np.concatenate(([0], run.astype(np.int8), [0])))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if stop - start >= UNANNOTATED_STOPPAGE and \
                np.searchsorted(starts, start) == np.searchsorted(starts, stop):
            out[start:stop] = D
    return out


# -- tracking ----------------------------------------------------------------------------------

def _tracking_csv(game):
    """Games 1 and 2: one CSV per team, x/y column pairs per player, the ball at the end."""
    halves = []
    for side in ("Home", "Away"):
        path = os.path.join(METRICA_DIR, "Sample_Game_{0}".format(game),
                            "Sample_Game_{0}_RawTrackingData_{1}_Team.csv".format(game, side))
        halves.append(pd.read_csv(path, skiprows=2))
    home, away = halves
    frames = home["Frame"].values

    def pairs(table):
        columns = list(table.columns)
        players = [c for c in columns if c.startswith("Player")]
        xs = np.stack([table[c].values for c in players], axis=1)
        ys = np.stack([table[columns[columns.index(c) + 1]].values for c in players], axis=1)
        return xs, ys

    hx, hy = pairs(home)
    ax, ay = pairs(away)
    ball_col = list(home.columns).index("Ball")
    length, width = METRICA_PITCH
    px = np.concatenate([hx, ax], axis=1).astype(np.float32) * length
    py = np.concatenate([hy, ay], axis=1).astype(np.float32) * width
    team = np.concatenate([np.zeros(hx.shape[1]), np.ones(ax.shape[1])]).astype(np.int8)
    bx = home.iloc[:, ball_col].values.astype(np.float32) * length
    by = home.iloc[:, ball_col + 1].values.astype(np.float32) * width
    if not np.array_equal(frames, np.arange(1, len(frames) + 1)):
        raise ValueError("game {}: tracking frames are not 1..n".format(game))
    return px, py, team, bx, by


def metrica_game(game, refresh=False):
    name = "metrica{}".format(game)

    def build():
        if game == 3:
            folder = os.path.join(METRICA_DIR, "Sample_Game_3")
            match = epts.load_match(name, os.path.join(folder, "Sample_Game_3_metadata.xml"),
                                    os.path.join(folder, "Sample_Game_3_tracking.txt"),
                                    METRICA_PITCH)
            events = _events_json(game, match.meta["team_names"])
        else:
            px, py, team, bx, by = _tracking_csv(game)
            match = Match(name, 25, METRICA_PITCH, px, py, team, bx, by,
                          np.zeros(len(bx), dtype=np.int8))
            events = _events_csv(game)
        match.labels, match.period = labels_from_events(events, match.n_frames)
        match.labels = unannotated_stoppages(match.labels, events, match.bx)
        return match

    return cached(name, build, refresh)


def labelled_games(refresh=False):
    return [metrica_game(g, refresh) for g in METRICA_GAMES]


def possession_summary(match):
    """Counts a human would sanity-check: possessions, stoppages, D share, ball coverage."""
    labels = match.labels
    graded = labels != X
    runs = np.flatnonzero(np.diff(np.concatenate(([X], labels))) != 0)
    starts = labels[runs]
    lengths = np.diff(np.concatenate((runs, [len(labels)])))
    spells = (starts == H) | (starts == A)
    return {
        "game": match.name,
        "graded_min": round(graded.sum() / match.fps / 60, 1),
        "H%": round(100 * np.mean(labels[graded] == H), 1),
        "A%": round(100 * np.mean(labels[graded] == A), 1),
        "D%": round(100 * np.mean(labels[graded] == D), 1),
        "possessions": int(spells.sum()),
        "mean_poss_s": round(lengths[spells].mean() / match.fps, 1),
        "stoppages": int((starts == D).sum()),
        "ball_seen%": round(100 * np.mean(~np.isnan(match.bx[graded])), 1),
        "players_seen": round(float(np.mean(np.sum(~np.isnan(match.px[graded]), axis=1))), 1),
    }


if __name__ == "__main__":
    rows = [possession_summary(m) for m in labelled_games(refresh=True)]
    print(pd.DataFrame(rows).to_string(index=False))
