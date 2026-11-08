"""IDSSE open Bundesliga data (7 games, CC BY 4.0) as labelled Matches. Fetch with src/fetch_idsse.

Bassek, Weber, Rein & Memmert (2025), Scientific Data, doi:10.1038/s41597-025-04505-y.

DFL position data: one FrameSet per person per half, plus a BALL FrameSet whose frames carry
BallStatus (1 alive, 0 dead) and BallPossession (1 home, 2 away). Those two flags are the labels,
read as they are -- the closest public analogue of what the BAR-ATH ground truth is derived from
("D where the corrected tracking marks the ball dead, otherwise the team it names as owning it").
How DFL operators set those flags is DFL's convention, not Metrica's, which is one reason the
benchmark also keeps the Metrica sample games.

Coordinates are metres from the centre spot; they are shifted to the corner origin Match uses.
The halves are joined with HALF_GAP frames of X between them, so nothing is scored across them.

    python -m src.idsse              # converts, caches and prints label statistics
"""

import glob
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from .match import A, D, H, X, Match, cached

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDSSE_DIR = os.path.join(ROOT, "data", "external", "idsse")
MATCH_IDS = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
SECTIONS = {"firstHalf": 1, "secondHalf": 2}
HALF_GAP = 250


def _file(kind, match_id):
    paths = glob.glob(os.path.join(IDSSE_DIR, "DFL_*_{}_*DFL-MAT-{}.xml".format(kind, match_id)))
    if len(paths) != 1:
        raise FileNotFoundError("{} file for {} not found in {} -- run python -m src.fetch_idsse"
                                .format(kind, match_id, IDSSE_DIR))
    return paths[0]


def _meta(match_id):
    general = next(e for e in ET.parse(_file("matchinformation", match_id)).getroot().iter()
                   if e.tag == "General")
    environment = next(e for e in ET.parse(_file("matchinformation", match_id)).getroot().iter()
                       if e.tag == "Environment")
    return (general.get("HomeTeamId"), general.get("GuestTeamId"),
            (float(environment.get("PitchX")), float(environment.get("PitchY"))))


def _framesets(match_id):
    """{(half, team_id, person_id): (N, X, Y, possession, status)} streamed off the position file."""
    out = {}
    key = None
    rows = []
    for event, elem in ET.iterparse(_file("positions_raw", match_id), events=("start", "end")):
        if event == "start" and elem.tag == "FrameSet":
            team = elem.get("TeamId")
            person = None if team == "BALL" else elem.get("PersonId")
            key = (SECTIONS[elem.get("GameSection")], team, person)
            rows = []
        elif event == "end" and elem.tag == "Frame":
            rows.append((int(elem.get("N")), float(elem.get("X")), float(elem.get("Y")),
                         int(elem.get("BallPossession", 0)), int(elem.get("BallStatus", 0))))
            elem.clear()
        elif event == "end" and elem.tag == "FrameSet":
            out[key] = np.array(rows, dtype=np.float64)
            elem.clear()
    return out


def idsse_match(match_id):
    home_id, away_id, pitch = _meta(match_id)
    framesets = _framesets(match_id)
    length, width = pitch

    # Timeline: each half spans its ball frames, the halves joined by HALF_GAP frames of X.
    offsets, total = {}, 0
    for half in (1, 2):
        frames = framesets[(half, "BALL", None)][:, 0].astype(np.int64)
        offsets[half] = (total - frames.min(), frames.min(), frames.max())
        total += frames.max() - frames.min() + 1 + HALF_GAP
    total -= HALF_GAP

    # Players only: some games also track the referees, who belong to neither team.
    framesets = {k: v for k, v in framesets.items() if k[1] in (home_id, away_id, "BALL")}
    people = sorted({(team, person) for (_, team, person) in framesets if team != "BALL"})
    slot_of = {p: i for i, p in enumerate(people)}
    px = np.full((total, len(people)), np.nan, dtype=np.float32)
    py = np.full_like(px, np.nan)
    bx = np.full(total, np.nan, dtype=np.float32)
    by = np.full_like(bx, np.nan)
    period = np.zeros(total, dtype=np.int8)
    labels = np.full(total, X, dtype=np.int8)

    for (half, team, person), rows in framesets.items():
        shift, first, last = offsets[half]
        keep = (rows[:, 0] >= first) & (rows[:, 0] <= last)
        rows = rows[keep]
        index = rows[:, 0].astype(np.int64) + shift
        x = (rows[:, 1] + length / 2).astype(np.float32)
        y = (rows[:, 2] + width / 2).astype(np.float32)
        if team == "BALL":
            bx[index], by[index] = x, y
            period[index] = half
            alive = rows[:, 4] == 1
            labels[index] = np.where(~alive, D, np.where(rows[:, 3] == 1, H, A))
        else:
            slot = slot_of[(team, person)]
            px[index, slot], py[index, slot] = x, y

    team = np.array([0 if t == home_id else 1 for t, _ in people], dtype=np.int8)
    if set(team.tolist()) != {0, 1}:
        raise ValueError("{}: expected players from both teams".format(match_id))
    return Match("idsse_" + match_id, 25, pitch, px, py, team, bx, by, period, labels)


def idsse_game(match_id, refresh=False):
    return cached("idsse_" + match_id, lambda: idsse_match(match_id), refresh)


def idsse_games(refresh=False):
    return [idsse_game(m, refresh) for m in MATCH_IDS]


if __name__ == "__main__":
    from .external import possession_summary
    rows = [possession_summary(m) for m in idsse_games(refresh=True)]
    print(pd.DataFrame(rows).to_string(index=False))
