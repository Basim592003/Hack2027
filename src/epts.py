"""FIFA/EPTS tracking (metadata XML + raw text) to a Match. Used for BAR-ATH and Metrica sample game 3.

Raw line: "<frame>:<player>;<player>;...:<ball>", each group "x,y[,speed]" normalised to 0-1. The
order of the player groups is the order of PlayerChannelRef entries in the DataFormatSpecification,
which is what this reads rather than assuming it matches the order of the Player elements. A file
may carry several specifications, each for a frame range -- Metrica sample game 3 starts a new one
at every substitution -- so each line is read with the one covering its frame. BAR-ATH has one.
"""

import xml.etree.ElementTree as ET

import numpy as np

from .match import Match

FIELD_SEP = ":"
GROUP_SEP = ";"


def _strip(tag):
    return tag.rsplit("}", 1)[-1]


def read_meta(meta_path):
    """(team per slot, specs, (home name, away name)). First team in the file is home.

    specs is [(start_frame, end_frame, [slot of each raw player group])]; a slot is one Player
    element, so a player keeps their slot across specifications.
    """
    root = ET.parse(meta_path).getroot()
    teams = [t for t in root.iter() if _strip(t.tag) == "Team"]
    team_ids = [t.get("id") for t in teams]
    names = tuple(next((c.text for c in t if _strip(c.tag) == "Name"), "") for t in teams)
    players = [p for p in root.iter() if _strip(p.tag) == "Player"]
    slot_of = {p.get("id"): i for i, p in enumerate(players)}
    team = np.array([team_ids.index(p.get("teamId")) for p in players], dtype=np.int8)
    channel_player = {c.get("id"): c.get("playerId")
                      for c in root.iter() if _strip(c.tag) == "PlayerChannel"}

    specs = []
    for spec in root.iter():
        if _strip(spec.tag) != "DataFormatSpecification":
            continue
        order = []
        for ref in spec.iter():
            if _strip(ref.tag) == "PlayerChannelRef":
                slot = slot_of[channel_player[ref.get("playerChannelId")]]
                if not order or order[-1] != slot:
                    order.append(slot)
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), order))
    return team, specs, names


def read_raw(raw_path, n_slots, specs):
    """(frame_ids, px, py, bx, by) in normalised units, one row per line of the raw file."""
    with open(raw_path) as handle:
        lines = handle.readlines()
    n = len(lines)
    frame_ids = np.zeros(n, dtype=np.int64)
    px = np.full((n, n_slots), np.nan, dtype=np.float32)
    py = np.full((n, n_slots), np.nan, dtype=np.float32)
    bx = np.full(n, np.nan, dtype=np.float32)
    by = np.full(n, np.nan, dtype=np.float32)

    for row, line in enumerate(lines):
        parts = line.rstrip("\n").split(FIELD_SEP)
        frame_id = frame_ids[row] = int(parts[0])
        order = next(o for start, end, o in specs if start <= frame_id <= end)
        for slot, group in zip(order, parts[1].split(GROUP_SEP)):
            if group[0] != "N":
                values = group.split(",")
                px[row, slot] = float(values[0])
                py[row, slot] = float(values[1])
        if len(parts) > 2 and parts[2] and parts[2][0] != "N":
            values = parts[2].split(",")
            bx[row] = float(values[0])
            by[row] = float(values[1])
    return frame_ids, px, py, bx, by


def load_match(name, meta_path, raw_path, pitch, fps=25):
    """A Match in metres. period and labels are left for the caller, who knows the halves."""
    team, specs, names = read_meta(meta_path)
    frame_ids, px, py, bx, by = read_raw(raw_path, len(team), specs)
    length, width = pitch
    match = Match(name, fps, tuple(pitch), px * length, py * width, team, bx * length,
                  by * width, np.zeros(len(bx), dtype=np.int8))
    match.meta.update(frame_ids=frame_ids, team_names=names)
    return match
