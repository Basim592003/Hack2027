"""Relabel the external games from "last touch" possession to the challenge's "controlled" one.

Both external label sets hand possession to whoever touched the ball last: a defender's deflection
or a clearance out of play is a possession of a frame or two. CHALLENGE.md defines it differently:

  * a change of possession is the opposing team gaining *controlled* possession
  * a simple touch without control, a deflection that does not intercept, or an aerial duel won
    without the team then controlling the ball, is not a change of possession
  * while the ball is contested or loose, possession stays with the last team that had it
  * whenever play restarts after a stoppage there is a clear possessor

controlled_labels applies exactly that to a label array, inside each live stretch:

  * the team that restarts play owns the ball, however briefly it keeps it
  * after that, a spell of the other team takes possession over only if it lasts at least
    MIN_CONTROL frames; a shorter one is a touch without control and goes to the current owner

MIN_CONTROL comes from the definition ("a player keeps the ball for a few seconds" is the proxy
CHALLENGE.md suggests; one second is the least that reads as control), not from any score: a
longer threshold removes more short possessions from the target and always makes it easier, so a
benchmark score cannot choose it. Dead-ball frames are never changed.

    python -m src.controlled          # how much the relabelling changes each external game

What it changed (2026-11-04). The 1 s rule removes 15-25% of the external games' possessions
(Metrica 380 -> 322, IDSSE 511 -> 395) while changing only 0.2-0.7% of frames. But retraining the
team model on these labels did not help: scored against controlled labels, the model trained on
last-touch labels does as well or marginally better (best spellF1 0.680 vs 0.676 IDSSE, 0.741 vs
0.733 Metrica), with the same best switch cost. The decoder's switch cost already refuses to flip
possession on a brief touch, which is this rule. So the submission keeps CONVENTION = "last_touch"
in src/run.py. What the controlled labels did show: possessions that start after a restart are
recovered 83-84% of the time, those that start with a turnover in play only 57-67%.
"""

import numpy as np
import pandas as pd

from .match import A, D, H, X

MIN_CONTROL = 25          # frames: one second
CONVENTIONS = ("last_touch", "controlled")
SUFFIX = "~ctl"           # appended to a game's name when its labels are relabelled


def _runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def controlled_labels(labels, min_control=MIN_CONTROL):
    out = labels.copy()
    for start, stop in _runs((labels == H) | (labels == A)):
        stretch = out[start:stop]
        changes = np.flatnonzero(stretch[1:] != stretch[:-1]) + 1
        bounds = np.concatenate(([0], changes, [len(stretch)]))
        owner = stretch[0]
        for first, last in zip(bounds[1:-1], bounds[2:]):
            team = labels[start + first]
            if team != owner and last - first >= min_control:
                owner = team
            stretch[first:last] = owner
    return out


def apply(match, convention):
    """The match with its labels in the given convention. Dead ball is never touched."""
    if convention == "last_touch":
        return match
    if convention != "controlled":
        raise ValueError("convention must be one of {}".format(CONVENTIONS))
    match.labels = controlled_labels(match.labels)
    match.name = match.name + SUFFIX
    return match


def base_name(name):
    """The game's name without the convention suffix -- for anything that does not depend on
    possession labels, such as the dead-ball model's out-of-fold predictions."""
    return name[:-len(SUFFIX)] if name.endswith(SUFFIX) else name


def spell_stats(labels, fps):
    spells = []
    for start, stop in _runs((labels == H) | (labels == A)):
        stretch = labels[start:stop]
        changes = np.flatnonzero(stretch[1:] != stretch[:-1]) + 1
        bounds = np.concatenate(([0], changes, [len(stretch)]))
        spells.extend(np.diff(bounds))
    spells = np.array(spells) / fps
    return len(spells), spells.mean(), np.mean(spells < 2)


if __name__ == "__main__":
    from .external import labelled_games
    from .idsse import idsse_games

    rows = []
    for match in labelled_games() + idsse_games():
        row = {"game": match.name}
        for tag, min_control in (("last touch", None), ("ctl 0.5s", 12), ("ctl 1s", 25),
                                 ("ctl 2s", 50)):
            labels = match.labels if min_control is None else \
                controlled_labels(match.labels, min_control)
            n, mean, short = spell_stats(labels, match.fps)
            row["{} n".format(tag)] = n
            row["{} mean_s".format(tag)] = round(mean, 1)
            row["{} <2s".format(tag)] = round(100 * short)
        changed = np.mean(controlled_labels(match.labels) != match.labels)
        row["frames changed (1s)"] = "{:.1f}%".format(100 * changed)
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))
