"""Second-stage team model (sprint item G4): P(home) again, from what the first stage said around
each frame.

The first stage (src/possession.py) reads the ball and the players over windows of 1-4 s and over
the segments between events, and the joint decoder (src/decode.py) adds its log-odds up frame by
frame as if every frame were separate evidence. Its mistakes are not separate: a wrong guess lasts
as long as whatever misleads it. The second stage gets the first stage's own features and, next to
them, what the first stage said over the 8 s around the frame: its log-odds at fixed lags, their
mean and spread over windows and over the segment and its neighbours, P(dead), and how far the
frame is from a stoppage. It learns from the other games when to believe the first stage.

It is trained on out-of-fold first-stage predictions only, so it learns what the first stage is like
on a game it has not seen. On the bench the folds are nested (src/experiment.py, `stack`): the
predictions that train it for a held-out game come from first-stage models that never saw that
game either. Every feature is from one team's point of view, as in the first stage, and the two
views are averaged at the end, so the output stays symmetric.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from . import deadball, possession, segments
from .features import CONTEXT_CAP, rolling_mean

LAGS = (25, 50, 100, 200)    # frames: 1, 2, 4 and 8 s, both ways
WINDOWS = (("c1", 12, 12), ("c4", 50, 50), ("c10", 125, 125), ("past4", 100, 0),
           ("next4", 0, 100), ("past8", 200, 0), ("next8", 0, 200))
SPREAD = (("c4", 50, 50), ("c10", 125, 125))
Z_CAP = 8.0                  # log-odds cap; nothing beyond it changes a decision


def logit(p):
    p = np.clip(p, deadball.CLIP, 1 - deadball.CLIP)
    return np.clip(np.log(p / (1 - p)), -Z_CAP, Z_CAP)


def shifted(values, lag):
    """values[t + lag] per frame, NaN past either end."""
    out = np.full(len(values), np.nan)
    if lag > 0:
        out[:-lag] = values[lag:]
    elif lag < 0:
        out[-lag:] = values[:lag]
    else:
        out[:] = values
    return out


def context(p_home, p_dead, dead, events):
    """The home view's context features as (odd, even): odd ones change sign for the away view
    (anything made from the home log-odds or from an event's team), even ones do not (spreads,
    P(dead), the segment's shape, the distance to a stoppage)."""
    n = len(p_home)
    z = logit(p_home)
    odd, even = {"z": z}, {}
    for lag in LAGS:
        odd["z_lag{}".format(lag)] = shifted(z, -lag)
        odd["z_lead{}".format(lag)] = shifted(z, lag)
    for tag, before, after in WINDOWS:
        odd["z_" + tag] = rolling_mean(z, before, after)
    for tag, before, after in SPREAD:
        mean = rolling_mean(z, before, after)
        even["z_sd_" + tag] = np.sqrt(np.maximum(rolling_mean(z * z, before, after) - mean ** 2, 0))

    frames, kinds, teams = events
    own, before, after = segments.pool(z, frames)
    odd.update(z_seg=own, z_seg_prev=before, z_seg_next=after, z_seg_from_prev=own - before,
               z_seg_to_next=after - own)
    layout = segments.layout(frames, kinds, teams, np.ones(n, dtype=bool))
    even["seg_log_len"], even["seg_pos"] = layout["seg_log_len"], layout["seg_pos"]
    for edge in ("start", "end"):
        team = layout["seg_{}_team".format(edge)]
        odd["seg_{}_team".format(edge)] = np.where(team < 0, 0.0, np.where(team == 0, 1.0, -1.0))

    even["z_dead"] = logit(p_dead)
    even["z_dead_c4"] = rolling_mean(even["z_dead"], 50, 50)
    index = np.arange(n)
    last = np.maximum.accumulate(np.where(dead, index, -1))
    following = np.minimum.accumulate(np.where(dead, index, n)[::-1])[::-1]
    even["log_since_dead"] = np.log1p(np.where(last >= 0, np.minimum(index - last, CONTEXT_CAP),
                                               CONTEXT_CAP))
    even["log_until_dead"] = np.log1p(np.where(following < n,
                                               np.minimum(following - index, CONTEXT_CAP),
                                               CONTEXT_CAP))
    return odd, even


def views(first, p_home, p_dead, dead, events):
    """(context names, both views): the first stage's view features followed by the context's,
    the away view's odd context features negated."""
    odd, even = context(p_home, p_dead, dead, events)
    names = sorted(odd) + sorted(even)
    home = np.stack([odd[k] for k in sorted(odd)] + [even[k] for k in sorted(even)], axis=1)
    away = np.stack([-odd[k] for k in sorted(odd)] + [even[k] for k in sorted(even)], axis=1)
    return names, [np.hstack([first[0], home]).astype(np.float32),
                   np.hstack([first[1], away]).astype(np.float32)]


def rows(first, p_home, p_dead, dead, events, labels, name):
    """possession.training_rows of one copy's second-stage views, without keeping the views."""
    _, both = views(first, p_home, p_dead, dead, events)
    return possession.training_rows({name: (both, labels, dead)}, [name])


def fit(parts):
    """The second stage on a list of rows() outputs."""
    x = np.concatenate([part[0] for part in parts])
    y = np.concatenate([part[1] for part in parts])
    return HistGradientBoostingClassifier(**possession.MODEL_PARAMS).fit(x, y)


def p_home(model, first, p_first, p_dead, dead, events):
    """Second-stage P(home), symmetric like possession.p_home."""
    return possession.p_home(model, views(first, p_first, p_dead, dead, events)[1])
