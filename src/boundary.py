"""Learned team-switch cost: b_t, the probability that possession changes hands in play within
WITHIN frames of t, turned into the joint decoder's per-frame cost of switching team.

A constant switch cost charges every frame the same, so the decoder flips wherever P(home) crosses
over for long enough. Making switches cheap near every event instead (src/decode.py's
event_costs, sprint item C2) lost: events come every 3 s and most are not hand-overs. This learns
where hand-overs are from the external games, using signals that carry no team, so it needs no team
views:

  events    each kind of event (src/segments.py) within EVENT_WINDOWS frames, and the distance to
            the nearest one
  P(home)   how much the out-of-fold P(home) steps between the STEP_WINDOWS frames before and
            after, and how far it is from 0.5
  nearest   flips of the team nearest the ball within FLIP_WINDOWS frames
  P(dead)   at the frame and over +/-12 frames

The cost is clip(TEAM_COST - beta (logit b_t - logit neutral), 0.5, 20): TEAM_COST where b_t equals
`neutral`, cheaper where a change is likelier, dearer where it is less likely. That is the
checklist's clip(alpha - beta logit b_t), with alpha = TEAM_COST + beta logit neutral.

Training rows are every positive frame and every NEGATIVE_STRIDE-th negative one among the live
frames; the target is controlled possession (src/controlled.py), as BAR-ATH's answer key counts it.
"""

import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from . import controlled, segments
from .bench import nearest_team
from .features import CONTEXT_CAP, rolling_mean
from .match import A, CACHE_DIR, H
from .touches import window_count

WITHIN = 3                 # frames either side of a real change that count as "a change here"
EVENT_WINDOWS = (6, 12, 25)
STEP_WINDOWS = (6, 12, 25, 50)
FLIP_WINDOWS = (12, 25)
DEAD_WINDOW = 12
NEGATIVE_STRIDE = 5
COST_RANGE = (0.5, 20.0)
CLIP = 1e-4
MODEL_PARAMS = dict(max_iter=300, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=200,
                    l2_regularization=1.0, early_stopping=False, random_state=0)


def flips(team):
    """Frames where the team differs from the last known one before it (-1, unknown, is skipped)."""
    known = np.flatnonzero(team >= 0)
    values = team[known]
    return known[1:][values[1:] != values[:-1]]


def frame_features(match, p_dead, p_home, events):
    """{name: (T,) float32} from a match, its out-of-fold (or, on BAR-ATH, held-out) P(dead) and
    P(home), and its events from segments.match_events."""
    n = match.n_frames
    index = np.arange(n)
    frames, kinds, _ = events
    feats = {}
    for name, flag in segments.KINDS:
        mine = frames[(kinds & flag) > 0]
        for k in EVENT_WINDOWS:
            feats["{}_within{}".format(name, k)] = segments.within(mine, index, k)
    at = np.searchsorted(frames, index)
    after = np.where(at < len(frames), frames[np.minimum(at, len(frames) - 1)] - index, CONTEXT_CAP)
    before = np.where(at > 0, index - frames[np.maximum(at - 1, 0)], CONTEXT_CAP)
    feats["log_to_event"] = np.log1p(np.minimum(np.minimum(after, before), CONTEXT_CAP))
    p_home = p_home.astype(np.float64)
    for k in STEP_WINDOWS:
        feats["home_step{}".format(k)] = np.abs(rolling_mean(p_home, 0, k - 1)
                                                - rolling_mean(p_home, k, -1))
    feats["home_margin"] = np.abs(p_home - 0.5)
    flipped = np.zeros(n, dtype=np.int64)
    flipped[flips(nearest_team(match))] = 1
    for k in FLIP_WINDOWS:
        feats["nearest_flips{}".format(k)] = window_count(flipped, k, k)
    feats["p_dead"] = p_dead
    feats["p_dead_c{}".format(DEAD_WINDOW)] = rolling_mean(p_dead.astype(np.float64), DEAD_WINDOW,
                                                           DEAD_WINDOW)
    return {k: np.asarray(v, dtype=np.float32) for k, v in feats.items()}


def as_matrix(feats):
    names = sorted(feats)
    return np.stack([feats[k] for k in names], axis=1)


def targets(labels):
    """(positive, live): frames within WITHIN of a controlled change of team between two live
    frames, and the live frames themselves."""
    truth = controlled.controlled_labels(labels)
    live = (truth == H) | (truth == A)
    changes = np.flatnonzero(live[1:] & live[:-1] & (truth[1:] != truth[:-1])) + 1
    return segments.within(changes, np.arange(len(labels)), WITHIN) & live, live


def training_rows(tables, names_in):
    xs, ys = [], []
    for name in names_in:
        matrix, positive, live = tables[name]
        keep = np.sort(np.concatenate((np.flatnonzero(positive),
                                       np.flatnonzero(live & ~positive)[::NEGATIVE_STRIDE])))
        xs.append(matrix[keep])
        ys.append(positive[keep])
    return np.concatenate(xs), np.concatenate(ys)


def fit(tables, names_in):
    x, y = training_rows(tables, names_in)
    return HistGradientBoostingClassifier(**MODEL_PARAMS).fit(x, y)


def switch_costs(b, beta, neutral, base):
    """Per-frame team cost from b_t; equal to `base` where b_t == neutral."""
    b = np.clip(b, CLIP, 1 - CLIP)
    logit = np.log(b / (1 - b)) - np.log(neutral / (1 - neutral))
    return np.clip(base - beta * logit, *COST_RANGE)


def oof_path(name, tag):
    return os.path.join(CACHE_DIR, "oof", "pswitch_{}@{}.npy".format(name, tag))
