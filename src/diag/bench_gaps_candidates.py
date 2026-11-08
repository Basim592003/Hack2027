"""External bench only (v4 caches): ball-gap length profile, and how many controlled in-play
turnovers already have a candidate event (kick / exclusive-close change / ball-gap edge) nearby.
Aggregates only."""
import numpy as np
import pandas as pd

from src import controlled
from src.bench import bench_games, source_of
from src.features import SHORT_BALL_GAP, fill_short_gaps
from src.match import A, H, X
from src.possession import CONTROL_RADIUS
from src.touches import kicks

TOL = 12  # frames either side


def runs(mask):
    e = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(e == 1), np.flatnonzero(e == -1)


def near(events, c, tol=TOL):
    if not len(events):
        return False
    i = np.searchsorted(events, c - tol)
    return i < len(events) and events[i] <= c + tol


rows = []
for m in bench_games():
    y = m.labels
    graded = (m.period > 0) & (y != X)
    live = (y == H) | (y == A)
    miss = np.isnan(m.bx) & graded
    s, e = runs(miss)
    gaplen = np.zeros(len(y), dtype=np.int64)
    for a, b in zip(s, e):
        gaplen[a:b] = b - a

    yc = controlled.controlled_labels(y)
    ch = np.flatnonzero(live[1:] & live[:-1] & (yc[1:] != yc[:-1])) + 1
    new_team = np.where(yc[ch] == H, 0, 1)

    bx = fill_short_gaps(m.bx, SHORT_BALL_GAP)
    by = fill_short_gaps(m.by, SHORT_BALL_GAP)
    dist = np.hypot(m.px - bx[:, None], m.py - by[:, None])
    kf, kt, _ = kicks(m.bx.astype(np.float64), m.by.astype(np.float64), dist, m.team, m.fps)
    close = [np.sum((dist < CONTROL_RADIUS) & (m.team == k)[None, :], axis=1) for k in (0, 1)]
    del dist
    excl = (close[0] > 0) != (close[1] > 0)
    cf = np.flatnonzero(excl)
    ct = np.where(close[0][cf] > 0, 0, 1)
    flip = np.flatnonzero(ct[1:] != ct[:-1]) + 1
    cc_frames, cc_team = cf[flip], ct[flip]
    seen = ~np.isnan(m.bx)
    edges = np.flatnonzero(np.diff(seen.astype(np.int8)) != 0) + 1

    k_new = [near(kf[kt == t], c) for c, t in zip(ch, new_team)]
    k_any = [near(kf, c) for c in ch]
    c_new = [near(cc_frames[cc_team == t], c) for c, t in zip(ch, new_team)]
    g_any = [near(edges, c) for c in ch]
    union = np.array(k_any) | np.array(c_new) | np.array(g_any)
    minutes = graded.sum() / m.fps / 60
    rows.append(dict(
        source=source_of(m.name), live=int(m.name.split("live")[1][:2]),
        miss_in_gaps_lt2s=(gaplen[miss] < 50).mean(),
        inplay_miss_in_gaps_lt2s=(gaplen[miss & live] < 50).mean(),
        inplay_miss_in_gaps_gt5s=(gaplen[miss & live] > 125).mean(),
        ctl_turnovers=len(ch),
        turnover_ball_missing=miss[ch].mean(),
        kick_new_team=np.mean(k_new), kick_any=np.mean(k_any),
        closechg_new_team=np.mean(c_new), gap_edge=np.mean(g_any),
        any_candidate=union.mean(),
        candidates_per_min=(len(kf) + len(cc_frames) + len(edges)) / minutes,
    ))
    print(" ", m.name, flush=True)

t = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(t.groupby(["source", "live"]).mean(numeric_only=True).T.to_string(float_format="%.3f"))
