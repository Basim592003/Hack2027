"""Sprint item D3, diagnostic: where live frames with the ball missing get decoded as dead ball.

Bench copies use a saved experiment run's out-of-fold P(dead) and P(home) (default c5); BAR-ATH
uses the submission's final_pdead / final_phome. Aggregates only.

    PYTHONPATH=. .venv/bin/python src/diag/dead_in_gaps.py [RUN]
"""
import os
import sys

import numpy as np
import pandas as pd

from src import controlled, decode, deadball, touches
from src.bench import bench_games, source_of
from src.experiment import load
from src.external import labelled_games
from src.features import SHORT_BALL_GAP, fill_short_gaps
from src.match import A, CACHE_DIR, D, H
from src.sources import bar_ath

GAP_BINS = (0, 25, 50, 100, 200, 400, np.inf)          # frames
KICK_BINS = (0, 25, 75, 1e9)                           # frames from the last kick to the gap's start


def gap_table(match, labels, own_missing=None):
    """Per raw ball-gap frame inside the halves: gap length, frames from the last kick to the gap's
    start, whether the source itself had no ball there, predicted D, and (bench) the truth."""
    missing = np.isnan(match.bx) & (match.period > 0)
    edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    length = np.zeros(match.n_frames, dtype=np.int64)
    start_of = np.zeros(match.n_frames, dtype=np.int64)
    for a, b in zip(starts, stops):
        length[a:b] = b - a
        start_of[a:b] = a
    bx = fill_short_gaps(match.bx, SHORT_BALL_GAP)
    by = fill_short_gaps(match.by, SHORT_BALL_GAP)
    dist = np.hypot(match.px - bx[:, None], match.py - by[:, None])
    kick_frames, _, _ = touches.kicks(match.bx.astype(np.float64), match.by.astype(np.float64),
                                      dist, match.team, match.fps)
    del dist
    at = np.searchsorted(kick_frames, start_of, side="right") - 1
    since_kick = np.where(at >= 0, start_of - kick_frames[np.maximum(at, 0)], 10 ** 9)
    rows = np.flatnonzero(missing)
    out = pd.DataFrame({"gap": pd.cut(length[rows], GAP_BINS, right=False),
                        "since_kick": pd.cut(since_kick[rows], KICK_BINS, right=False),
                        "pred_dead": labels[rows] == D})
    if own_missing is not None:
        out["own_gap"] = own_missing[rows]
    if match.labels is not None:
        truth = controlled.controlled_labels(match.labels)
        out["live"] = np.isin(truth[rows], (H, A))
    return out


def main(run_name):
    run = load(run_name)
    dead_tag, home_tag = run.dead_tag.iloc[0], run.home_tag.iloc[0]
    clean = {m.name: np.isnan(m.bx) for m in labelled_games()}
    tables = []
    for match in bench_games():
        p_dead, p_home = decode.oof(match, dead_tag=dead_tag, home_tag=home_tag)
        labels = decode.decode(p_dead, p_home, match.period, decode.TEAM_COST, decode.DEAD_COST,
                               decode.MEMORY)
        own = clean.get(deadball.clean_name(match))
        table = gap_table(match, labels, own if own is not None
                          else np.zeros(match.n_frames, dtype=bool))
        tables.append(table.assign(source=source_of(match.name)))
    bench = pd.concat(tables, ignore_index=True)

    bar = bar_ath()
    p_dead = np.load(os.path.join(CACHE_DIR, "final_pdead_BAR-ATH.npy"))
    p_home = np.load(os.path.join(CACHE_DIR, "final_phome_BAR-ATH.npy"))
    labels = decode.decode(p_dead, p_home, bar.period, decode.TEAM_COST, decode.DEAD_COST,
                           decode.MEMORY)
    barg = gap_table(bar, labels).assign(source="BAR-ATH")

    pd.set_option("display.width", 200)
    print("== frames inside raw ball gaps: share truly live, share decoded D, and decoded D among the "
          "truly live ones")
    live = bench[bench.live]
    summary = pd.DataFrame({
        "gap frames": bench.groupby("source").size(),
        "truly live": bench.groupby("source").live.mean(),
        "decoded D": bench.groupby("source").pred_dead.mean(),
        "D | live": live.groupby("source").pred_dead.mean(),
    })
    summary.loc["BAR-ATH"] = [len(barg), np.nan, barg.pred_dead.mean(), np.nan]
    print(summary.to_string(float_format="%.3f"))
    for column in ("gap", "since_kick"):
        print("== by {} (frames): share of gap frames, decoded D; bench also truly live and D | live"
              .format(column))
        cols = {}
        for source, group in bench.groupby("source"):
            by = group.groupby(column, observed=True)
            cols[source + " share"] = by.size() / len(group)
            cols[source + " live"] = by.live.mean()
            cols[source + " D"] = by.pred_dead.mean()
            cols[source + " D|live"] = group[group.live].groupby(column, observed=True).pred_dead.mean()
        by = barg.groupby(column, observed=True)
        cols["BAR-ATH share"] = by.size() / len(barg)
        cols["BAR-ATH D"] = by.pred_dead.mean()
        print(pd.DataFrame(cols).to_string(float_format="%.3f"))
    metrica = live[live.source == "metrica"]
    print("== Metrica, live gap frames: the sample's own gaps vs gaps added by noise v5")
    print(metrica.groupby("own_gap").pred_dead.agg(["mean", "size"]).to_string(float_format="%.3f"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "c5")
