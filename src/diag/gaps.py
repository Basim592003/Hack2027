"""Where the pipeline still loses possessions (2026-11-11, after sprint item G4).

Out-of-fold predictions of a saved `experiment run`, decoded at the shipped costs, against the
controlled labels. One cause per real possession -- matched at IoU 0.5 by a same-team predicted
possession, or else wrong team, called dead, merged into a longer one, or split / boundary off --
by length, by how it starts and by how much of it has the ball seen; and what the unmatched
predicted possessions are. External bench only; aggregates only.

    python -m src.diag.gaps [RUN]     # RUN defaults to g4-t12
"""
import sys

import numpy as np
import pandas as pd

from src import controlled, decode, experiment
from src.bench import bench_games, source_of
from src.match import A, D, H, X

LENGTHS = ([0, 2, 4, 8, 16, np.inf], ["<2s", "2-4s", "4-8s", "8-16s", ">16s"])
IOU = 0.5


def runs(labels):
    """(start, stop, team) of every maximal H or A run."""
    change = np.flatnonzero(np.diff(labels.astype(np.int8)) != 0) + 1
    starts = np.concatenate(([0], change))
    stops = np.concatenate((change, [len(labels)]))
    return [(a, b, labels[a]) for a, b in zip(starts, stops) if labels[a] in (H, A)]


def best_iou(a, b, team, others, owner):
    """Highest IoU of [a, b) with a run in `others` of the same team; owner maps frames to runs."""
    best, best_len = 0.0, 0
    for i in set(owner[a:b].tolist()) - {-1}:
        pa, pb, pteam = others[i]
        if pteam != team:
            continue
        iou = (min(b, pb) - max(a, pa)) / (max(b, pb) - min(a, pa))
        if iou > best:
            best, best_len = iou, pb - pa
    return best, best_len


def owners(spells, n):
    owner = np.full(n, -1)
    for i, (a, b, _) in enumerate(spells):
        owner[a:b] = i
    return owner


def kind(labels, t):
    if t < 0 or t >= len(labels) or labels[t] == X:
        return "half edge"
    return "stoppage" if labels[t] == D else "turnover"


def game_rows(match, dead_tag, home_tag):
    truth = controlled.controlled_labels(match.labels)
    p_dead, p_home = decode.oof(match, dead_tag=dead_tag, home_tag=home_tag)
    pred = decode.decode(p_dead, p_home, match.period, decode.TEAM_COST, decode.DEAD_COST,
                         decode.MEMORY)
    pred = np.where(match.period > 0, pred, X).astype(np.int8)
    true_runs, pred_runs = runs(truth), runs(pred)
    true_owner, pred_owner = owners(true_runs, len(truth)), owners(pred_runs, len(truth))
    seen = ~np.isnan(match.bx)
    real, made = [], []
    for a, b, team in true_runs:
        seg = pred[a:b]
        other = np.mean((seg != team) & ((seg == H) | (seg == A)))
        iou, length = best_iou(a, b, team, pred_runs, pred_owner)
        if iou >= IOU:
            cause = "matched"
        elif other > 0.5:
            cause = "wrong team"
        elif np.mean(seg == D) > 0.5:
            cause = "called dead"
        elif length > 2 * (b - a):
            cause = "merged into a longer one"
        else:
            cause = "split / boundary off"
        real.append(dict(source=source_of(match.name), copy=match.name, cause=cause,
                         seconds=(b - a) / match.fps, starts=kind(truth, a - 1),
                         seen=np.mean(seen[a:b]), iou=iou))
    for a, b, team in pred_runs:
        iou, _ = best_iou(a, b, team, true_runs, true_owner)
        if iou >= IOU:
            continue
        seg = truth[a:b]
        if np.mean((seg != team) & ((seg == H) | (seg == A))) > 0.5:
            cause = "other team's possession"
        elif np.mean(seg == D) > 0.5:
            cause = "stoppage"
        else:
            cause = "piece of a real one"
        made.append(dict(source=source_of(match.name), copy=match.name, cause=cause,
                         seconds=(b - a) / match.fps))
    return real, made


def per_match(table, index, columns):
    """Counts per match (copies averaged), rows x columns, per source."""
    copies = table.groupby("source")["copy"].nunique()
    counts = table.groupby(["source"] + index + columns, observed=True).size()
    return (counts / copies.reindex(counts.index.get_level_values("source")).values).unstack(
        columns, fill_value=0)


def main(name):
    saved = experiment.load(name)
    dead_tag, home_tag = saved.dead_tag.iloc[0], saved.home_tag.iloc[0]
    real, made = [], []
    for match in bench_games():
        r, m = game_rows(match, dead_tag, home_tag)
        real += r
        made += m
    real, made = pd.DataFrame(real), pd.DataFrame(made)
    real["length"] = pd.cut(real.seconds, LENGTHS[0], labels=LENGTHS[1], right=False)
    made["length"] = pd.cut(made.seconds, LENGTHS[0], labels=LENGTHS[1], right=False)
    real["ball"] = np.where(real.seen < 0.5, "ball seen <50%", "ball seen >=50%")
    print("== {} (dead @{}, home @{}), team {} / dead {}, controlled labels, IoU {}".format(
        name, dead_tag, home_tag, decode.TEAM_COST, decode.DEAD_COST, IOU))
    print("== real possessions per match, by cause")
    print(per_match(real, ["cause"], ["length"]).assign(
        all=lambda t: t.sum(axis=1)).to_string(float_format="%.1f"))
    print("== real possessions per match, by how they start: matched share and count")
    print(real.assign(matched=real.cause == "matched").groupby(["source", "starts"]).agg(
        matched=("matched", "mean"), per_match=("copy", lambda c: len(c) / c.nunique()))
        .to_string(float_format="%.3f"))
    print("== real possessions, by how much of them has the ball seen: matched share")
    print(real.assign(matched=real.cause == "matched").groupby(["source", "ball"]).agg(
        matched=("matched", "mean"), share=("copy", "size")).assign(
        share=lambda t: t.share / t.groupby("source").share.transform("sum"))
        .to_string(float_format="%.3f"))
    print("== unmatched predicted possessions per match, by what they really are")
    print(per_match(made, ["cause"], ["length"]).assign(
        all=lambda t: t.sum(axis=1)).to_string(float_format="%.1f"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "g4-t12")
