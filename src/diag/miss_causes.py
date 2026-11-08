"""Why real possessions are missed (v4 bench, out-of-fold, joint decoder): one cause per GT spell."""
import numpy as np
import pandas as pd
from src import decode
from src.bench import bench_games, source_of
from src.match import A, D, H, X

def runs(labels):
    out, start = [], 0
    for t in range(1, len(labels) + 1):
        if t == len(labels) or labels[t] != labels[start]:
            if labels[start] in (H, A):
                out.append((start, t, labels[start]))
            start = t
    return out

rows = []
for m in bench_games():
    y = m.labels
    p_dead, p_home = decode.oof(m)
    pred = decode.decode(p_dead, p_home, m.period, decode.TEAM_COST, decode.DEAD_COST, False)
    pred_runs = runs(pred)
    owner = np.full(len(y), -1); 
    for i, (a, b, _) in enumerate(pred_runs):
        owner[a:b] = i
    for a, b, team in runs(y):
        L = b - a
        seg = pred[a:b]
        same, other, dead = np.mean(seg == team), np.mean((seg != team) & (seg != D) & (seg != X)), np.mean(seg == D)
        # best same-team predicted spell by IoU
        best, best_len = 0.0, 0
        for i in set(owner[a:b][seg == team].tolist()):
            pa, pb, _ = pred_runs[i]
            inter = min(b, pb) - max(a, pa)
            iou = inter / (max(b, pb) - min(a, pa))
            if iou > best:
                best, best_len = iou, pb - pa
        start_kind = "after restart" if a > 0 and y[a - 1] == D else "in play"
        if best >= 0.5:
            cause = "matched"
        elif other > 0.5:
            cause = "wrong team"
        elif dead > 0.5:
            cause = "called dead"
        elif best_len > 2 * L:
            cause = "merged into a longer spell"
        else:
            cause = "split / boundary off"
        rows.append(dict(source=source_of(m.name), cause=cause, start=start_kind,
                         length=pd.cut([L / m.fps], [0, 2, 4, 8, 1e9], labels=["<2s", "2-4s", "4-8s", ">8s"])[0]))
t = pd.DataFrame(rows)
print("== share of real possessions, by cause")
print(pd.crosstab(t.cause, t.source, normalize="columns").to_string(float_format="%.3f"))
print("== among MISSED possessions: cause x length (share of missed)")
miss = t[t.cause != "matched"]
for s in ("idsse", "metrica"):
    print("--", s)
    print(pd.crosstab(miss[miss.source == s].cause, miss[miss.source == s].length, normalize="all").to_string(float_format="%.3f"))
print("== missed share by start type")
print(t.assign(missed=t.cause != "matched").groupby(["source", "start"]).missed.agg(["mean", "size"]).to_string(float_format="%.3f"))
