"""Where the remaining spellF1 is, on the v4 bench, from the out-of-fold caches."""
import numpy as np
import pandas as pd
from src import decode
from src.bench import bench_games
from src.match import A, D, H, X

HINT = 12   # frames either side of a true in-play team change where switching is cheap

def viterbi4_varying(e_home, e_away, e_dead, team_cost, dead):
    """decode.viterbi4 (no restart memory) with a per-frame team-switch cost."""
    n = len(e_home)
    back = np.zeros((n, 4), dtype=np.int8)
    s = [0.0, 0.0, 0.0, 0.0]
    for t, (eh, ea, ed, tc) in enumerate(zip(e_home.tolist(), e_away.tolist(), e_dead.tolist(), team_cost.tolist())):
        h, a, dh, da = s
        cands_h = (h, a - tc, dh - dead, da - dead)
        cands_a = (h - tc, a, dh - dead, da - dead)
        fh = int(np.argmax(cands_h)); fa = int(np.argmax(cands_a))
        fdh = 2 if dh >= h - dead else 0
        fda = 3 if da >= a - dead else 1
        back[t] = (fh, fa, fdh, fda)
        s = [cands_h[fh] + eh, cands_a[fa] + ea, (dh if fdh == 2 else h - dead) + ed, (da if fda == 3 else a - dead) + ed]
    state = int(np.argmax(s)); path = np.empty(n, dtype=np.int8)
    for t in range(n - 1, -1, -1):
        path[t] = state; state = back[t, state]
    return np.array([H, A, D, D], dtype=np.int8)[path]

rows = []
for m in bench_games():
    y = m.labels
    p_dead, p_home = decode.oof(m)
    live = (y == H) | (y == A)
    true_dead = np.where(y == D, 1.0, np.where(live, 0.0, p_dead))
    true_home = np.where(y == H, 1.0, np.where(y == A, 0.0, p_home))
    def run(pd_, ph_, tag):
        lab = decode.decode(pd_, ph_, m.period, decode.TEAM_COST, decode.DEAD_COST, False)
        rows.append(decode.score(m, lab, case=tag))
    run(p_dead, p_home, "1 as predicted")
    run(true_dead, p_home, "2 true D, predicted team")
    run(p_dead, true_home, "3 predicted D, true team")
    # perfect "when" hint: cheap switches only near real in-play changes, team still predicted
    change = np.flatnonzero(live[1:] & live[:-1] & (y[1:] != y[:-1])) + 1
    near = np.zeros(len(y), bool)
    for c in change:
        near[max(c - HINT, 0):c + HINT] = True
    cost = np.where(near, 0.5, 20.0)
    e_h, e_a, e_d = decode.log_emissions(p_dead, p_home)
    lab = np.full(len(y), X, dtype=np.int8)
    for half in np.unique(m.period[m.period > 0]):
        f = np.flatnonzero(m.period == half); sl = slice(f[0], f[-1] + 1)
        lab[sl] = viterbi4_varying(e_h[sl], e_a[sl], e_d[sl], cost[sl], decode.DEAD_COST)
    rows.append(decode.score(m, lab, case="4 predicted D+team, perfect WHEN-to-switch hint"))
    print(" ", m.name, flush=True)
t = pd.DataFrame(rows)
print(t.groupby(["case", "source"])[["spellF1", "n_pred", "n_gt"]].mean().unstack("source").to_string(float_format="%.3f"))
