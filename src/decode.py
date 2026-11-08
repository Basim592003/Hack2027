"""Joint decoder: one Viterbi pass over {H, A, D after H, D after A}, then boundary snapping.

Steps 2 and 3 decide dead ball and team one after the other. This decides them together, from

    P(D) = p_dead      P(H) = (1 - p_dead) p_home      P(A) = (1 - p_dead) (1 - p_home)

with p_dead from src/deadball.py and p_home from src/possession.py. D is split by the team that
had the ball before the stoppage, so a restart can be charged for keeping or for switching the
ball: on the external games about 70% of stoppages hand it to the other team (CLAUDE.md). That
rate is measured under the external games' label conventions, so it is a variant that is kept
only if it helps there.

Costs (all subtracted from the path score, chosen on the external games' out-of-fold
probabilities, never on BAR-ATH):

    team      H <-> A inside live play
    dead      entering or leaving a stoppage
    keep / switch   extra cost of leaving "D after T" to T / to the other team

Snapping then moves each boundary to the event it should sit on, within SNAP_WINDOW frames:
a team change to the first frame the new team's player is nearest the ball and stays nearest, a
stoppage start to the frame the ball crosses a line, a restart to the frame a still ball is kicked.

    python -m src.decode                    # grid of costs on the external bench
    python -m src.decode --snap 6 40 0      # snapping variants at those costs

What the bench said (2026-11-03, out-of-fold probabilities, 20 degraded games):
  * the joint pass matches steps 2 + 3 run one after the other -- spellF1 0.618 vs 0.616 IDSSE,
    0.678 vs 0.674 Metrica, far inside the 0.04 noise band. Kept because it is one pass.
  * restart memory changes nothing: step 3's restart features already carry who restarts, and
    -log(0.3) vs -log(0.7) is small next to the evidence. Off.
  * snapping hurts -- team changes -0.04 / -0.02, stoppage edges ~0 -- the decoded boundaries
    already sit better than these rules put them. Off.
  * costs: team 4-10 and dead 10-40 are a plateau; team 6, dead 40 chosen.
So the remaining errors are in the evidence, not the decoding: real possessions under 2 s
(25-30% of all on the external games) are almost never recovered.

Re-checked 2026-11-06 with the touch features (src/touches.py) on noise v3: team costs 4-10 are
still a plateau (0.673-0.682 IDSSE, 0.718-0.723 Metrica), so team 6, dead 40 stay.
"""

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

from . import controlled
from .bench import bench_games, source_of
from .features import SHORT_BALL_GAP, ball_geometry, fill_short_gaps, lagged_speed
from .match import A, CACHE_DIR, D, H, X
from .possession import nearest_by_team

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

CLIP = 1e-4
SNAP_WINDOW = 25          # frames, 1 s either way
SNAP_HOLD = 6             # seen frames the new team must stay nearest for a team-change snap
KICK_FROM, KICK_TO = 1.0, 3.0   # m/s: a still ball (below) starting to move (above) is a kick

TEAM_COSTS = (4.0, 6.0, 10.0)
DEAD_COSTS = (10.0, 20.0, 40.0)
SWITCH_RATE = 0.7         # share of stoppages that hand the ball over, external games

# The settings the submission uses, chosen on the external bench (see the docstring).
TEAM_COST, DEAD_COST, MEMORY, SNAP = 6.0, 40.0, False, False


def log_emissions(p_dead, p_home):
    p_dead = np.clip(p_dead, CLIP, 1 - CLIP)
    p_home = np.clip(p_home, CLIP, 1 - CLIP)
    alive = np.log1p(-p_dead)
    return alive + np.log(p_home), alive + np.log1p(-p_home), np.log(p_dead)


def viterbi4(e_home, e_away, e_dead, team, dead, keep, switch):
    """Best path over states 0 H, 1 A, 2 D-after-H, 3 D-after-A. Returns the label codes."""
    n = len(e_home)
    back = np.zeros((n, 4), dtype=np.int8)
    s = [0.0, 0.0, 0.0, 0.0]
    leave_keep, leave_switch = dead + keep, dead + switch
    for t, (eh, ea, ed) in enumerate(zip(e_home.tolist(), e_away.tolist(), e_dead.tolist())):
        h, a, dh, da = s
        # into H: stay, from A, restart after H (keep), restart after A (switch)
        best_h, from_h = h, 0
        if a - team > best_h:
            best_h, from_h = a - team, 1
        if dh - leave_keep > best_h:
            best_h, from_h = dh - leave_keep, 2
        if da - leave_switch > best_h:
            best_h, from_h = da - leave_switch, 3
        best_a, from_a = a, 1
        if h - team > best_a:
            best_a, from_a = h - team, 0
        if da - leave_keep > best_a:
            best_a, from_a = da - leave_keep, 3
        if dh - leave_switch > best_a:
            best_a, from_a = dh - leave_switch, 2
        best_dh, from_dh = (dh, 2) if dh >= h - dead else (h - dead, 0)
        best_da, from_da = (da, 3) if da >= a - dead else (a - dead, 1)
        back[t] = (from_h, from_a, from_dh, from_da)
        s = [best_h + eh, best_a + ea, best_dh + ed, best_da + ed]
    state = int(np.argmax(s))
    path = np.empty(n, dtype=np.int8)
    for t in range(n - 1, -1, -1):
        path[t] = state
        state = back[t, state]
    return np.array([H, A, D, D], dtype=np.int8)[path]


def decode(p_dead, p_home, period, team, dead, memory):
    """Label codes for a whole match, decoded within each half; X outside the halves."""
    keep, switch = (-np.log(1 - SWITCH_RATE), -np.log(SWITCH_RATE)) if memory else (0.0, 0.0)
    e_home, e_away, e_dead = log_emissions(p_dead, p_home)
    labels = np.full(len(p_dead), X, dtype=np.int8)
    for half in np.unique(period[period > 0]):
        frames = np.flatnonzero(period == half)
        span = slice(frames[0], frames[-1] + 1)
        labels[span] = viterbi4(e_home[span], e_away[span], e_dead[span], team, dead, keep,
                                switch)
    return labels


# -- snapping ----------------------------------------------------------------------------------

def snap_context(match):
    """Per-frame signals the snapping rules read: nearest team, boundary distance, ball speed."""
    bx = fill_short_gaps(match.bx, SHORT_BALL_GAP)
    by = fill_short_gaps(match.by, SHORT_BALL_GAP)
    dist = np.hypot(match.px - bx[:, None], match.py - by[:, None])
    d_home, d_away = nearest_by_team(dist, match.team, 0)[0], nearest_by_team(dist, match.team, 1)[0]
    seen = ~np.isnan(bx)
    nearest = np.full(match.n_frames, -1, dtype=np.int8)
    both = seen & ~(np.isnan(d_home) & np.isnan(d_away))
    nearest[both] = np.where(np.fmin(d_home, np.inf)[both] <= np.fmin(d_away, np.inf)[both], 0, 1)
    return {"nearest": nearest,
            "boundary": ball_geometry(bx, by, match.pitch)["boundary_dist"],
            "speed": lagged_speed(bx, by, 6, match.fps)}


def _team_change_frame(nearest, team, low, high):
    """First frame in [low, high) where `team` becomes nearest and holds it for SNAP_HOLD seen
    frames, or None."""
    for f in range(low, high):
        if nearest[f] != team:
            continue
        window = nearest[f:f + 4 * SNAP_HOLD]
        seen = window[window >= 0]
        if len(seen) >= SNAP_HOLD and np.all(seen[:SNAP_HOLD] == team):
            return f
    return None


def snap(labels, ctx, period, team_changes=True, dead_edges=True):
    """Move each boundary onto its event, never past a neighbouring boundary or out of a half."""
    out = labels.copy()
    changes = np.flatnonzero(out[1:] != out[:-1]) + 1
    for i, t in enumerate(changes):
        before, after = out[t - 1], out[t]
        if X in (before, after) or period[t - 1] != period[t]:
            continue
        low = max(t - SNAP_WINDOW, changes[i - 1] + 1 if i else 0)
        high = min(t + SNAP_WINDOW, changes[i + 1] - 1 if i + 1 < len(changes) else len(out))
        target = None
        if team_changes and before in (H, A) and after in (H, A):
            target = _team_change_frame(ctx["nearest"], 0 if after == H else 1, low, high)
        elif dead_edges and after == D:
            out_frames = np.flatnonzero(ctx["boundary"][low:high] < 0)
            target = low + out_frames[0] if len(out_frames) else None
        elif dead_edges and before == D:
            speed = ctx["speed"][low:high]
            kicks = np.flatnonzero((speed[1:] > KICK_TO) & (speed[:-1] < KICK_FROM))
            target = low + kicks[0] + 1 if len(kicks) else None
        if target is None or target == t:
            continue
        if target < t:
            out[target:t] = after
        else:
            out[t:target] = before
    return out


# -- bench -------------------------------------------------------------------------------------

def oof(match, team_from=None):
    """Out-of-fold P(dead) and P(home). team_from names the game whose team-model predictions to
    use -- by default this one; pass a game's other-convention name to score a team model trained
    on one possession definition against labels in another."""
    p_dead = np.load(os.path.join(CACHE_DIR, "pdead_oof_{}.npy".format(
        controlled.base_name(match.name))))
    p_home = np.load(os.path.join(CACHE_DIR, "phome_oof_{}.npy".format(team_from or match.name)))
    return p_dead, p_home


def score(match, labels, **tags):
    result = pm.evaluate(match.labels, labels)
    return dict(tags, source=source_of(match.name), game=match.name,
                spellF1=result["spell_f1_mean"], macroF1=result["macro_f1"],
                n_pred=result["n_pred_spells"], n_gt=result["n_gt_spells"])


def reference(match, p_dead, p_home):
    """Steps 2 + 3 as they stand: dead ball first (cost 40), then team (cost 6)."""
    from . import deadball, possession
    dead = deadball.decode(p_dead, match.period, 40.0)
    return possession.decode_teams(p_home, dead, match.period, 6.0)


def grid(games):
    rows = []
    for match in games:
        p_dead, p_home = oof(match)
        rows.append(score(match, reference(match, p_dead, p_home), variant="steps 2+3"))
        for team, dead, memory in itertools.product(TEAM_COSTS, DEAD_COSTS, (False, True)):
            labels = decode(p_dead, p_home, match.period, team, dead, memory)
            rows.append(score(match, labels, variant="joint", team=team, dead=dead,
                              memory=memory))
        print("  {}".format(match.name), flush=True)
    return pd.DataFrame(rows)


def snapping(games, team, dead, memory):
    rows = []
    for match in games:
        p_dead, p_home = oof(match)
        labels = decode(p_dead, p_home, match.period, team, dead, memory)
        ctx = snap_context(match)
        for name, kwargs in (("no snap", None), ("team changes", dict(dead_edges=False)),
                             ("dead edges", dict(team_changes=False)), ("both", {})):
            snapped = labels if kwargs is None else snap(labels, ctx, match.period, **kwargs)
            rows.append(score(match, snapped, snap=name))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--snap", nargs=3, metavar=("TEAM", "DEAD", "MEMORY"),
                        help="only run the snapping comparison at these costs (memory 0/1)")
    parser.add_argument("--labels", choices=controlled.CONVENTIONS, default="last_touch",
                        help="possession definition of the external labels (src/controlled.py)")
    args = parser.parse_args()
    games = bench_games(convention=args.labels)
    columns = ["spellF1", "macroF1", "n_pred", "n_gt"]
    if args.snap:
        team, dead, memory = float(args.snap[0]), float(args.snap[1]), args.snap[2] == "1"
        table = snapping(games, team, dead, memory)
        print(table.groupby(["source", "snap"])[columns].mean().to_string(float_format="%.4f"))
    else:
        table = grid(games)
        ref = table[table.variant == "steps 2+3"].groupby("source")[columns].mean()
        print("== steps 2+3 as they stand")
        print(ref.to_string(float_format="%.4f"))
        joint = table[table.variant == "joint"]
        print("== joint decoder")
        print(joint.groupby(["source", "memory", "team", "dead"])[columns].mean()
              .to_string(float_format="%.4f"))
