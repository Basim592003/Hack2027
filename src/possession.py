"""Team-in-possession evidence: P(home has the ball) on every live frame.

A gradient-boosted classifier trained only on the labelled external games (RULES.md, clarified
2026-11-03: training and tuning use external data only). Its features come in three families:

  ball      each team's nearest player, players within CONTROL_RADIUS, whether the nearest player
            moves with the ball -- and all of it averaged over the past and next seconds, which
            is how "controlled possession" enters: a single touch barely moves a 1-4 s average,
            keeping the ball does (CHALLENGE.md: a touch or deflection without control does not
            change possession).
  shape     each team's height up the pitch, length, width and forward speed, plus who was
            nearest the ball when it was last seen and when it is next seen. This is what is left
            on the 41% of frames without a ball.
  restart   after each stoppage found by src/deadball.py: where the ball restarts -- corners go to
            the attacking team, goal kicks to the defending one, by the Laws of the Game -- and
            whose player is nearest it.
  touches   the team of the last and next kick of the ball, kicks in a row by one team, and the
            last and next team alone within CONTROL_RADIUS (src/touches.py). Sharp where the
            averages above are blurred: at the moment the ball changes hands.

Every feature is computed from one team's point of view ("own" vs "opp", pitch oriented so own
attacks +x) and the model is trained on both teams' views of every frame, so it cannot learn that
the home side usually has the ball. At inference the two views are averaged, so the output is
symmetric by construction.

    python -m src.possession              # leave-one-game-out on the external games
    python -m src.possession --bar        # train on everything, apply to BAR-ATH, aggregates only
    python -m src.possession --no-touches # without the touch family, for comparison
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from . import controlled, deadball, touches
from .bench import bench_games, source_of
from .features import (CONTEXT_CAP, SHORT_BALL_GAP, ball_geometry, fill_short_gaps, nan_row_mean,
                       rolling_mean, team_sides)
from .match import A, CACHE_DIR, D, H, X

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

CONTROL_RADIUS = 1.5     # metres: a player this close to the ball can be playing it
VELOCITY_LAG = 6         # frames, for ball and player velocity vectors
SHAPE_LAG = 24           # frames, ~1 s, for team forward speed
RESTART_WINDOW = 25      # frames either side of a stoppage's end in which to find the ball
DEAD_PENALTY = 40.0      # step 2's switch cost, chosen by cross-validation there
TEAM_PENALTIES = (0.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0)
WINDOWS = (("c1", 12, 12), ("c4", 50, 50), ("past1", 25, 0), ("past4", 100, 0),
           ("next1", 0, 25), ("next4", 0, 100))
TRAIN_STRIDE = 5
MODEL_PARAMS = dict(max_iter=300, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=200,
                    l2_regularization=1.0, random_state=0)
TOUCHES = True           # the touch feature family; --no-touches turns it off for comparison
TURNS = True             # change in each team's forward speed; --no-turns turns it off


def lagged_delta(values, lag):
    """values(t + lag/2) - values(t - lag/2), NaN at the ends; works on (T,) and (T, P)."""
    half = lag // 2
    out = np.full(values.shape, np.nan, dtype=np.float64)
    n = len(values)
    if n > lag:
        out[half:half + n - lag] = values[lag:] - values[:n - lag]
    return out


def nearest_by_team(dist, team, k):
    """(distance, slot) of team k's nearest player to the ball per frame; NaN / -1 if none."""
    masked = np.where(np.isnan(dist) | (team != k)[None, :], np.inf, dist)
    slot = np.argmin(masked, axis=1)
    best = masked[np.arange(len(masked)), slot]
    return np.where(np.isfinite(best), best, np.nan), np.where(np.isfinite(best), slot, -1)


# -- restarts ----------------------------------------------------------------------------------

def restart_team(match, dead, sides, nearest_team):
    """Per frame, for the latest restart: (rule team, nearest team, frames since), -1 if unknown.

    rule team: corner -> the team attacking that goal, goal kick -> the team defending it, from
    where the ball is around the end of the stoppage. Neither needs to know who had the ball
    before, which is what makes them safe to use.
    """
    n = match.n_frames
    length, width = match.pitch
    rule = np.full(n, -1, dtype=np.int8)
    near = np.full(n, -1, dtype=np.int8)
    since = np.full(n, CONTEXT_CAP, dtype=np.float64)
    ends = np.flatnonzero(np.diff(dead.astype(np.int8)) == -1) + 1
    bounds = list(ends) + [n]
    for end, following in zip(ends, bounds[1:]):
        window = np.arange(max(end - RESTART_WINDOW, 0), min(end + RESTART_WINDOW, n))
        seen = window[~np.isnan(match.bx[window])]
        team_rule, team_near = -1, -1
        if len(seen):
            frame = seen[np.argmin(np.abs(seen - end))]
            x, y = float(match.bx[frame]), float(match.by[frame])
            geometry = ball_geometry(np.array([x]), np.array([y]), match.pitch)
            # The team defending the goal at this end: side -1 defends x=0, +1 defends x=length.
            end_side = 1 if x > length / 2 else -1
            defender = 0 if sides[0][frame] == end_side else 1
            if geometry["near_corner"][0]:
                team_rule = 1 - defender
            elif geometry["in_goal_area"][0]:
                team_rule = defender
            team_near = nearest_team[frame]
        span = slice(end, following)
        rule[span] = team_rule
        near[span] = team_near
        since[span] = np.arange(following - end)
    return rule, near, since


# -- features ----------------------------------------------------------------------------------

def match_context(match, dead):
    """Everything both points of view share, computed once per match."""
    fps = match.fps
    bx = fill_short_gaps(match.bx, SHORT_BALL_GAP)
    by = fill_short_gaps(match.by, SHORT_BALL_GAP)
    px, py = match.px.astype(np.float64), match.py.astype(np.float64)
    dist = np.hypot(px - bx[:, None], py - by[:, None])
    near = [nearest_by_team(dist, match.team, k) for k in (0, 1)]
    close = [np.sum((dist < CONTROL_RADIUS) & (match.team == k)[None, :], axis=1) for k in (0, 1)]
    seen = ~np.isnan(bx)
    nearest_team = np.where(~seen | (np.isnan(near[0][0]) & np.isnan(near[1][0])), -1,
                            np.where(np.fmin(near[0][0], np.inf) <= np.fmin(near[1][0], np.inf),
                                     0, 1)).astype(np.int8)

    bvx = lagged_delta(bx, VELOCITY_LAG) * fps / VELOCITY_LAG
    bvy = lagged_delta(by, VELOCITY_LAG) * fps / VELOCITY_LAG
    pvx = lagged_delta(px, VELOCITY_LAG) * fps / VELOCITY_LAG
    pvy = lagged_delta(py, VELOCITY_LAG) * fps / VELOCITY_LAG
    rows = np.arange(match.n_frames)
    comove = []
    for _, slot in near:
        valid = slot >= 0
        safe = np.where(valid, slot, 0)
        gap = np.hypot(bvx - pvx[rows, safe], bvy - pvy[rows, safe])
        comove.append(np.where(valid & seen, gap, np.nan))

    sides = team_sides(match)
    rule, near_restart, since_restart = restart_team(match, dead, sides, nearest_team)
    index = np.arange(match.n_frames)
    last = np.maximum.accumulate(np.where(seen, index, -1))
    following = np.minimum.accumulate(np.where(seen, index, match.n_frames)[::-1])[::-1]
    touch = touches.touch_context(match.bx, match.by, dist, close, match.team, fps) \
        if TOUCHES else None
    return dict(bx=bx, by=by, px=px, py=py, seen=seen, near=near, close=close, comove=comove,
                nearest_team=nearest_team, sides=sides, rule=rule, near_restart=near_restart,
                since_restart=since_restart, last=last, following=following, touch=touch)


def view_features(match, ctx, me):
    """Features from team `me`'s point of view, pitch oriented so `me` attacks +x."""
    fps = match.fps
    length, width = match.pitch
    opp = 1 - me
    n = match.n_frames
    flip = ctx["sides"][me] == 1          # me defends x=length, so it attacks -x
    u_ball = np.where(flip, length - ctx["bx"], ctx["bx"])
    u_players = np.where(flip[:, None], length - ctx["px"], ctx["px"])
    feats = {}

    def signed(team_array):
        """+1 where the team is me, -1 where it is the opponent, NaN where unknown."""
        return np.where(team_array < 0, np.nan, np.where(team_array == me, 1.0, -1.0))

    # -- ball ------------------------------------------------------------------------------
    d_own, d_opp = ctx["near"][me][0], ctx["near"][opp][0]
    feats["d_own"], feats["d_opp"] = d_own, d_opp
    feats["d_diff"] = d_own - d_opp
    feats["close_own"] = np.where(ctx["seen"], ctx["close"][me], np.nan)
    feats["close_opp"] = np.where(ctx["seen"], ctx["close"][opp], np.nan)
    feats["comove_own"], feats["comove_opp"] = ctx["comove"][me], ctx["comove"][opp]
    feats["ball_u"] = u_ball
    feats["ball_y_abs"] = np.abs(ctx["by"] - width / 2)
    nearest = signed(ctx["nearest_team"])
    control = np.where(ctx["seen"], (d_own < CONTROL_RADIUS).astype(float)
                       - (d_opp < CONTROL_RADIUS).astype(float), np.nan)
    for tag, before, after in WINDOWS:
        feats["nearest_" + tag] = rolling_mean(nearest, before, after)
        feats["control_" + tag] = rolling_mean(control, before, after)
        feats["d_diff_" + tag] = rolling_mean(feats["d_diff"], before, after)

    # Where the ball was last seen and where it reappears, and whose player was nearest.
    last, following = ctx["last"], ctx["following"]
    has_last, has_next = last >= 0, following < n
    last_c, next_c = np.clip(last, 0, n - 1), np.clip(following, 0, n - 1)
    index = np.arange(n)
    feats["last_nearest"] = np.where(has_last, nearest[last_c], np.nan)
    feats["next_nearest"] = np.where(has_next, nearest[next_c], np.nan)
    feats["last_ball_u"] = np.where(has_last, u_ball[last_c], np.nan)
    feats["next_ball_u"] = np.where(has_next, u_ball[next_c], np.nan)
    feats["log_since_seen"] = np.log1p(np.minimum(np.where(has_last, index - last, CONTEXT_CAP),
                                                  CONTEXT_CAP))
    feats["log_until_seen"] = np.log1p(np.minimum(np.where(has_next, following - index,
                                                           CONTEXT_CAP), CONTEXT_CAP))

    # -- team shape ------------------------------------------------------------------------
    forward = lagged_delta(u_players, SHAPE_LAG) * fps / SHAPE_LAG
    for name, k, sign in (("own", me, 1.0), ("opp", opp, -1.0)):
        members = (match.team == k)[None, :] & ~np.isnan(u_players)
        u = np.where(members, u_players, np.nan)
        y = np.where(members, ctx["py"], np.nan)
        feats[name + "_height"] = nan_row_mean(u)
        feats[name + "_length"] = nan_row_mean(np.abs(u - feats[name + "_height"][:, None]))
        feats[name + "_width"] = nan_row_mean(np.abs(y - nan_row_mean(y)[:, None]))
        feats[name + "_forward"] = nan_row_mean(np.where(members, sign * forward, np.nan))
        feats[name + "_n"] = members.sum(axis=1).astype(np.float64)
    feats["height_gap"] = feats["own_height"] - feats["opp_height"]
    feats["forward_gap"] = feats["own_forward"] - feats["opp_forward"]
    feats["spread_gap"] = (feats["own_length"] + feats["own_width"]
                           - feats["opp_length"] - feats["opp_width"])
    for name in ("height_gap", "forward_gap", "spread_gap", "own_forward", "opp_forward"):
        for tag, before, after in (("c4", 50, 50), ("past4", 100, 0), ("next4", 0, 100)):
            feats["{}_{}".format(name, tag)] = rolling_mean(feats[name], before, after)
    if TURNS:
        # A team that loses the ball stops pushing up and drops back; one that wins it pushes on:
        # forward speed over the next 2 / 4 s minus over the last 2 / 4 s.
        for name in ("forward_gap", "own_forward", "opp_forward"):
            for span in (50, 100):
                feats["{}_turn{}".format(name, span // 25)] = (
                    rolling_mean(feats[name], 0, span) - rolling_mean(feats[name], span, 0))

    # -- restart ---------------------------------------------------------------------------
    feats["restart_rule"] = np.where(ctx["rule"] < 0, 0.0, signed(ctx["rule"]))
    feats["restart_nearest"] = np.where(ctx["near_restart"] < 0, 0.0,
                                        signed(ctx["near_restart"]))
    feats["log_since_restart"] = np.log1p(ctx["since_restart"])

    # -- touches ---------------------------------------------------------------------------
    touch = ctx["touch"]
    if touch is not None:
        for name in ("kick_last", "kick_next", "kick_prev", "close_last", "close_next"):
            feats[name] = signed(touch[name])
        feats["kick_last_streak"] = feats["kick_last"] * touch["kick_last_streak"]
        feats["kick_next_streak"] = feats["kick_next"] * touch["kick_next_streak"]
        for name in ("kick_since", "kick_until", "close_since", "close_until"):
            feats["log_" + name] = np.log1p(touch[name])
        for name, k in (("own", me), ("opp", opp)):
            feats["kicks_{}_past2".format(name)] = touch["kicks_past_{}".format(k)]
            feats["kicks_{}_next2".format(name)] = touch["kicks_next_{}".format(k)]

    return {k: np.asarray(v, dtype=np.float32) for k, v in feats.items()}


def both_views(match, dead):
    ctx = match_context(match, dead)
    names = None
    views = []
    for me in (0, 1):
        feats = view_features(match, ctx, me)
        names = sorted(feats) if names is None else names
        views.append(np.stack([feats[k] for k in names], axis=1))
    return names, views


def predicted_dead(match, p_dead_path):
    return deadball.decode(np.load(p_dead_path), match.period, DEAD_PENALTY)


# -- training and decoding ---------------------------------------------------------------------

def training_rows(tables, names_in):
    """Both points of view of every TRAIN_STRIDE-th live frame. y = 'this view's team has it'."""
    xs, ys = [], []
    for name in names_in:
        views, labels, _ = tables[name]
        keep = np.flatnonzero((labels == H) | (labels == A))[::TRAIN_STRIDE]
        for me, team_label in ((0, H), (1, A)):
            xs.append(views[me][keep])
            ys.append(labels[keep] == team_label)
    return np.concatenate(xs), np.concatenate(ys)


def fit(tables, names_in):
    x, y = training_rows(tables, names_in)
    return HistGradientBoostingClassifier(**MODEL_PARAMS).fit(x, y)


def p_home(model, views):
    """Symmetric P(home has the ball): the home view's answer averaged with the away view's."""
    return 0.5 * (model.predict_proba(views[0])[:, 1] + 1 - model.predict_proba(views[1])[:, 1])


def decode_teams(p, dead, period, penalty):
    """Label codes: D where dead, else the team from a 2-state Viterbi run on each live stretch.

    Each live stretch starts free -- who restarts is evidence (restart features), not a switch --
    and every change of team inside a stretch costs `penalty`.
    """
    logodds = np.log(np.clip(p, deadball.CLIP, 1 - deadball.CLIP)
                     / np.clip(1 - p, deadball.CLIP, 1 - deadball.CLIP))
    labels = np.full(len(p), X, dtype=np.int8)
    live = (~dead) & (period > 0)
    labels[dead & (period > 0)] = D
    edges = np.diff(np.concatenate(([0], live.astype(np.int8), [0])))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        # viterbi2's "dead" state stands in for "home" here: +logodds per frame in it.
        home = deadball.viterbi2(logodds[start:stop], penalty) if penalty > 0 \
            else logodds[start:stop] > 0
        labels[start:stop] = np.where(home, H, A)
    return labels


def score_game(match, p, dead, penalty):
    labels = decode_teams(p, dead, match.period, penalty)
    result = pm.evaluate(match.labels, labels)
    return {"source": source_of(match.name), "game": match.name, "penalty": penalty,
            "spellF1": result["spell_f1_mean"], "macroF1": result["macro_f1"],
            "teamAcc": result["team_acc_gt_alive"], "n_pred": result["n_pred_spells"],
            "n_gt": result["n_gt_spells"]}


def build_tables(games):
    tables, names = {}, None
    for match in games:
        dead = predicted_dead(match, os.path.join(CACHE_DIR, "pdead_oof_{}.npy".format(
            controlled.base_name(match.name))))
        names, views = both_views(match, dead)
        tables[match.name] = (views, match.labels, dead)
    return tables, names


def cross_validate(games, tables):
    groups = sorted({deadball.clean_name(m) for m in games})
    p = {}
    for held_out in groups:
        model = fit(tables, [m.name for m in games if deadball.clean_name(m) != held_out])
        for match in games:
            if deadball.clean_name(match) == held_out:
                p[match.name] = p_home(model, tables[match.name][0])
                np.save(os.path.join(CACHE_DIR, "phome_oof_{}.npy".format(match.name)),
                        p[match.name].astype(np.float32))
        print("  fold {:<14} done".format(held_out), flush=True)
    return p


def report(games, tables, p):
    rows = [score_game(m, p[m.name], tables[m.name][2], pen) for m in games
            for pen in TEAM_PENALTIES]
    table = pd.DataFrame(rows)
    columns = ["spellF1", "macroF1", "teamAcc", "n_pred", "n_gt"]
    print("== step 2 dead ball + step 3 team evidence (leave one game out)")
    print(table.groupby(["source", "penalty"])[columns].mean().to_string(float_format="%.4f"))
    reference = pd.DataFrame([
        {"source": source_of(m.name),
         "spellF1": pm.evaluate(m.labels, np.where(m.period > 0, deadball.with_baseline_teams(
             m, tables[m.name][2]), X).astype(np.int8))["spell_f1_mean"]} for m in games])
    print("== reference: same dead ball, baseline's team guess")
    print(reference.groupby("source").mean().to_string(float_format="%.4f"))
    return table


def apply_to_bar(games, tables, penalty):
    from .sources import bar_ath
    model = fit(tables, [m.name for m in games])
    bar = bar_ath()
    dead = predicted_dead(bar, os.path.join(CACHE_DIR, "pdead_BAR-ATH.npy"))
    _, views = both_views(bar, dead)
    p = p_home(model, views)
    np.save(os.path.join(CACHE_DIR, "phome_BAR-ATH.npy"), p.astype(np.float32))
    labels = decode_teams(p, dead, bar.period, penalty)
    graded = bar.period > 0
    runs = pm.valid_runs(np.where(graded, D, X).astype(np.int8))
    spells = pm.possession_spells(labels, runs)
    lengths = np.array([b - a for a, b, _ in spells])
    print("== BAR-ATH, aggregates only (team penalty {})".format(penalty))
    print("  possessions          {}".format(len(spells)))
    print("  mean possession      {:.1f} s".format(lengths.mean() / bar.fps))
    print("  home share of live   {:.1f}%".format(
        100 * np.mean(labels[graded & (labels != D)] == H)))
    return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bar", type=float, default=None, metavar="PENALTY",
                        help="train on all external games and apply to BAR-ATH")
    parser.add_argument("--labels", choices=controlled.CONVENTIONS, default="last_touch",
                        help="possession definition of the external labels (src/controlled.py)")
    parser.add_argument("--no-touches", action="store_true", help="leave out the touch features")
    parser.add_argument("--no-turns", action="store_true",
                        help="leave out the change in each team's forward speed")
    args = parser.parse_args()
    TOUCHES = not args.no_touches
    TURNS = not args.no_turns

    games = bench_games(convention=args.labels)
    tables, names = build_tables(games)
    print("{} games, {} features per view".format(len(games), len(names)), flush=True)
    if args.bar is not None:
        apply_to_bar(games, tables, args.bar)
    else:
        report(games, tables, cross_validate(games, tables))
