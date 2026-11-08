"""A/B harness on the external bench: tagged out-of-fold caches, both label conventions, paired comparisons.

Every choice between two versions of the pipeline is made here (docs/sprint_checklist.md). A variant is
whatever the code and flags are when it runs:

    python -m src.experiment run NAME        # out-of-fold P(dead) and P(home), joint decode, scores
    python -m src.experiment compare A B     # two saved runs side by side, per source
    python -m src.experiment oracles         # headroom: true D, true team, perfect switch timing
    python -m src.experiment crossbench --train 4 --test 5   # trained on one noise version, scored on another

Scoring. BAR-ATH's answer key counts *controlled* possession (CHALLENGE.md), and the external labels give
it to the last toucher, so every decode is scored against both: the labels as they are ("lt") and
controlled.controlled_labels of them ("ctl", the primary number). Same functions as
possession_metrics.evaluate, minus the extras it computes that nothing here reads.

Caches. Out-of-fold predictions are tagged with a hash of the source of the modules that make them and
of the settings they were made with (deadball.oof_path, possession.oof_path), and a manifest is written
only after the last fold. A changed model therefore recomputes instead of silently reading another
version's files, and an interrupted run leaves nothing that looks finished. Results go to
data/cache/experiments/NAME.csv (gitignored with the rest of data/).

External games only; nothing here touches BAR-ATH.
"""

import argparse
import hashlib
import inspect
import json
import os
import sys

import numpy as np
import pandas as pd

from . import controlled, deadball, decode, features, noise, possession, touches
from .bench import bench_games, source_of
from .match import A, CACHE_DIR, D, H, X, Match

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

OOF_DIR = os.path.join(CACHE_DIR, "oof")
RESULTS_DIR = os.path.join(CACHE_DIR, "experiments")
CONVENTIONS = (("lt", None), ("ctl", controlled.controlled_labels))
SHORT_SPELL = 50          # frames: "spells under 2 s"
OFFSET_WINDOW = 50        # frames either side in which a predicted change can answer a real one
ON_TIME = 8               # frames: the scorer's table puts 8 frames late at 0.89 of a possession


# -- tagged out-of-fold caches -------------------------------------------------------------------

def signature(modules, **settings):
    """Short hash of the source of the modules that produce an array, plus its settings."""
    digest = hashlib.sha1()
    for module in modules:
        digest.update(inspect.getsource(module).encode("utf-8"))
    digest.update(json.dumps(settings, sort_keys=True, default=str).encode("utf-8"))
    return digest.hexdigest()[:10]


def _manifest(kind, tag):
    return os.path.join(OOF_DIR, "{}@{}.json".format(kind, tag))


def _finished(kind, tag, paths):
    return os.path.isfile(_manifest(kind, tag)) and all(os.path.isfile(p) for p in paths)


def _finish(kind, tag, games, settings):
    with open(_manifest(kind, tag), "w") as handle:
        json.dump({"kind": kind, "tag": tag, "games": [m.name for m in games],
                   "settings": settings}, handle, indent=1, sort_keys=True, default=str)


def dead_oof(games):
    """Tag of the out-of-fold P(dead) for these games, computing it only if it is not cached."""
    settings = dict(games=[m.name for m in games], params=deadball.MODEL_PARAMS,
                    stride=deadball.TRAIN_STRIDE)
    tag = signature((features, touches, deadball), **settings)
    paths = [deadball.oof_path(m.name, tag) for m in games]
    if _finished("pdead", tag, paths):
        print("P(dead) out-of-fold: cached @{}".format(tag), flush=True)
        return tag
    print("P(dead) out-of-fold @{}".format(tag), flush=True)
    table, _ = deadball.tables(games)
    deadball.out_of_fold(games, table, tag)
    _finish("pdead", tag, games, settings)
    return tag


def home_oof(games, dead_tag):
    """Tag of the out-of-fold P(home), on top of the given P(dead) caches."""
    settings = dict(dead=dead_tag, games=[m.name for m in games], params=possession.MODEL_PARAMS,
                    stride=possession.TRAIN_STRIDE, touches=possession.TOUCHES,
                    turns=possession.TURNS)
    tag = signature((features, touches, deadball, possession), **settings)
    paths = [possession.oof_path(m.name, tag) for m in games]
    if _finished("phome", tag, paths):
        print("P(home) out-of-fold: cached @{}".format(tag), flush=True)
        return tag
    print("P(home) out-of-fold @{}".format(tag), flush=True)
    tables, _ = possession.build_tables(games, dead_tag)
    possession.cross_validate(games, tables, tag)
    _finish("phome", tag, games, settings)
    return tag


# -- scoring ---------------------------------------------------------------------------------------

def score(truth, pred):
    """spellF1 and macroF1 exactly as possession_metrics.evaluate computes them."""
    runs = pm.valid_runs(truth)
    spells = pm.spell_scores(truth, pm.fill_abstentions(pred, runs), runs)
    return {"spellF1": spells["mean_f1"],
            "macroF1": pm.macro_f1(pm.scored_confusion(pm.confusion(truth, pred))),
            "n_pred": spells["n_pred"], "n_gt": spells["n_gt"]}


def score_both(match, labels):
    """score() against the labels as they are and against their controlled relabelling."""
    out = {}
    for suffix, relabel in CONVENTIONS:
        truth = match.labels if relabel is None else relabel(match.labels)
        out.update({"{}_{}".format(k, suffix): v for k, v in score(truth, labels).items()})
    return out


def in_play_changes(codes, runs):
    """(frames, new team) where the team changes between two live frames inside a valid run."""
    frames, teams = [], []
    for start, stop in runs:
        window = codes[start:stop]
        live = (window == H) | (window == A)
        at = np.flatnonzero(live[1:] & live[:-1] & (window[1:] != window[:-1])) + 1
        frames.append(start + at)
        teams.append(window[at])
    return np.concatenate(frames), np.concatenate(teams)


def diagnostics(truth, pred, p_home):
    """Where the remaining errors are, for one label set:
    recovered    real possessions that start with a turnover in play, matched at IoU 0.5
    p_true       mean P(true team) over in-play possessions shorter than SHORT_SPELL
    offset       median frames from a real in-play change to the nearest predicted one with the
                 same new team within OFFSET_WINDOW (positive = late), and the share answered
                 within ON_TIME frames
    """
    runs = pm.valid_runs(truth)
    filled = pm.fill_abstentions(pred, runs)
    gt_spells = pm.possession_spells(truth, runs)
    pred_spells = pm.possession_spells(filled, runs)
    matched = pm.matched_gt_flags(pm.spell_overlaps(gt_spells, pred_spells), len(gt_spells), 0.5)
    run_start = {start for start, _ in runs}
    in_play = np.array([a not in run_start and truth[a - 1] in (H, A) for a, _, _ in gt_spells])

    short = [(a, b, k) for (a, b, k), play in zip(gt_spells, in_play) if play and b - a < SHORT_SPELL]
    p_true = [p_home[a:b] if k == H else 1 - p_home[a:b] for a, b, k in short]

    true_at, true_team = in_play_changes(truth, runs)
    pred_at, pred_team = in_play_changes(filled, runs)
    offsets = []
    for frame, team in zip(true_at, true_team):
        near = pred_at[(pred_team == team) & (np.abs(pred_at - frame) <= OFFSET_WINDOW)]
        if len(near):
            offsets.append(int(near[np.argmin(np.abs(near - frame))] - frame))
    offsets = np.array(offsets)
    return {
        "recovered": float(matched[in_play].mean()) if in_play.any() else np.nan,
        "p_true": float(np.concatenate(p_true).mean()) if p_true else np.nan,
        "offset": float(np.median(offsets)) if len(offsets) else np.nan,
        "on_time": float(np.sum(np.abs(offsets) <= ON_TIME) / len(true_at)) if len(true_at) else np.nan,
    }


def game_row(match, labels, p_home):
    row = {"source": source_of(match.name), "game": deadball.clean_name(match), "copy": match.name}
    row.update(score_both(match, labels))
    for suffix, relabel in CONVENTIONS:
        truth = match.labels if relabel is None else relabel(match.labels)
        row.update({"{}_{}".format(k, suffix): v
                    for k, v in diagnostics(truth, labels, p_home).items()})
    return row


# -- runs and comparisons --------------------------------------------------------------------------

def run(name, games=None):
    """Out-of-fold predictions (cached by tag), the joint decoder as src.run uses it, scores."""
    games = games if games is not None else bench_games()
    dead_tag = dead_oof(games)
    home_tag = home_oof(games, dead_tag)
    rows = []
    for match in games:
        p_dead, p_home = decode.oof(match, dead_tag=dead_tag, home_tag=home_tag)
        labels = decode.decode(p_dead, p_home, match.period, decode.TEAM_COST, decode.DEAD_COST,
                               decode.MEMORY)
        labels = np.where(match.period > 0, labels, X).astype(np.int8)
        rows.append(dict(game_row(match, labels, p_home), dead_tag=dead_tag, home_tag=home_tag))
    table = pd.DataFrame(rows)
    save(name, table, "dead @{}, home @{}".format(dead_tag, home_tag))
    return table


SUMMARY = ["spellF1_ctl", "spellF1_lt", "macroF1_ctl", "n_pred_ctl", "n_gt_ctl", "n_gt_lt",
           "recovered_ctl", "p_true_ctl", "offset_ctl", "on_time_ctl"]


def summary(table):
    """Per-source means; sources are never pooled (their label conventions differ)."""
    return table.groupby("source")[SUMMARY].mean().T


def load(name):
    return pd.read_csv(os.path.join(RESULTS_DIR, "{}.csv".format(name)))


SMOOTHING_BANDS = (0.0, 4.0, 7.0, np.inf)   # frames; BAR-ATH's ball reads as about 7.5-8 (noise.py)


def ball_sd(copy):
    """The ball-filter width (frames) a degraded copy was made with: 0 before noise v5, NaN for a
    version other than the current one (noise_params follows today's settings only)."""
    clean, made = copy.split("~")[:2]                      # e.g. "v5live10s0"
    version = int(made[1:made.index("live")])
    if version < 5:
        return 0.0
    if version != noise.VERSION:
        return np.nan
    level, seed = int(made.split("live")[1][:2]) / 100, int(made.split("s")[-1])
    return noise.noise_params(clean, level, seed)["ball_sd"]


def by_smoothing(both):
    """Controlled spellF1, B against A, by the ball smoothing each copy was made with (all sources
    together: the BAR-ATH-like band holds few copies of each)."""
    sd = both["copy"].map(ball_sd)
    if not (sd > 0).any():
        return None
    diff = both["spellF1_ctl_b"] - both["spellF1_ctl_a"]
    band = pd.cut(sd, SMOOTHING_BANDS, right=False,
                  labels=["sd<4", "sd 4-7", "sd>=7 (BAR-ATH-like)"])
    grouped = diff.groupby(band, observed=True)
    return pd.DataFrame({"copies": grouped.size(), "diff": grouped.mean(),
                         "won": grouped.agg(lambda d: "{}/{}".format(int((d > 0).sum()), len(d)))})


def save(name, table, header):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    table.to_csv(os.path.join(RESULTS_DIR, "{}.csv".format(name)), index=False)
    print("== {} ({})".format(name, header))
    print(summary(table).to_string(float_format="%.4f"))


# -- one noise version against another --------------------------------------------------------

def games_at(version, current):
    """The bench at noise version `version`: `current` when that is noise.VERSION, otherwise the
    cached copies of the same games under the older version's names (they must be cached)."""
    if version == noise.VERSION:
        return current
    out = []
    for match in current:
        name = match.name.replace("~v{}".format(noise.VERSION), "~v{}".format(version))
        old = Match.load(name)
        old.name = name
        out.append(old)
    return out


def crossbench(train_version, test_version):
    """Train on one noise version's bench, score on another's, leaving one clean game out -- the
    like-for-like test for a simulator change. Each held-out copy goes through the pipeline exactly
    as BAR-ATH does in src/run.py. Saved as cross-vA-vB, comparable with `run` on the test bench."""
    current = bench_games()
    train, test = games_at(train_version, current), games_at(test_version, current)
    dead_tag = dead_oof(train)
    dead_table, dead_names = deadball.tables(train)
    team_tables, _ = possession.build_tables(train, dead_tag)
    rows = []
    for held_out in sorted({deadball.clean_name(m) for m in train}):
        names_in = [m.name for m in train if deadball.clean_name(m) != held_out]
        dead_model = deadball.fit(dead_table, names_in)
        team_model = possession.fit(team_tables, names_in)
        for match in test:
            if deadball.clean_name(match) != held_out:
                continue
            _, matrix = features.as_matrix(features.frame_features(match), dead_names)
            p_dead = dead_model.predict_proba(matrix)[:, 1]
            stoppages = deadball.decode(p_dead, match.period, possession.DEAD_PENALTY)
            _, views = possession.both_views(match, stoppages)
            p_home = possession.p_home(team_model, views)
            labels = decode.decode(p_dead, p_home, match.period, decode.TEAM_COST,
                                   decode.DEAD_COST, decode.MEMORY)
            labels = np.where(match.period > 0, labels, X).astype(np.int8)
            rows.append(dict(game_row(match, labels, p_home), dead_tag=dead_tag, home_tag="cross"))
        print("  fold {} done".format(held_out), flush=True)
    name = "cross-v{}-v{}".format(train_version, test_version)
    save(name, pd.DataFrame(rows), "trained on v{}, scored on v{}".format(train_version,
                                                                          test_version))


# -- oracles ---------------------------------------------------------------------------------------

HINT = 12                 # frames either side of a real in-play change where the oracle switches cheaply
HINT_COSTS = (0.5, 20.0)  # team cost near / away from those changes


def candidate_events(match):
    """Frames of the events item C1 formalises: kicks by either team, changes of the team alone
    within the control radius, and ball-gap edges (as src/diag/bench_gaps_candidates.py)."""
    bx = features.fill_short_gaps(match.bx, features.SHORT_BALL_GAP)
    by = features.fill_short_gaps(match.by, features.SHORT_BALL_GAP)
    dist = np.hypot(match.px - bx[:, None], match.py - by[:, None])
    kick_frames, _, _ = touches.kicks(match.bx.astype(np.float64), match.by.astype(np.float64),
                                      dist, match.team, match.fps)
    close = [np.sum((dist < possession.CONTROL_RADIUS) & (match.team == k)[None, :], axis=1)
             for k in (0, 1)]
    exclusive = np.flatnonzero((close[0] > 0) != (close[1] > 0))
    close_team = np.where(close[0][exclusive] > 0, 0, 1)
    close_changes = exclusive[1:][close_team[1:] != close_team[:-1]]
    seen = ~np.isnan(match.bx)
    gap_edges = np.flatnonzero(np.diff(seen.astype(np.int8)) != 0) + 1
    return np.unique(np.concatenate([kick_frames, close_changes, gap_edges]))


def hint_costs(changes, n):
    """Per-frame team cost that makes switching cheap only within HINT frames of the given changes."""
    cost = np.full(n, HINT_COSTS[1])
    for frame in changes:
        cost[max(frame - HINT, 0):frame + HINT] = HINT_COSTS[0]
    return cost


def oracles(games=None):
    """Headroom, from the current out-of-fold predictions, against both label sets:
    true D, true team, perfect switch timing (cheap switches only near real in-play changes), and
    perfect timing restricted to changes that have a candidate event within HINT frames -- the
    ceiling for an event-gated switch cost."""
    games = games if games is not None else bench_games()
    dead_tag = dead_oof(games)
    home_tag = home_oof(games, dead_tag)
    rows, identical = [], 0
    for match in games:
        p_dead, p_home = decode.oof(match, dead_tag=dead_tag, home_tag=home_tag)
        n = len(p_dead)

        def run_decoder(p_d, p_h, costs=None):
            return decode.decode(p_d, p_h, match.period, decode.TEAM_COST, decode.DEAD_COST,
                                 False, team_costs=costs)

        predicted = run_decoder(p_dead, p_home)
        identical += np.array_equal(predicted,
                                    run_decoder(p_dead, p_home, np.full(n, decode.TEAM_COST)))
        events = candidate_events(match)
        for suffix, relabel in CONVENTIONS:
            truth = match.labels if relabel is None else relabel(match.labels)
            live = (truth == H) | (truth == A)
            true_dead = np.where(truth == D, 1.0, np.where(live, 0.0, p_dead))
            true_home = np.where(truth == H, 1.0, np.where(truth == A, 0.0, p_home))
            changes, _ = in_play_changes(truth, pm.valid_runs(truth))
            at = np.searchsorted(events, changes)
            after = events[np.clip(at, 0, len(events) - 1)]
            before = events[np.clip(at - 1, 0, len(events) - 1)]
            near_event = np.minimum(np.abs(after - changes), np.abs(before - changes)) <= HINT
            cases = {
                "1 as predicted": predicted,
                "2 true D": run_decoder(true_dead, p_home),
                "3 true team": run_decoder(p_dead, true_home),
                "4 perfect timing": run_decoder(p_dead, p_home, hint_costs(changes, n)),
                "5 perfect timing at events": run_decoder(p_dead, p_home,
                                                          hint_costs(changes[near_event], n)),
            }
            for case, labels in cases.items():
                rows.append(dict(source=source_of(match.name), copy=match.name, labels=suffix,
                                 case=case, events_near=near_event.mean(), **score(truth, labels)))
        print("  {}".format(match.name), flush=True)
    print("viterbi4_costs at a constant cost == viterbi4 on {}/{} copies".format(identical,
                                                                                 len(games)))
    table = pd.DataFrame(rows)
    print("== spellF1 by case (rows: label set, case; columns: source)")
    print(table.groupby(["labels", "case", "source"]).spellF1.mean().unstack("source")
          .to_string(float_format="%.4f"))
    print("== share of real in-play changes with a candidate event within {} frames".format(HINT))
    print(table[table.case == "1 as predicted"].groupby(["labels", "source"]).events_near.mean()
          .unstack("source").to_string(float_format="%.3f"))
    return table


def compare(name_a, name_b):
    """B against A, per source and label set: means, paired differences, copies and games won."""
    a, b = load(name_a), load(name_b)
    both = a.merge(b, on=["source", "game", "copy"], suffixes=("_a", "_b"))
    if len(both) != len(a) or len(both) != len(b):
        raise SystemExit("{} and {} were not run on the same games".format(name_a, name_b))
    rows = []
    for source, group in both.groupby("source"):
        for suffix in ("ctl", "lt"):
            col = "spellF1_" + suffix
            diff = group[col + "_b"] - group[col + "_a"]
            per_game = (group.assign(d=diff).groupby("game").d.mean())
            rows.append({"source": source, "labels": suffix,
                         "A": group[col + "_a"].mean(), "B": group[col + "_b"].mean(),
                         "diff": diff.mean(), "min": diff.min(), "max": diff.max(),
                         "copies won": "{}/{}".format(int((diff > 0).sum()), len(diff)),
                         "games won": "{}/{}".format(int((per_game > 0).sum()), len(per_game))})
    print("== spellF1, {} (A) vs {} (B)".format(name_a, name_b))
    print(pd.DataFrame(rows).to_string(index=False, float_format="%.4f"))
    bands = by_smoothing(both)
    if bands is not None:
        print("== controlled spellF1 by ball smoothing, B - A")
        print(bands.to_string(float_format="%.4f"))
    print("== diagnostics, A | B")
    side = pd.concat({name_a: summary(a), name_b: summary(b)}, axis=1)
    print(side.to_string(float_format="%.4f"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run", help="score the pipeline as it stands").add_argument("name")
    cmp_parser = sub.add_parser("compare", help="compare two saved runs")
    cmp_parser.add_argument("a")
    cmp_parser.add_argument("b")
    sub.add_parser("oracles", help="headroom: true D, true team, perfect switch timing")
    cross_parser = sub.add_parser("crossbench", help="train on one noise version, score another")
    cross_parser.add_argument("--train", type=int, required=True, metavar="VERSION")
    cross_parser.add_argument("--test", type=int, required=True, metavar="VERSION")
    args = parser.parse_args()
    if args.command == "run":
        run(args.name)
    elif args.command == "compare":
        compare(args.a, args.b)
    elif args.command == "oracles":
        oracles()
    elif args.command == "crossbench":
        crossbench(args.train, args.test)
