"""Dead-ball detector: P(dead) per frame from src/features.py, decoded into stoppages.

A gradient-boosted classifier trained only on the labelled external games (src/bench.py), then a
two-state Viterbi with a cost per switch that turns per-frame probabilities into whole stoppages.
The switch cost is chosen by leave-one-game-out cross-validation on those games, never on BAR-ATH.

    python -m src.deadball            # cross-validate on the external games, print the table
    python -m src.deadball --transfer # train on one source, test on the other
    python -m src.deadball --bar      # train on everything, apply to BAR-ATH, print aggregates

Out-of-fold probabilities for every bench game and the BAR-ATH probabilities are cached under
data/cache/ so the decoder in step 4 is evaluated on predictions its games were not trained on.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from . import controlled
from .bench import bench_games, nearest_team, source_of
from .features import as_matrix, frame_features
from .match import CACHE_DIR, D, H, X

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "starter"))
import possession_metrics as pm  # noqa: E402

TRAIN_STRIDE = 5   # consecutive frames are near-duplicates; every 5th keeps the information
MODEL_PARAMS = dict(max_iter=300, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=200,
                    l2_regularization=1.0, random_state=0)
PENALTIES = (0.0, 5.0, 10.0, 20.0, 40.0, 80.0)
CLIP = 1e-4


def make_model():
    return HistGradientBoostingClassifier(**MODEL_PARAMS)


def clean_name(match):
    return match.name.split("~")[0]


def oof_path(name, tag=None):
    """Where a game's out-of-fold P(dead) is cached. Untagged files are the ones src.run and the
    command lines here write; src/experiment.py tags its files with a hash of what made them, so a
    changed model never reads another version's predictions."""
    base = controlled.base_name(name)
    if tag is None:
        return os.path.join(CACHE_DIR, "pdead_oof_{}.npy".format(base))
    return os.path.join(CACHE_DIR, "oof", "pdead_{}@{}.npy".format(base, tag))


# -- decoding ----------------------------------------------------------------------------------

def viterbi2(logodds, penalty):
    """Most likely alive/dead path: each frame scores +logodds if dead, each switch costs penalty."""
    n = len(logodds)
    from_other = np.zeros((n, 2), dtype=bool)
    alive, dead = 0.0, 0.0
    for t, e in enumerate(logodds.tolist()):
        alive_switch, dead_switch = dead - penalty, alive - penalty
        if alive_switch > alive:
            from_other[t, 0] = True
            new_alive = alive_switch
        else:
            new_alive = alive
        if dead_switch > dead:
            from_other[t, 1] = True
            new_dead = dead_switch
        else:
            new_dead = dead
        alive, dead = new_alive, new_dead + e
    path = np.zeros(n, dtype=bool)
    state = dead > alive
    for t in range(n - 1, -1, -1):
        path[t] = state
        if from_other[t, int(state)]:
            state = not state
    return path


def decode(p_dead, period, penalty):
    """Dead/alive per frame, decoded within each half separately."""
    logodds = np.log(np.clip(p_dead, CLIP, 1 - CLIP) / np.clip(1 - p_dead, CLIP, 1 - CLIP))
    dead = np.zeros(len(p_dead), dtype=bool)
    for half in np.unique(period[period > 0]):
        frames = np.flatnonzero(period == half)
        start, stop = frames[0], frames[-1] + 1
        dead[start:stop] = viterbi2(logodds[start:stop], penalty) if penalty > 0 \
            else logodds[start:stop] > 0
    return dead


# -- scoring -----------------------------------------------------------------------------------

def dead_spell_scores(gt_labels, dead):
    """Stoppages matched one-to-one by IoU, exactly as the scorer matches possessions."""
    gt = np.where(gt_labels == X, X, np.where(gt_labels == D, H, D)).astype(np.int8)
    pred = np.where(dead, H, D).astype(np.int8)
    return pm.spell_scores(gt, pred, pm.valid_runs(gt))


def with_baseline_teams(match, dead, window=50):
    """Full H/A/D labels: the starter baseline's team on live frames, our stoppages as D.

    The team comes from the baseline's nearest-player vote, carried forward through frames where
    the vote has no ball to go on -- so the only thing this changes is where D is.
    """
    import baseline
    letters = baseline.smooth(nearest_team(match), window)
    team = np.where(letters == "H", 0, np.where(letters == "A", 1, -1))
    known = np.maximum.accumulate(np.where(team >= 0, np.arange(len(team)), 0))
    team = team[known]
    team[team < 0] = 0
    return np.where(dead, D, team).astype(np.int8)


def score_game(match, p_dead, penalty):
    dead = decode(p_dead, match.period, penalty)
    labels = np.where(match.period > 0, with_baseline_teams(match, dead), X).astype(np.int8)
    full = pm.evaluate(match.labels, labels)
    spells = dead_spell_scores(match.labels, dead)
    return {
        "source": source_of(match.name),
        "game": match.name,
        "penalty": penalty,
        "aliveDeadF1": full["alive_dead_f1"],
        "deadSpellF1": spells["mean_f1"],
        "n_dead_pred": spells["n_pred"],
        "n_dead_gt": spells["n_gt"],
        "spellF1": full["spell_f1_mean"],
        "n_poss_pred": full["n_pred_spells"],
        "n_poss_gt": full["n_gt_spells"],
    }


# -- training ----------------------------------------------------------------------------------

def tables(games):
    """{game name: (feature matrix, labels)} and the shared feature names."""
    out, names = {}, None
    for match in games:
        names, matrix = as_matrix(frame_features(match), names)
        out[match.name] = (matrix, match.labels)
    return out, names


def training_rows(table, names_in):
    xs, ys = [], []
    for name in names_in:
        matrix, labels = table[name]
        keep = np.flatnonzero(labels != X)[::TRAIN_STRIDE]
        xs.append(matrix[keep])
        ys.append(labels[keep] == D)
    return np.concatenate(xs), np.concatenate(ys)


def fit(table, names_in):
    x, y = training_rows(table, names_in)
    return make_model().fit(x, y)


def out_of_fold(games, table, tag=None):
    """P(dead) for every game from a model that never saw that game, at any noise level."""
    groups = sorted({clean_name(m) for m in games})
    p_dead = {}
    for held_out in groups:
        train = [m.name for m in games if clean_name(m) != held_out]
        model = fit(table, train)
        for match in games:
            if clean_name(match) == held_out:
                p_dead[match.name] = model.predict_proba(table[match.name][0])[:, 1]
                path = oof_path(match.name, tag)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                np.save(path, p_dead[match.name].astype(np.float32))
        print("  fold {:<14} trained on {} games".format(held_out, len(train)), flush=True)
    return p_dead


def report(games, p_dead, penalties=PENALTIES, label=""):
    rows = [score_game(m, p_dead[m.name], pen) for m in games if m.name in p_dead
            for pen in penalties]
    table = pd.DataFrame(rows)
    columns = ["aliveDeadF1", "deadSpellF1", "n_dead_pred", "n_dead_gt", "spellF1",
               "n_poss_pred", "n_poss_gt"]
    summary = table.groupby(["source", "penalty"])[columns].mean()
    print("== dead-ball detector {}".format(label))
    print(summary.to_string(float_format="%.4f"))
    return table


def baseline_reference(games):
    """The same table for the starter baseline's own D (no ball within +/-2 s)."""
    import baseline
    rows = []
    for match in games:
        letters = baseline.smooth(nearest_team(match), 50)
        dead = letters == "D"
        labels = np.where(match.period > 0, pm.label_codes(letters), X).astype(np.int8)
        full = pm.evaluate(match.labels, labels)
        spells = dead_spell_scores(match.labels, dead)
        rows.append({"source": source_of(match.name), "aliveDeadF1": full["alive_dead_f1"],
                     "deadSpellF1": spells["mean_f1"], "n_dead_pred": spells["n_pred"],
                     "n_dead_gt": spells["n_gt"], "spellF1": full["spell_f1_mean"]})
    print("== starter baseline's D, for reference")
    print(pd.DataFrame(rows).groupby("source").mean().to_string(float_format="%.4f"))


def transfer(games, table):
    """Train on one source only and test on the other: how far a model carries across sources."""
    for train_source, test_source in (("idsse", "metrica"), ("metrica", "idsse")):
        model = fit(table, [m.name for m in games if source_of(m.name) == train_source])
        test = [m for m in games if source_of(m.name) == test_source]
        p_dead = {m.name: model.predict_proba(table[m.name][0])[:, 1] for m in test}
        report(test, p_dead, label="trained on {} only".format(train_source))


def apply_to_bar(games, table, penalty):
    from .sources import bar_ath
    model = fit(table, [m.name for m in games])
    bar = bar_ath()
    names, matrix = as_matrix(frame_features(bar))
    p_dead = model.predict_proba(matrix)[:, 1]
    np.save(os.path.join(CACHE_DIR, "pdead_BAR-ATH.npy"), p_dead.astype(np.float32))
    dead = decode(p_dead, bar.period, penalty)
    graded = bar.period > 0
    starts = np.flatnonzero(np.diff(np.concatenate(([0], (dead & graded).astype(np.int8)))) == 1)
    print("== BAR-ATH, aggregates only (penalty {})".format(penalty))
    print("  D share of graded frames  {:.1f}%".format(100 * dead[graded].mean()))
    print("  stoppages                 {}".format(len(starts)))
    print("  mean stoppage length      {:.1f} s".format(
        dead[graded].sum() / max(len(starts), 1) / bar.fps))
    return p_dead


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--transfer", action="store_true")
    parser.add_argument("--bar", type=float, default=None, metavar="PENALTY",
                        help="train on all external games and apply to BAR-ATH")
    args = parser.parse_args()

    games = bench_games()
    table, names = tables(games)
    print("{} games, {} features".format(len(games), len(names)), flush=True)
    if args.transfer:
        transfer(games, table)
    elif args.bar is not None:
        apply_to_bar(games, table, args.bar)
    else:
        baseline_reference(games)
        report(games, out_of_fold(games, table), label="(leave one game out)")
