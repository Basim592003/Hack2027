"""Score a predictor on the labelled external games, degraded to BAR-ATH-like tracking.

This is where every choice between two versions of the pipeline is made -- never on BAR-ATH,
whose public five minutes are for checking alignment only (RULES.md). Scoring is the official
evaluation/possession_metrics.evaluate, unchanged.

    python -m src.bench                      # the baseline method, as the reference row
    python -m src.bench --check-noise        # degraded games' coverage next to BAR-ATH's

A predictor is any function Match -> int8 label codes (H=0 A=1 D=2 X=3), one per frame.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

from . import controlled
from .external import labelled_games
from .idsse import idsse_games
from .match import X, cached
from .noise import degrade, noisy_name

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "starter"))
import possession_metrics as pm  # noqa: E402

ALIVE_MISSING = (0.10, 0.20)  # live-ball dropout levels bracketing BAR-ATH; see src/noise.py
SEEDS = (0,)


def bench_games(levels=ALIVE_MISSING, seeds=SEEDS, convention="last_touch"):
    """Every labelled game, degraded at each live-ball dropout level. Metrica first, then IDSSE.

    convention is the possession definition the labels follow (src/controlled.py). The noise does
    not depend on it -- only dead-ball labels shape the noise, and the relabelling never changes
    them -- so the degraded games are cached once and relabelled on load.
    """
    games = []
    for clean in labelled_games() + idsse_games():
        for level in levels:
            for seed in seeds:
                name = noisy_name(clean.name, level, seed)
                games.append(cached(name, lambda: degrade(clean, level, seed)))
                games[-1].name = name
                controlled.apply(games[-1], convention)
    return games


def source_of(name):
    return name.split("~")[0].rstrip("0123456789").split("_")[0]


def score(match, labels):
    result = pm.evaluate(match.labels, labels)
    return {
        "source": source_of(match.name),
        "live": int(match.name.split("live")[1][:2]) if "live" in match.name else None,
        "game": match.name.split("~")[0],
        "spellF1": result["spell_f1_mean"],
        "macroF1": result["macro_f1"],
        "aliveDeadF1": result["alive_dead_f1"],
        "teamAcc": result["team_acc_gt_alive"],
        "n_pred": result["n_pred_spells"],
        "n_gt": result["n_gt_spells"],
    }


def run(predict, games=None, label="predictor", verbose=True):
    games = games if games is not None else bench_games()
    rows = []
    for match in games:
        labels = np.asarray(predict(match), dtype=np.int8)
        labels = np.where(match.period > 0, labels, X).astype(np.int8)
        rows.append(score(match, labels))
    table = pd.DataFrame(rows)
    if verbose:
        print("== {}".format(label))
        print(table.to_string(index=False, float_format="%.4f"))
        print(summary(table).to_string(float_format="%.4f"))
    return table


def summary(table):
    """Means per source and live-ball dropout level. Sources are never pooled: their label
    conventions differ (DFL marks more of the match dead and splits possession more finely)."""
    columns = ["spellF1", "macroF1", "aliveDeadF1", "teamAcc", "n_pred", "n_gt"]
    return table.groupby(["source", "live"])[columns].mean()


# -- the reference: starter/baseline.py's method, applied to a Match ------------------------------

def nearest_team(match):
    """Team of the tracked player nearest the ball per frame; -1 where the ball is unseen."""
    out = np.full(match.n_frames, -1, dtype=np.int8)
    seen = ~np.isnan(match.bx)
    dx = match.px[seen] - match.bx[seen, None]
    dy = match.py[seen] - match.by[seen, None]
    d2 = np.where(np.isnan(dx), np.inf, dx * dx + dy * dy)
    has_player = np.isfinite(d2).any(axis=1)
    rows = np.flatnonzero(seen)[has_player]
    out[rows] = match.team[np.argmin(d2[has_player], axis=1)]
    return out


def baseline_predictor(window=50):
    import baseline

    def predict(match):
        return pm.label_codes(baseline.smooth(nearest_team(match), window))
    return predict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check-noise", action="store_true")
    args = parser.parse_args()

    if args.check_noise:
        from .coverage import coverage
        from .sources import bar_ath
        rows = [coverage(m) for m in [bar_ath()] + bench_games()]
        print(pd.DataFrame(rows).to_string(index=False))
    else:
        run(baseline_predictor(50), label="baseline nearest player, +/-50 majority vote")
