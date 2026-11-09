"""Regenerate the submission from the provided data and the external games. One command.

    python -m src.fetch_metrica                 # once: external data, pinned
    python -m src.fetch_idsse                   # once: external data, MD5-checked
    python -m src.run --out evaluation/results/<team>.csv

What it does, in order -- every model is trained on the external games only, and every cost was
chosen on them (RULES.md, clarification of 2026-11-03):

  1. the 10 labelled external games, each degraded at two live-ball dropout levels (src/noise.py,
     fixed seeds), so 20 training games, plus the same again at another noise seed whose training
     rows alone join the team model (src/possession.py, EXTRA_SEEDS)
  2. dead-ball model (src/deadball.py): out-of-fold P(dead) for every training game -- the team
     model's restart features need predictions their games were not trained on -- then one model
     on all 20 applied to BAR-ATH
  3. team model (src/possession.py) on all 20, applied to BAR-ATH using step 2's stoppages
  4. joint decoder (src/decode.py) over H / A / D, X outside the two graded halves
  5. write the CSV and run evaluation/validate_submission.py on it

Deterministic: fixed seeds, fixed model random_state, pinned inputs. Running it twice, with or
without data/cache/, gives the same file byte for byte. P(dead) and P(home) for BAR-ATH are kept
in data/cache/ as intermediate outputs.
"""

import argparse
import os
import subprocess
import sys

import numpy as np

from . import deadball, decode, possession
from .bench import bench_games
from .features import as_matrix, frame_features
from .match import CACHE_DIR, D, H, LABELS
from .sources import bar_ath

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "starter"))
import baseline  # noqa: E402  -- for write_submission, the format in four lines

DEFAULT_OUT = os.path.join(ROOT, "evaluation", "results", "possession.csv")
# Possession definition the external training labels follow (src/controlled.py).
CONVENTION = "last_touch"


def predict_bar(games):
    """(labels, p_dead, p_home) for BAR-ATH."""
    bar = bar_ath()

    print("step 2  dead ball", flush=True)
    extra = bench_games(seeds=possession.EXTRA_SEEDS, convention=CONVENTION) \
        if possession.EXTRA_SEEDS else []
    table, names = deadball.tables(games)
    extra_table = deadball.tables(extra)[0]
    deadball.out_of_fold(games, table, extra=extra_table)
    dead_in = [m.name for m in games] + (list(extra_table) if deadball.TRAIN_EXTRA else [])
    dead_model = deadball.fit(dict(table, **extra_table), dead_in)
    del table, extra_table
    _, matrix = as_matrix(frame_features(bar), names)
    p_dead = dead_model.predict_proba(matrix)[:, 1]
    del matrix

    print("step 3  team in possession", flush=True)
    rows = possession.extra_rows(extra)
    del extra
    tables, team_names = possession.build_tables(games)
    team_model = possession.fit(tables, [m.name for m in games], team_names, rows, list(rows))
    del tables, rows
    stoppages = deadball.decode(p_dead, bar.period, possession.DEAD_PENALTY)
    _, views = possession.both_views(bar, stoppages)
    p_home = possession.p_home(team_model, views)

    print("step 4  joint decoding", flush=True)
    labels = decode.decode(p_dead, p_home, bar.period, decode.TEAM_COST, decode.DEAD_COST,
                           decode.MEMORY)
    if decode.SNAP:
        labels = decode.snap(labels, decode.snap_context(bar), bar.period)
    return bar, labels, p_dead, p_home


def summary(bar, labels):
    graded = bar.period > 0
    live = graded & (labels != D)
    changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = [t for t in changes if labels[t] == D and graded[t]]
    print("\nBAR-ATH, aggregates only")
    print("  dead ball            {:.1f}% of graded frames, {} stoppages".format(
        100 * np.mean(labels[graded] == D), len(starts)))
    print("  home share of live   {:.1f}%".format(100 * np.mean(labels[live] == H)))


def main(out):
    games = bench_games(convention=CONVENTION)
    bar, labels, p_dead, p_home = predict_bar(games)
    np.save(os.path.join(CACHE_DIR, "final_pdead_BAR-ATH.npy"), p_dead.astype(np.float32))
    np.save(os.path.join(CACHE_DIR, "final_phome_BAR-ATH.npy"), p_home.astype(np.float32))

    path = baseline.write_submission(out, LABELS[labels])
    summary(bar, labels)
    print("\nwrote {}".format(os.path.relpath(path, ROOT)), flush=True)
    check = subprocess.run([sys.executable, os.path.join(ROOT, "evaluation",
                                                         "validate_submission.py"), path])
    if check.returncode != 0:
        raise SystemExit("validation failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=DEFAULT_OUT, help="where to write the CSV")
    main(parser.parse_args().out)
