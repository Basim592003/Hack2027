"""Old pipeline vs new pipeline, both scored on the new (v4) bench, leave one game out.

Old pipeline = trained on the v3-roughened games with the 88 old dead-ball features, team model
on v3 games with their cached v3 out-of-fold P(dead) -- exactly as the v3 cross-validation did.
New pipeline = the v4 out-of-fold caches written by src.deadball / src.possession.
Check: the old pipeline on its own (v3) held-out games must reproduce 0.681 / 0.721.
"""
import os
import numpy as np
import pandas as pd
from src import deadball, decode, possession
from src.bench import bench_games, source_of
from src.features import as_matrix, frame_features
from src.match import CACHE_DIR, Match

NEW = ("out_not_restarted", "log_since_off_pitch", "nearest_speed")

def old_features(match):
    return {k: v for k, v in frame_features(match).items() if not k.startswith(NEW)}

def joint(match, p_dead, p_home, tag):
    labels = decode.decode(p_dead, p_home, match.period, decode.TEAM_COST, decode.DEAD_COST, False)
    return decode.score(match, labels, pipeline=tag)

new_games = bench_games()                                   # v4
old_games = [Match.load(m.name.replace("~v4", "~v3")) for m in new_games]
for g, m in zip(old_games, new_games):
    g.name = m.name.replace("~v4", "~v3")

# old pipeline's training tables, on v3 games
deadball.frame_features = old_features
dead_tables, names = deadball.tables(old_games)
team_tables, _ = possession.build_tables(old_games)        # uses cached v3 OOF P(dead)
print("tables built", flush=True)

rows = []
groups = sorted({deadball.clean_name(m) for m in new_games})
for held in groups:
    train = [g.name for g in old_games if deadball.clean_name(g) != held]
    dead_model = deadball.fit(dead_tables, train)
    team_model = possession.fit(team_tables, train)
    for old, new in zip(old_games, new_games):
        if deadball.clean_name(new) != held:
            continue
        # old pipeline on the new game
        _, x = as_matrix(old_features(new), names)
        p_dead = dead_model.predict_proba(x)[:, 1]
        dead = deadball.decode(p_dead, new.period, possession.DEAD_PENALTY)
        _, views = possession.both_views(new, dead)
        rows.append(joint(new, p_dead, possession.p_home(team_model, views), "old pipeline, new bench"))
        # check: old pipeline on its own old game (OOF P(dead) as the v3 CV used)
        p_dead_old = np.load(os.path.join(CACHE_DIR, "pdead_oof_{}.npy".format(old.name)))
        rows.append(joint(old, p_dead_old, possession.p_home(team_model, team_tables[old.name][0]),
                          "old pipeline, old bench (check)"))
        # new pipeline on the new game, from its cached out-of-fold predictions
        rows.append(joint(new, *decode.oof(new), "new pipeline, new bench"))
    print("  fold {} done".format(held), flush=True)

t = pd.DataFrame(rows)
print(t.groupby(["pipeline", "source"])[["spellF1", "macroF1", "n_pred", "n_gt"]].mean()
      .unstack("source").to_string(float_format="%.4f"))
# per game, paired: how often does new beat old on the new bench
p = t[t.pipeline != "old pipeline, old bench (check)"].pivot_table(index="game", columns="pipeline", values="spellF1")
p["diff"] = p["new pipeline, new bench"] - p["old pipeline, new bench"]
p["source"] = [source_of(g) for g in p.index]
print(p.to_string(float_format="%.4f"))
print(p.groupby("source")["diff"].agg(["mean", "min", "max", lambda d: (d > 0).sum()]).to_string(float_format="%.4f"))
