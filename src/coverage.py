"""Aggregate tracking-quality statistics: how often and for how long the ball and players go missing.

Input data only, no labels. These numbers are what src/noise.py calibrates against, so that the
external games are degraded to look like the BAR-ATH tracking rather than like clean data.

    python -m src.coverage
"""

import numpy as np
import pandas as pd

from . import touches
from .external import labelled_games
from .sources import bar_ath


def run_lengths(mask):
    """Lengths of the True runs in a boolean array."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


def kink_rates(match, lags=(4, 12)):
    """Ball direction or speed changes of at least touches.KINK_SPEED per graded minute, measured
    over each lag, and the ratio of the shortest to the longest. A tracker's ball filter removes the
    short-lag kinks and keeps the long-lag ones, so the ratio reads smoothing whatever share of
    frames the ball is seen on."""
    graded = match.period > 0
    minutes = graded.sum() / match.fps / 60
    rates = []
    for lag in lags:
        strength = touches.kink_strength(match.bx.astype(np.float64), match.by.astype(np.float64),
                                         match.fps, lag)
        filled = np.where(np.isnan(strength), -np.inf, strength)
        window = np.lib.stride_tricks.sliding_window_view(
            np.pad(filled, lag, constant_values=-np.inf), 2 * lag + 1)
        peaks = (filled >= touches.KINK_SPEED) & (filled >= window.max(axis=1)) & graded
        rates.append(peaks.sum() / minutes)
    out = {"kinks/min ±{}".format(lag): round(rate, 1) for lag, rate in zip(lags, rates)}
    out["kink_ratio"] = round(rates[0] / max(rates[-1], 1e-9), 2)
    return out


def coverage(match):
    graded = match.period > 0
    missing = np.isnan(match.bx) & graded
    gaps = run_lengths(missing)
    seen = np.sum(~np.isnan(match.px), axis=1)[graded]
    # Slot churn: how often a slot starts a new tracklet, per minute of play.
    starts = np.sum(np.diff((~np.isnan(match.px[graded])).astype(np.int8), axis=0) == 1)
    return {
        "game": match.name,
        **kink_rates(match),
        "ball_missing%": round(100 * missing.sum() / graded.sum(), 1),
        "gaps/min": round(len(gaps) / (graded.sum() / match.fps / 60), 1),
        "gap_p50_f": int(np.percentile(gaps, 50)),
        "gap_p90_f": int(np.percentile(gaps, 90)),
        "gap_p99_f": int(np.percentile(gaps, 99)),
        "frames_in_gaps>2s%": round(100 * gaps[gaps > 50].sum() / max(gaps.sum(), 1), 1),
        "players_mean": round(float(seen.mean()), 1),
        "players_p05": int(np.percentile(seen, 5)),
        "tracklet_starts/min": round(starts / (graded.sum() / match.fps / 60), 1),
    }


if __name__ == "__main__":
    rows = [coverage(m) for m in [bar_ath()] + labelled_games()]
    print(pd.DataFrame(rows).to_string(index=False))
