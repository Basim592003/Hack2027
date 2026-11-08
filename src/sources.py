"""BAR-ATH, the test game, as a Match. It carries no labels: nothing here reads a ground truth."""

import os
import sys

import numpy as np

from . import epts
from .match import cached

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, asset_paths, half_video_blocks  # noqa: E402


def bar_ath(refresh=False):
    def build():
        paths = asset_paths(GAME)
        match = epts.load_match(GAME, paths["atd_meta"], paths["atd_raw"], GAMES[GAME]["pitch"],
                                GAMES[GAME]["fps"])
        frame_ids = match.meta["frame_ids"]
        if not np.array_equal(frame_ids, np.arange(1, GAMES[GAME]["n_video_frames"] + 1)):
            raise ValueError("ATD is not one record per video frame numbered from 1")
        # Row i is video frame i (frame_id - 1), so the published half blocks index it directly.
        for half, first, last in half_video_blocks(GAME):
            match.period[first:last + 1] = half
        return match

    return cached(GAME, build, refresh)
