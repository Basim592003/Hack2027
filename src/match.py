"""One match in the shape every stage of the pipeline reads: positions in metres, one row per frame.

BAR-ATH and the external games are converted into this same container, so a feature, detector or
decoder written once runs unchanged on the test game and on the games we are allowed to tune on.

    px, py   (T, P) float32  player positions in metres, origin at a pitch corner, NaN if unseen
    team     (P,)   int8     0 home, 1 away -- per slot, which is a tracklet and not a person
    bx, by   (T,)   float32  ball position in metres, NaN if unseen
    period   (T,)   int8     1 or 2 on frames inside a half, 0 elsewhere
    labels   (T,)   int8     possession_metrics codes H=0 A=1 D=2 X=3, or None when unknown

Saved as an uncompressed .npz under data/cache/, which is gitignored with the rest of data/.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT, "data", "cache")

# Same codes as evaluation/possession_metrics.py, so labels go straight into its evaluate().
H, A, D, X = 0, 1, 2, 3
LABELS = np.array(["H", "A", "D", "X"])


@dataclass
class Match:
    name: str
    fps: int
    pitch: Tuple[float, float]
    px: np.ndarray
    py: np.ndarray
    team: np.ndarray
    bx: np.ndarray
    by: np.ndarray
    period: np.ndarray
    labels: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)

    @property
    def n_frames(self):
        return len(self.bx)

    def copy(self):
        return Match(self.name, self.fps, self.pitch, self.px.copy(), self.py.copy(),
                     self.team.copy(), self.bx.copy(), self.by.copy(), self.period.copy(),
                     None if self.labels is None else self.labels.copy(), dict(self.meta))

    def save(self, path=None):
        path = path or cache_path(self.name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        arrays = dict(px=self.px, py=self.py, team=self.team, bx=self.bx, by=self.by,
                      period=self.period, fps=np.int32(self.fps),
                      pitch=np.array(self.pitch, dtype=np.float64))
        if self.labels is not None:
            arrays["labels"] = self.labels
        np.savez(path, **arrays)
        return path

    @classmethod
    def load(cls, name, path=None):
        with np.load(path or cache_path(name)) as z:
            return cls(name, int(z["fps"]), tuple(float(v) for v in z["pitch"]),
                       z["px"], z["py"], z["team"], z["bx"], z["by"], z["period"],
                       z["labels"] if "labels" in z.files else None)


def cache_path(name):
    return os.path.join(CACHE_DIR, "{}.npz".format(name))


def cached(name, build, refresh=False):
    """Load a converted match from the cache, or build it once and store it."""
    path = cache_path(name)
    if not refresh and os.path.isfile(path):
        return Match.load(name, path)
    match = build()
    match.save(path)
    return match
