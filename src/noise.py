"""Degrade a clean labelled game until its tracking looks like the BAR-ATH ATD.

Calibrated against aggregate input statistics only (python -m src.coverage), never labels:

                         BAR-ATH ATD    Metrica sample, clean
  players seen, mean          20.5            22.0
  players seen, 5th pct         14              22
  tracklet starts / min       11.3             0.1
  track slots                   46           26-35, one per person

The ball, while play is dead. The Metrica sample tracking blanks the ball for exactly the whole of
every stoppage (gap edges within a frame of the event-derived D run) and IDSSE never loses it, so
neither looks like BAR-ATH: around the Smart Tagging's automatic set-piece times, its ball is
missing on ~75% of frames in the seconds before the restart and stays missing for several seconds
after, with no sharp edge. So stoppages are re-rendered: the ball is gone from about DEAD_MISSING
of them, with edges blurred by RESTART_EDGE_SD, and in the rest it is visible.

What a visible dead ball does. Neither source has one to copy: IDSSE freezes the ball exactly where
play stopped for every dead frame (zero movement, then a jump of median 6 m to the restart spot),
and Metrica has no ball at all. v3 showed IDSSE's frozen ball and rested Metrica's on the restart
spot, so every visible stoppage on the bench was a still ball, and the dead-ball model learned
"seen and moving = live". On BAR-ATH the ball is fetched, carried and thrown back while play is
stopped -- in the 5 s before a tagged set piece it is still on only 4% of the frames it is seen --
and the model called D on 0-1% of them (92-95% with the ball missing). v4 renders every stoppage
from what happens in one (dead_ball_path): the ball runs on and slows, is passed or brought to the
restart taker (the player nearest the restart spot when play resumes, whose real movement is in
the data), is carried by them, and is set down on the spot. Durations and speeds are round numbers
of football, drawn per stoppage over wide ranges so no single rendering can be learnt as a tell.

The ball, while play is live. The sample games lose it on only 9-14% of live frames, while
DATA.md says BAR-ATH loses it disproportionately in crowded boxes, duels and the air. How often
cannot be measured without labels -- the mixture bound is roughly 10-25% of live frames -- so
live-ball dropout is a parameter, and every method is scored at several values of it.

Position error. BAR-ATH's tracking is smooth frame to frame -- the median second difference of a
player's position is 0.001 m, as on the clean external games -- so its error wanders slowly rather
than jumping each frame. v2 added independent 0.15 m noise to every frame, which made the bench's
second differences ~0.4 m: velocity-based signals (a still ball, a player moving with the ball)
were mostly noise on the bench and clean on BAR-ATH, a shift between training and use. v3 keeps
the 0.15 m size but lets the error drift, low-passed over DRIFT_FRAMES.

Deterministic for a given seed, so a benchmark run reproduces exactly.
"""

import numpy as np

from .match import A, D, H, X

VERSION = 4              # part of every cached degraded game's name; bump on any change here
TEAM_SLOTS = 23         # 46 slots for two teams, as in the BAR-ATH metadata
PLAYER_STARTS_PER_MIN = 11.3 / 22
PLAYER_MISSING_SHARE = 1 - 20.5 / 22
CROWD_RADIUS = 2.0       # metres; a player this close to another is likelier to be occluded
CROWD_BOOST = 3.0
POSITION_NOISE = 0.15    # metres, players and ball
DRIFT_FRAMES = 12        # ~0.5 s: how slowly that error wanders (Gaussian low-pass sd)
BALL_GAP_MEDIAN = 40     # frames; a lost-in-play ball comes back within a couple of seconds
BALL_CROWD_RADIUS = 5.0
DEAD_MISSING = 0.75      # share of stoppages with the ball gone; BAR-ATH before tagged restarts
RESTART_EDGE_SD = 25     # frames; how far a dead-ball gap's ends stray from the stoppage's
# A visible stoppage (dead_ball_path). Ranges are drawn uniformly per stoppage.
ROLL_S = (1.0, 3.0)      # seconds the ball runs on before anyone fetches it
ROLL_DECAY_S = 0.7       # e-folding time of a rolling, bouncing ball's speed
FETCH_SPEED = (4.0, 10.0)  # m/s: passed, thrown or kicked back to the taker
NEW_BALL = 0.2           # share of stoppages where a fresh ball turns up with the taker instead
CARRY_S = (2.0, 10.0)    # seconds the taker carries it to the spot
CARRY_OFFSET = (0.3, 1.0)  # metres from the carrier's body centre: in hands or at the feet
SET_S = (0.5, 4.0)       # seconds it sits on the spot before the restart
SET_BLEND = 12           # frames to move from the carrier onto the spot
EXIT_LAG = 6             # frames over which the ball's exit velocity is measured
EDGE_SEARCH = 25         # frames to look for a tracked ball either side of a stoppage
MAX_EXIT_SPEED = 35.0    # m/s
OFF_PITCH_LIMIT = 8.0    # metres beyond a line a rolling ball is followed (boards, stands)


def _runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def _crowding(px, py, radius):
    """(T, P) True where another tracked player is within radius. Chunked to bound memory."""
    out = np.zeros(px.shape, dtype=bool)
    for start in range(0, len(px), 5000):
        x, y = px[start:start + 5000], py[start:start + 5000]
        d2 = (x[:, :, None] - x[:, None, :]) ** 2 + (y[:, :, None] - y[:, None, :]) ** 2
        idx = np.arange(px.shape[1])
        d2[:, idx, idx] = np.inf
        out[start:start + 5000] = np.nanmin(np.where(np.isnan(d2), np.inf, d2), axis=2) < radius ** 2
    return out


def _drift(rng, n):
    """(n, 2) position error of sd POSITION_NOISE that wanders smoothly: white noise low-passed
    by a Gaussian of sd DRIFT_FRAMES, the kernel scaled so the output keeps unit variance."""
    half = 3 * DRIFT_FRAMES
    offsets = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (offsets / DRIFT_FRAMES) ** 2)
    kernel /= np.sqrt(np.sum(kernel ** 2))
    white = rng.normal(0, 1, size=(n + 2 * half, 2))
    smooth = np.stack([np.convolve(white[:, k], kernel, mode="valid") for k in (0, 1)], axis=1)
    return (POSITION_NOISE * smooth).astype(np.float32)


def drop_players(match, rng):
    """Missing runs per player, likelier when crowded; each reappearance takes a fresh slot."""
    px, py = match.px, match.py
    n_frames, n_players = px.shape
    crowd = _crowding(px, py, CROWD_RADIUS)
    # Per-frame hazard of a run starting, so the expected starts per minute match BAR-ATH.
    base = PLAYER_STARTS_PER_MIN / (match.fps * 60)
    hazard = base * np.where(crowd, CROWD_BOOST, 1.0)
    hazard *= base / max(hazard[~np.isnan(px)].mean(), 1e-12)
    mean_len = PLAYER_MISSING_SHARE * match.fps * 60 / PLAYER_STARTS_PER_MIN

    visible = ~np.isnan(px)
    starts = (rng.random(px.shape) < hazard) & visible
    for player in range(n_players):
        for frame in np.flatnonzero(starts[:, player]):
            length = int(rng.exponential(mean_len)) + 1
            visible[frame:frame + length, player] = False

    # Re-slot: every visible stretch of a player is a tracklet, placed in a free slot of its team.
    slots_x = np.full((n_frames, 2 * TEAM_SLOTS), np.nan, dtype=np.float32)
    slots_y = np.full_like(slots_x, np.nan)
    tracklets = []
    for player in range(n_players):
        for start, stop in zip(*_runs(visible[:, player])):
            tracklets.append((start, stop, player))
    tracklets.sort()
    free_at = np.zeros(2 * TEAM_SLOTS, dtype=np.int64)
    for start, stop, player in tracklets:
        team = match.team[player]
        pool = np.arange(team * TEAM_SLOTS, (team + 1) * TEAM_SLOTS)
        candidates = pool[free_at[pool] <= start]
        if not len(candidates):
            continue  # more simultaneous tracklets than slots; the tracker would drop one too
        slot = rng.choice(candidates)
        free_at[slot] = stop
        noise = _drift(rng, stop - start)
        slots_x[start:stop, slot] = px[start:stop, player] + noise[:, 0]
        slots_y[start:stop, slot] = py[start:stop, player] + noise[:, 1]
    team = np.repeat(np.array([0, 1], dtype=np.int8), TEAM_SLOTS)
    return slots_x, slots_y, team


def _seen_near(match, frame, step):
    """First live frame from frame, moving by step, with the ball tracked; -1 if none nearby."""
    for t in range(frame, frame + step * EDGE_SEARCH, step):
        if not 0 <= t < match.n_frames or match.labels[t] not in (H, A):
            return -1
        if not np.isnan(match.bx[t]):
            return t
    return -1


def _filled(values):
    """values with NaNs replaced by the nearest tracked value; None if nothing is tracked."""
    known = np.flatnonzero(~np.isnan(values))
    if not len(known):
        return None
    index = np.arange(len(values))
    pos = np.searchsorted(known, index)
    after = known[np.clip(pos, 0, len(known) - 1)]
    before = known[np.clip(pos - 1, 0, len(known) - 1)]
    return values[np.where(np.abs(before - index) < np.abs(after - index), before, after)]


def dead_ball_path(match, start, stop, rng):
    """(stop - start, 2) ball positions through one stoppage: it runs on from where play stopped
    and slows, is passed or brought to the restart taker, carried, and set on the restart spot.

    The taker is the player nearest the ball when play resumes; their real movement during the
    stoppage carries the ball. Without a restart (the end of a half) the ball just runs on and
    lies there; without an exit (the start of a half) it starts on the restart spot.
    """
    fps = match.fps
    span = stop - start
    length, width = match.pitch
    roll = int(rng.uniform(*ROLL_S) * fps)
    fetch_speed = rng.uniform(*FETCH_SPEED)
    new_ball = rng.random() < NEW_BALL
    carry = int(rng.uniform(*CARRY_S) * fps)
    angle = rng.uniform(0, 2 * np.pi)
    offset = rng.uniform(*CARRY_OFFSET) * np.array([np.cos(angle), np.sin(angle)])
    settle = int(rng.uniform(*SET_S) * fps)

    exit_frame, restart_frame = _seen_near(match, start - 1, -1), _seen_near(match, stop, 1)
    spot = None if restart_frame < 0 else \
        np.array([match.bx[restart_frame], match.by[restart_frame]], dtype=np.float64)
    if exit_frame >= 0:
        here = np.array([match.bx[exit_frame], match.by[exit_frame]], dtype=np.float64)
        back = exit_frame - EXIT_LAG
        velocity = np.zeros(2)
        if back >= 0 and not np.isnan(match.bx[back]):
            velocity = (here - [match.bx[back], match.by[back]]) * fps / EXIT_LAG
            speed = np.hypot(*velocity)
            if speed > MAX_EXIT_SPEED:
                velocity *= MAX_EXIT_SPEED / speed
    elif spot is not None:
        here, velocity = spot, np.zeros(2)
    else:
        return np.full((span, 2), np.nan)

    t = np.arange(span)[:, None] / fps
    path = here + velocity * ROLL_DECAY_S * (1 - np.exp(-t / ROLL_DECAY_S))
    path[:, 0] = np.clip(path[:, 0], -OFF_PITCH_LIMIT, length + OFF_PITCH_LIMIT)
    path[:, 1] = np.clip(path[:, 1], -OFF_PITCH_LIMIT, width + OFF_PITCH_LIMIT)
    if spot is None:
        return path

    # Phases, counted back from the restart: set on the spot, carried before that, fetched before
    # that. A short stoppage cuts each down so the fetch still has at least half the time between
    # the roll and the set, and the ball does not jump.
    roll = min(roll, span // 3)
    settle = min(settle, (span - roll) // 2)
    carry = min(carry, (span - roll - settle) // 2)
    set_at = span - settle
    pick_at = set_at - carry
    taker_x = taker_y = None
    if carry > 0:
        d = np.hypot(match.px[restart_frame] - spot[0], match.py[restart_frame] - spot[1])
        if np.isfinite(d).any():
            k = int(np.nanargmin(np.where(np.isnan(d), np.inf, d)))
            taker_x = _filled(match.px[start:stop, k].astype(np.float64))
            taker_y = _filled(match.py[start:stop, k].astype(np.float64))
    if taker_x is None:
        pick_at, carry = set_at, 0
    if carry > 0:
        carried = np.stack([taker_x, taker_y], axis=1) + offset
        path[pick_at:set_at] = carried[pick_at:set_at]
        target = carried[pick_at]
    else:
        target = spot
    if not new_ball:
        # Fetched at fetch_speed, or faster when there is less time than that takes.
        travel = int(np.ceil(np.hypot(*(target - path[roll])) / fetch_speed * fps))
        go = max(roll, pick_at - travel)
        if go < pick_at:
            origin = path[max(go - 1, 0)].copy()
            share = (np.arange(go, pick_at) - go + 1)[:, None] / (pick_at - go)
            path[go:pick_at] = origin + share * (target - origin)
    if settle > 0:
        before = path[set_at - 1] if set_at > 0 else spot
        blend = min(SET_BLEND, settle)
        share = (np.arange(blend) + 1)[:, None] / blend
        path[set_at:set_at + blend] = before + share * (spot - before)
        path[set_at + blend:] = spot
    return path


def render_dead_ball(match, rng, dead_missing=DEAD_MISSING):
    """(bx, by) with every stoppage re-rendered by dead_ball_path, then the ball gone from about
    dead_missing of them, with the gap's edges blurred into the play either side."""
    bx, by = match.bx.copy(), match.by.copy()
    n = len(bx)
    for start, stop in zip(*_runs(match.labels == D)):
        path = dead_ball_path(match, start, stop, rng)
        bx[start:stop] = path[:, 0]
        by[start:stop] = path[:, 1]
        if rng.random() < dead_missing:
            first = int(np.clip(start + rng.normal(0, RESTART_EDGE_SD), 0, n))
            last = int(np.clip(stop + rng.normal(0, RESTART_EDGE_SD), 0, n))
            bx[first:last] = np.nan
            by[first:last] = np.nan
    return bx, by


def drop_ball_in_play(match, rng, alive_missing, bx=None, by=None):
    """Add live-ball gaps, likelier near crowds, until alive_missing of live frames lack a ball."""
    bx = match.bx.copy() if bx is None else bx.copy()
    by = match.by.copy() if by is None else by.copy()
    alive = (match.labels != D) & (match.labels != X)
    missing = np.isnan(bx)
    target = int(alive_missing * alive.sum())
    if missing[alive].sum() >= target:
        return bx, by

    near = np.zeros(len(bx))
    seen = ~missing
    dx = match.px[seen] - bx[seen, None]  # crowding uses the clean players, before they are dropped
    dy = match.py[seen] - by[seen, None]
    near[seen] = np.nansum(dx * dx + dy * dy < BALL_CROWD_RADIUS ** 2, axis=1)
    weight = np.where(alive & seen, 1.0 + near, 0.0)
    weight /= weight.sum()

    while missing[alive].sum() < target:
        frame = rng.choice(len(bx), p=weight)
        length = int(rng.lognormal(np.log(BALL_GAP_MEDIAN), 0.8)) + 1
        missing[frame:frame + length] = True
    bx[missing] = np.nan
    by[missing] = np.nan
    return bx, by


def degrade(match, alive_missing=0.15, seed=0):
    """A copy of a labelled match with BAR-ATH-like tracking. Labels and periods are untouched."""
    rng = np.random.default_rng(seed)
    out = match.copy()
    bx, by = render_dead_ball(match, rng)
    out.bx, out.by = drop_ball_in_play(match, rng, alive_missing, bx, by)
    noise = _drift(rng, len(out.bx))
    out.bx += noise[:, 0]
    out.by += noise[:, 1]
    out.px, out.py, out.team = drop_players(match, rng)
    out.name = noisy_name(match.name, alive_missing, seed)
    return out


def noisy_name(clean_name, alive_missing, seed):
    return "{}~v{}live{:02d}s{}".format(clean_name, VERSION, int(round(100 * alive_missing)), seed)
