"""Core code for "Does seeing the agent's plan make human feedback more useful?"

A first test with a tabular Q-learning agent in a 5x5 gridworld (Sutton & Barto,
2018, Ch. 6), a simulated trainer whose good/bad feedback is sometimes wrong and
sometimes missing, and a small human pilot (agent only vs. agent + policy arrows).

This file is the single source of truth for the project's code:
  * the notebook embeds every "# %% [name]" section below as a code cell,
  * pilot/pilot_app.py and pilot/analyze_pilot.py import it,
  * tests/ import it.
"""

# %% [imports]
import csv
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.ticker
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

__version__ = "1.0.0"

# %% [env]
# Actions as (row, col) moves. Row 0 is the TOP row of the grid.
ACTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
ACTION_NAMES = ("up", "right", "down", "left")
N_ACTIONS = len(ACTIONS)

FREE, WALL, GOAL, LAVA = 0, 1, 2, 3

LAYOUTS = {
    # Main 5x5 map.  S = start, G = goal, # = wall, L = lava, . = free.
    # Shortest route (8 moves): along the bottom row, then up the right column.
    # Going up the left column leads to a 12-move detour; lava sits in the middle.
    "A": ["...#G",
          ".#.#.",
          ".#...",
          ".#L#.",
          "S...."],
    # Practice map for the pilot (no walls, no lava).
    "practice": ["....G",
                 ".....",
                 ".....",
                 ".....",
                 "S...."],
    # Sutton & Barto (2018) Example 6.6, Cliff Walking: 4 x 12, L = the cliff.
    "cliff": ["............",
              "............",
              "............",
              "SLLLLLLLLLLG"],
}
# Mirror image of map A, used for the second pilot session so that a participant
# cannot simply reuse what they memorised about the first map.
LAYOUTS["B"] = [row[::-1] for row in LAYOUTS["A"]]


class GridWorld:
    """Deterministic gridworld in the style of Sutton & Barto (2018), Ch. 6.

    * Moving off the grid or into a wall leaves the agent where it is (a "bump").
    * Entering the goal ends the episode with `goal_reward`.
    * Entering lava gives `lava_reward` and either ends the episode (default) or,
      with `lava_resets=True`, sends the agent back to the start (Cliff Walking).
    * Every other move gives `step_reward`.
    * Episodes are cut off (truncated, not terminated) after `max_episode_steps`.
    """

    def __init__(self, layout="A", step_reward=-0.01, goal_reward=1.0, lava_reward=-1.0,
                 lava_resets=False, max_episode_steps=50):
        self.name = layout if isinstance(layout, str) else "custom"
        rows = LAYOUTS[layout] if isinstance(layout, str) else list(layout)
        self.layout = [str(r) for r in rows]
        self.H, self.W = len(self.layout), len(self.layout[0])
        if any(len(r) != self.W for r in self.layout):
            raise ValueError("all rows of a layout must have the same length")
        self.n_states, self.n_actions = self.H * self.W, N_ACTIONS
        codes = {".": FREE, "S": FREE, "#": WALL, "G": GOAL, "L": LAVA}
        self.cell_type = np.full(self.n_states, FREE, dtype=int)
        starts = []
        for r, row in enumerate(self.layout):
            for c, ch in enumerate(row):
                if ch not in codes:
                    raise ValueError(f"unknown cell {ch!r} at ({r}, {c})")
                self.cell_type[r * self.W + c] = codes[ch]
                if ch == "S":
                    starts.append(r * self.W + c)
        if len(starts) != 1:
            raise ValueError("a layout needs exactly one start cell 'S'")
        if not np.any(self.cell_type == GOAL):
            raise ValueError("a layout needs at least one goal cell 'G'")
        self.start = starts[0]
        self.step_reward, self.goal_reward, self.lava_reward = step_reward, goal_reward, lava_reward
        self.lava_resets = lava_resets
        self.max_episode_steps = max_episode_steps
        # T[s, a] = cell reached from cell s with action a (before any lava handling).
        self.T = np.empty((self.n_states, N_ACTIONS), dtype=int)
        for s in range(self.n_states):
            r, c = divmod(s, self.W)
            for a, (dr, dc) in enumerate(ACTIONS):
                nr, nc = r + dr, c + dc
                inside = 0 <= nr < self.H and 0 <= nc < self.W
                blocked = (not inside) or self.cell_type[nr * self.W + nc] == WALL
                self.T[s, a] = s if blocked else nr * self.W + nc
        self._T = self.T.tolist()          # plain lists: faster single lookups
        self._type = self.cell_type.tolist()
        self.reset()

    def rc(self, s):
        """(row, col) of state index s."""
        return divmod(int(s), self.W)

    def reset(self):
        self.s, self.t = self.start, 0
        return self.s

    def step(self, a):
        """Take action a. Returns (next_state, reward, terminated, truncated)."""
        s2 = self._T[self.s][a]
        kind = self._type[s2]
        terminated = False
        if kind == GOAL:
            reward, terminated = self.goal_reward, True
        elif kind == LAVA:
            reward = self.lava_reward
            if self.lava_resets:
                s2 = self.start
            else:
                terminated = True
        else:
            reward = self.step_reward
        self.t += 1
        truncated = (not terminated and self.max_episode_steps is not None
                     and self.t >= self.max_episode_steps)
        self.s = s2
        return s2, reward, terminated, truncated

    def __repr__(self):
        return f"GridWorld({self.name!r}, {self.H}x{self.W})"


# %% [oracle]
class Oracle:
    """Shortest-path 'ground truth', used to score actions and feedback.

    dist[s]        number of moves from s to the goal (walls and lava are impassable)
    optimal[s, a]  True if action a moves one step closer to the goal
    states         non-terminal cells that can reach the goal (used for policy accuracy)
    route          non-terminal cells on a shortest path from the start (the cells the
                   'learned' criterion depends on)
    """

    def __init__(self, env):
        n = env.n_states
        passable = (env.cell_type != WALL) & (env.cell_type != LAVA)
        dist = np.full(n, np.inf)
        frontier = [int(g) for g in np.flatnonzero(env.cell_type == GOAL)]
        for g in frontier:
            dist[g] = 0.0
        while frontier:                       # breadth-first search outward from the goal
            nxt = []
            for s in frontier:
                for a in range(N_ACTIONS):
                    p = int(env.T[s, a])      # grid moves are reversible
                    if p != s and passable[p] and np.isinf(dist[p]):
                        dist[p] = dist[s] + 1
                        nxt.append(p)
            frontier = nxt
        self.dist = dist
        self.optimal = np.zeros((n, N_ACTIONS), dtype=bool)
        states = []
        for s in range(n):
            if env.cell_type[s] == FREE and np.isfinite(dist[s]):
                states.append(s)
                for a in range(N_ACTIONS):
                    p = int(env.T[s, a])
                    if p != s and passable[p] and dist[p] == dist[s] - 1:
                        self.optimal[s, a] = True
        self.states = np.array(states, dtype=int)
        self._opt = self.optimal.tolist()
        self.start_dist = int(dist[env.start]) if np.isfinite(dist[env.start]) else None
        # Cells on some shortest path from the start: follow optimal moves forward.
        route, frontier = set(), [env.start] if self.start_dist is not None else []
        while frontier:
            nxt = []
            for s in frontier:
                if s in route or env.cell_type[s] == GOAL:
                    continue
                route.add(s)
                nxt.extend(int(env.T[s, a]) for a in np.flatnonzero(self.optimal[s]))
            frontier = nxt
        self.route = np.array(sorted(route), dtype=int)


def strict_greedy(Q):
    """Greedy action in every state, or -1 where the best value is tied (no preference yet)."""
    best = Q.max(axis=1, keepdims=True)
    tied = (Q == best).sum(axis=1) > 1
    greedy = Q.argmax(axis=1)
    greedy[tied] = -1
    return greedy


def policy_accuracy(Q, oracle, states=None):
    """Fraction of cells (default: all non-terminal cells) whose greedy action is a
    shortest-path action."""
    states = oracle.states if states is None else states
    g = strict_greedy(Q)[states]
    ok = g >= 0
    ok[ok] = oracle.optimal[states[ok], g[ok]]
    return float(ok.mean())


def route_accuracy(Q, oracle):
    """Policy accuracy on the cells of the shortest route from the start only."""
    return policy_accuracy(Q, oracle, oracle.route)


def greedy_path_optimal(Q, env, oracle):
    """True if following the greedy policy from the start reaches the goal in the fewest
    possible moves (ties count as 'not decided'). This is the 'agent has learned the
    task' criterion used throughout the project."""
    if oracle.start_dist is None:
        return False
    s = env.start
    for _ in range(oracle.start_dist):
        q = Q[s]
        a = int(q.argmax())
        if (q == q[a]).sum() > 1 or not oracle._opt[s][a]:
            return False
        s = env._T[s][a]
    return env._type[s] == GOAL


# %% [agents]
class QLearningAgent:
    """Tabular Q-learning (Sutton & Barto, 2018, Sec. 6.5, Eq. 6.8):

        Q(S,A) <- Q(S,A) + alpha * [ R + gamma * max_a Q(S',a) - Q(S,A) ]

    with epsilon-greedy exploration and random tie-breaking."""

    def __init__(self, n_states, n_actions=N_ACTIONS, alpha=0.5, gamma=0.95, epsilon=0.1,
                 q_init=0.0, seed=None):
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.n_actions = n_actions
        self.Q = np.full((n_states, n_actions), float(q_init))
        self.rng = np.random.default_rng(seed)

    def greedy(self, s):
        q = self.Q[s]
        best = np.flatnonzero(q == q.max())
        return int(best[0]) if best.size == 1 else int(self.rng.choice(best))

    def act(self, s):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.greedy(s)

    def update(self, s, a, r, s2, terminated):
        target = r if terminated else r + self.gamma * self.Q[s2].max()
        td_error = target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error
        return td_error


class SarsaAgent(QLearningAgent):
    """Sarsa, on-policy TD control (Sutton & Barto, 2018, Sec. 6.4, Eq. 6.7):

        Q(S,A) <- Q(S,A) + alpha * [ R + gamma * Q(S',A') - Q(S,A) ]

    Used only for the Cliff Walking sanity check."""

    def update(self, s, a, r, s2, terminated, a2=None):
        target = r if terminated else r + self.gamma * self.Q[s2, a2]
        td_error = target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error
        return td_error


def run_episodes(env, agent, n_episodes, oracle=None, snapshot_at=()):
    """Train `agent` on `env` for `n_episodes`, using only the environment reward.

    Returns per-episode length and return; if an oracle is given, also the policy
    accuracy and whether the greedy path is optimal after each episode; plus copies
    of Q after the episodes in `snapshot_at` (0 = before training)."""
    sarsa = isinstance(agent, SarsaAgent)
    lengths = np.zeros(n_episodes, dtype=int)
    returns = np.zeros(n_episodes)
    accuracy = np.full(n_episodes, np.nan)
    path_ok = np.zeros(n_episodes, dtype=bool)
    snapshots = {0: agent.Q.copy()} if 0 in snapshot_at else {}
    for ep in range(n_episodes):
        s = env.reset()
        a = agent.act(s)
        G = 0.0
        while True:
            s2, r, terminated, truncated = env.step(a)
            G += r
            if sarsa:                        # Sarsa picks A' before updating
                a2 = agent.act(s2)
                agent.update(s, a, r, s2, terminated, a2)
            else:                            # Q-learning picks A' after updating
                agent.update(s, a, r, s2, terminated)
                a2 = None if (terminated or truncated) else agent.act(s2)
            if terminated or truncated:
                break
            s, a = s2, a2
        lengths[ep], returns[ep] = env.t, G
        if oracle is not None:
            accuracy[ep] = policy_accuracy(agent.Q, oracle)
            path_ok[ep] = greedy_path_optimal(agent.Q, env, oracle)
        if ep + 1 in snapshot_at:
            snapshots[ep + 1] = agent.Q.copy()
    return {"length": lengths, "return": returns, "accuracy": accuracy,
            "path_optimal": path_ok, "snapshots": snapshots}


def greedy_route(Q, env, max_len=60):
    """Cells visited when following the greedy policy from the start (for plotting)."""
    s, route = env.start, [env.start]
    for _ in range(max_len):
        a = int(Q[s].argmax())
        s2 = env._T[s][a]
        if s2 == s or env._type[s2] == LAVA:
            break
        route.append(s2)
        s = s2
        if env._type[s] == GOAL:
            break
    return route


# %% [trainer]
class SimulatedTrainer:
    """A simulated person who watches every move and may press GOOD (+1) or BAD (-1).

    The trainer knows the shortest-path oracle, but
      * speaks only with probability `feedback_prob` (otherwise silent: h = 0), and
      * when speaking, gives the wrong sign with probability `error_rate`.
    The counters record how much feedback was given and how much of it was correct."""

    def __init__(self, oracle, error_rate=0.0, feedback_prob=1.0, seed=None):
        if not (0.0 <= error_rate <= 1.0 and 0.0 <= feedback_prob <= 1.0):
            raise ValueError("error_rate and feedback_prob must be in [0, 1]")
        self.oracle, self.error_rate, self.feedback_prob = oracle, error_rate, feedback_prob
        self.rng = np.random.default_rng(seed)
        self.n_given = 0
        self.n_correct = 0

    def __call__(self, s, a):
        if self.feedback_prob <= 0.0 or self.rng.random() >= self.feedback_prob:
            return 0
        good = self.oracle._opt[s][a]
        h = 1 if good else -1
        if self.rng.random() < self.error_rate:
            h = -h
        self.n_given += 1
        self.n_correct += int((h > 0) == good)
        return h


def run_with_trainer(layout="A", error_rate=0.0, feedback_prob=1.0, seed=0, max_steps=3000,
                     alpha=0.5, gamma=0.95, epsilon=0.1, beta=1.0, record_every=10,
                     stop_when_learned=False, env_kwargs=None):
    """One run of Q-learning with a simulated trainer.

    The agent learns from   r_total = r_env + beta * h,   with h in {+1, -1, 0} the
    trainer's feedback on the action just taken (human feedback as reward shaping).

    Returns a dict with
      steps_to_learn     environment steps until the greedy path from the start is
                         optimal (np.nan if that did not happen within max_steps)
      feedback_to_learn  feedback signals given until then (all of them, if never learned)
      correct_to_learn   how many of those were correct
      accuracy_curve     policy accuracy (all cells) every `record_every` steps
      route_curve        policy accuracy on the shortest route, every `record_every` steps
      learned_curve      whether the greedy path was optimal, every `record_every` steps
    """
    env = GridWorld(layout, **(env_kwargs or {}))
    oracle = Oracle(env)
    agent_seed, trainer_seed = np.random.SeedSequence(seed).spawn(2)
    agent = QLearningAgent(env.n_states, alpha=alpha, gamma=gamma, epsilon=epsilon,
                           seed=agent_seed)
    trainer = SimulatedTrainer(oracle, error_rate, feedback_prob, seed=trainer_seed)
    n_points = max_steps // record_every if record_every else 0
    acc_curve = np.full(n_points, np.nan)
    route_curve = np.full(n_points, np.nan)
    learned_curve = np.zeros(n_points, dtype=bool)
    steps_to_learn = np.nan
    fb_to_learn = correct_to_learn = None
    episodes = 1
    s = env.reset()
    for t in range(1, max_steps + 1):
        a = agent.act(s)
        s2, r, terminated, truncated = env.step(a)
        h = trainer(s, a)
        agent.update(s, a, r + beta * h, s2, terminated)
        if np.isnan(steps_to_learn) and greedy_path_optimal(agent.Q, env, oracle):
            steps_to_learn = float(t)
            fb_to_learn, correct_to_learn = trainer.n_given, trainer.n_correct
            if stop_when_learned:
                break
        if n_points and t % record_every == 0:
            i = t // record_every - 1
            acc_curve[i] = policy_accuracy(agent.Q, oracle)
            route_curve[i] = route_accuracy(agent.Q, oracle)
            learned_curve[i] = greedy_path_optimal(agent.Q, env, oracle)
        if terminated or truncated:
            s = env.reset()
            episodes += 1
        else:
            s = s2
    if fb_to_learn is None:
        fb_to_learn, correct_to_learn = trainer.n_given, trainer.n_correct
    return {
        "error_rate": error_rate, "feedback_prob": feedback_prob, "seed": seed,
        "steps_to_learn": steps_to_learn, "learned": bool(not np.isnan(steps_to_learn)),
        "feedback_to_learn": fb_to_learn, "correct_to_learn": correct_to_learn,
        "feedback_total": trainer.n_given, "correct_total": trainer.n_correct,
        "episodes": episodes, "accuracy_curve": acc_curve, "route_curve": route_curve,
        "learned_curve": learned_curve, "final_accuracy": policy_accuracy(agent.Q, oracle),
        "Q": agent.Q,
    }


def run_many(n_seeds, seed0=0, **kwargs):
    """run_with_trainer over seeds seed0 .. seed0 + n_seeds - 1. Every condition uses the
    same seeds, so conditions are compared on matched random streams."""
    return [run_with_trainer(seed=seed0 + i, **kwargs) for i in range(n_seeds)]


def censored_steps(runs, max_steps):
    """steps_to_learn of each run, with runs that never learned set to max_steps."""
    x = np.array([r["steps_to_learn"] for r in runs], dtype=float)
    return np.where(np.isnan(x), float(max_steps), x)


def sweep(error_rates, feedback_probs, n_seeds, max_steps=3000, **kwargs):
    """Time-to-learn for every (error rate, feedback probability) pair. One row per run."""
    rows = []
    for p in feedback_probs:
        for e in error_rates:
            for i in range(n_seeds):
                r = run_with_trainer(error_rate=e, feedback_prob=p, seed=i, max_steps=max_steps,
                                     record_every=0, stop_when_learned=True, **kwargs)
                rows.append({k: r[k] for k in ("error_rate", "feedback_prob", "seed",
                                               "steps_to_learn", "learned",
                                               "feedback_to_learn", "correct_to_learn",
                                               "episodes")})
    df = pd.DataFrame(rows)
    df["steps_to_learn_c"] = df["steps_to_learn"].fillna(float(max_steps))
    return df


# %% [plotting]
# Colours come from a validated, colour-blind-safe palette (checked with a CVD
# validator): categorical slots for nominal groups, a one-hue blue ramp for ordered
# levels such as error rates, grey for reference lines, status colours only for
# good/bad meaning (always with a text label).
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb"}
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]               # categorical slots 1-3
RAMP4 = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]   # ordinal blue ramp, light -> dark
BLUE_SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
STATUS = {"good": "#0ca30c", "critical": "#d03b3b", "good_text": "#006300"}
CELL = {"wall": "#52514e", "lava": "#d03b3b", "goal": "#0ca30c", "start": "#cde2fb",
        "free": "#fcfcfb", "line": "#c3c2b7"}
AGENT_COLOR, MOVE_COLOR, POLICY_COLOR = "#2a78d6", "#eb6834", "#6250d6"


def set_style():
    """Quiet, consistent matplotlib defaults for every figure in the project."""
    plt.rcParams.update({
        "figure.facecolor": INK["surface"], "axes.facecolor": INK["surface"],
        "savefig.facecolor": INK["surface"], "savefig.bbox": "tight",
        "figure.dpi": 100, "savefig.dpi": 150,
        "axes.edgecolor": INK["axis"], "axes.labelcolor": INK["secondary"],
        "axes.titlecolor": INK["primary"], "axes.titleweight": "bold",
        "axes.titlesize": 11, "axes.titlelocation": "left", "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": INK["grid"], "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.color": INK["axis"], "ytick.color": INK["axis"],
        "xtick.labelcolor": INK["secondary"], "ytick.labelcolor": INK["secondary"],
        "legend.frameon": False, "legend.fontsize": 9, "font.size": 10,
        "lines.linewidth": 1.8, "axes.prop_cycle": cycler(color=CAT),
        "figure.titlesize": 12.5, "figure.titleweight": "bold",
    })


def seq_cmap(colors=None, name="seq"):
    """One-hue sequential colormap (light = low, dark = high)."""
    return LinearSegmentedColormap.from_list(name, colors or BLUE_SEQ)


def div_cmap():
    """Diverging colormap: red (slower) <- grey (no change) -> blue (faster)."""
    return LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])


def mean_ci(x, axis=0):
    """Mean and 95% normal-approximation confidence half-width along `axis`."""
    x = np.asarray(x, dtype=float)
    n = np.sum(~np.isnan(x), axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.nanmean(x, axis=axis)
        sd = np.nanstd(x, axis=axis, ddof=1)
        half = 1.96 * sd / np.sqrt(n)
    return m, half


def smooth(y, w):
    """Trailing moving average with window w (output has the same length as y)."""
    y = np.asarray(y, dtype=float)
    if w <= 1:
        return y
    c = np.concatenate([[0.0], np.cumsum(y)])
    i = np.arange(len(y))
    lo = np.maximum(0, i - w + 1)
    return (c[i + 1] - c[lo]) / (i + 1 - lo)


def label_end(ax, x, y, text, dy=0.0):
    """Direct label to the right of a line end, in ink colour (the line carries identity)."""
    ax.annotate(text, xy=(x, y), xytext=(5, dy), textcoords="offset points", va="center",
                ha="left", fontsize=8.5, color=INK["secondary"], annotation_clip=False)


def label_ends(ax, items, min_gap=11.0, dx=5.0):
    """Direct labels at several line ends, nudged apart vertically so they never overlap.
    `items` is a list of (x, y, text) in data coordinates; `min_gap` is in points.
    Call it after the axis limits, scales and layout are final."""
    if not items:
        return
    fig = ax.figure
    px_per_pt = fig.dpi / 72.0
    placed = []
    for x, y, text in items:
        _, ydisp = ax.transData.transform((x, y))
        placed.append([ydisp, x, y, text])
    placed.sort(key=lambda p: p[0])
    original = np.array([p[0] for p in placed])
    target = original.copy()
    for i in range(1, len(target)):                   # push labels up until spaced
        target[i] = max(target[i], target[i - 1] + min_gap * px_per_pt)
    target -= np.mean(target - original)               # re-centre: spacing is unchanged
    for (ydisp, x, y, text), ty in zip(placed, target):
        ax.annotate(text, xy=(x, y), xytext=(dx, (ty - ydisp) / px_per_pt),
                    textcoords="offset points", va="center", ha="left", fontsize=8.5,
                    color=INK["secondary"], annotation_clip=False)


def draw_grid(ax, env, values=None, cmap=None, vmin=None, vmax=None, labels=True,
              show_start=True, fontsize=8):
    """Draw the gridworld on `ax` with row 0 at the top. Optional `values` (one per state,
    e.g. V(s) = max_a Q(s, a)) colour the free cells. Returns the colour Normalize used."""
    ax.set_xlim(-0.5, env.W - 0.5)
    ax.set_ylim(env.H - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    norm = None
    if values is not None:
        cmap = cmap or seq_cmap()
        free = np.array([env.cell_type[s] == FREE and np.isfinite(values[s])
                         for s in range(env.n_states)])
        vals = np.asarray(values, dtype=float)[free]
        lo = vals.min() if vmin is None else vmin
        hi = vals.max() if vmax is None else vmax
        norm = Normalize(lo, hi if hi > lo else lo + 1e-9)
    for s in range(env.n_states):
        r, c = env.rc(s)
        kind = env.cell_type[s]
        if kind == WALL:
            face = CELL["wall"]
        elif kind == LAVA:
            face = CELL["lava"]
        elif kind == GOAL:
            face = CELL["goal"]
        elif values is not None and np.isfinite(values[s]):
            face = cmap(norm(values[s]))
        elif s == env.start and show_start:
            face = CELL["start"]
        else:
            face = CELL["free"]
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=face,
                               edgecolor=CELL["line"], lw=1.0, zorder=1))
        if labels and kind == GOAL:
            ax.text(c, r, "GOAL", ha="center", va="center", color="white",
                    fontsize=fontsize, weight="bold", zorder=2)
        elif labels and kind == LAVA:
            ax.text(c, r, "LAVA", ha="center", va="center", color="white",
                    fontsize=fontsize, weight="bold", zorder=2)
        elif labels and s == env.start and show_start:
            ax.text(c, r + 0.36, "start", ha="center", va="center", color=INK["secondary"],
                    fontsize=fontsize - 1, zorder=2)
    return norm


def draw_policy(ax, Q, env, oracle, color=POLICY_COLOR, mark_wrong=False, length=0.58,
                show_ties=True):
    """Arrows for the agent's greedy action in every non-terminal cell; a small grey dot
    where actions are still tied. With mark_wrong=True, actions that are not on a
    shortest path are drawn in red (analysis figures only - never shown to pilot users)."""
    g = strict_greedy(Q)
    X, Y, U, V, colors, tie_x, tie_y = [], [], [], [], [], [], []
    for s in oracle.states:
        r, c = env.rc(s)
        a = g[s]
        if a < 0:
            tie_x.append(c)
            tie_y.append(r)
            continue
        dr, dc = ACTIONS[a]
        X.append(c)
        Y.append(r)
        U.append(dc * length)
        V.append(dr * length)
        wrong = mark_wrong and not oracle.optimal[s, a]
        colors.append(STATUS["critical"] if wrong else color)
    if X:
        ax.quiver(X, Y, U, V, color=colors, angles="xy", scale_units="xy", scale=1,
                  pivot="middle", width=0.012, headwidth=4.0, headlength=4.0,
                  headaxislength=3.6, zorder=4)
    if show_ties and tie_x:
        ax.scatter(tie_x, tie_y, s=10, color=INK["muted"], zorder=4)


class GridRenderer:
    """Draws what a pilot participant sees: the map, the agent, the move it has just
    made and - in the explanation condition - the agent's current plan, i.e. its
    greedy action in every cell (faded arrow = less sure). Shared by the pilot app,
    the interface figure and the training animation."""

    def __init__(self, ax, env, show_policy, oracle=None):
        self.ax, self.env, self.show_policy = ax, env, show_policy
        self.oracle = oracle if oracle is not None else Oracle(env)
        draw_grid(ax, env, fontsize=9)
        r0, c0 = env.rc(env.start)
        self.ghost = Circle((c0, r0), 0.2, facecolor=AGENT_COLOR, alpha=0.22,
                            edgecolor="none", zorder=5, visible=False)
        self.agent = Circle((c0, r0), 0.25, facecolor=AGENT_COLOR, edgecolor="white",
                            lw=1.5, zorder=6)
        self.move = FancyArrowPatch((c0, r0), (c0, r0), arrowstyle="-|>", mutation_scale=16,
                                    lw=2.6, color=MOVE_COLOR, zorder=7, shrinkA=0, shrinkB=0,
                                    visible=False)
        for patch in (self.ghost, self.agent, self.move):
            ax.add_patch(patch)
        (self.bump,) = ax.plot([], [], linestyle="none", marker="x", ms=11, mew=2.8,
                               color=MOVE_COLOR, zorder=7)
        self.states = self.oracle.states
        self.xs = np.array([env.rc(s)[1] for s in self.states], dtype=float)
        self.ys = np.array([env.rc(s)[0] for s in self.states], dtype=float)
        self.quiver = self.ties = None
        if show_policy:
            n = len(self.states)
            self.quiver = ax.quiver(self.xs, self.ys, np.zeros(n), np.zeros(n), angles="xy",
                                    scale_units="xy", scale=1, pivot="middle",
                                    color=POLICY_COLOR, width=0.011, headwidth=4.2,
                                    headlength=4.2, headaxislength=3.8, zorder=3)
            self.ties = ax.scatter(self.xs[:0], self.ys[:0], s=9, color=INK["muted"], zorder=3)

    def update(self, Q, s_prev=None, a=None, s_now=None):
        """Show the agent at s_now after taking action a in s_prev (None = just reset)."""
        env = self.env
        s_now = env.start if s_now is None else s_now
        r, c = env.rc(s_now)
        self.agent.center = (c, r)
        self.bump.set_data([], [])
        self.move.set_visible(False)
        self.ghost.set_visible(False)
        if s_prev is not None and a is not None:
            pr, pc = env.rc(s_prev)
            dr, dc = ACTIONS[a]
            self.ghost.center = (pc, pr)
            self.ghost.set_visible(True)
            if s_now == s_prev:                       # walked into a wall or the edge
                self.bump.set_data([pc + 0.42 * dc], [pr + 0.42 * dr])
            else:
                # A short arrow across the border between the two cells, so it never
                # hides the policy arrow drawn in the middle of the cell it came from.
                self.move.set_positions((pc + 0.34 * dc, pr + 0.34 * dr),
                                        (c - 0.29 * dc, r - 0.29 * dr))
                self.move.set_visible(True)
        if self.show_policy:
            q = Q[self.states]
            g = strict_greedy(Q)[self.states]
            has = g >= 0
            moves = np.array(ACTIONS, dtype=float)[np.maximum(g, 0)]
            U = np.where(has, moves[:, 1] * 0.56, 0.0)
            V = np.where(has, moves[:, 0] * 0.56, 0.0)
            top2 = np.sort(q, axis=1)[:, -2:]
            confidence = 1.0 - np.exp(-(top2[:, 1] - top2[:, 0]) / 0.25)
            rgba = np.tile(np.array(to_rgba(POLICY_COLOR)), (len(g), 1))
            rgba[:, 3] = np.where(has, 0.3 + 0.7 * confidence, 0.0)
            self.quiver.set_UVC(U, V)
            self.quiver.set_facecolor(rgba)
            self.quiver.set_edgecolor(rgba)
            self.ties.set_offsets(np.column_stack([self.xs[~has], self.ys[~has]]))


# %% [pilot]
CONDITIONS = ("agent_only", "explanation")
CONDITION_LABELS = {"agent_only": "Agent only", "explanation": "Agent + policy arrows",
                    "practice": "Practice"}


def participant_number(participant):
    """P1 -> 1, 'p12' -> 12. Raises ValueError for IDs without a number."""
    m = re.search(r"(\d+)", str(participant))
    if not m:
        raise ValueError(f"participant id {participant!r} needs a number, e.g. P1")
    return int(m.group(1))


def counterbalanced_order(participant):
    """Odd participants see 'agent only' first, even participants see 'explanation'
    first. Session 1 always uses map A and session 2 its mirror image (map B), so map
    and condition are crossed across participants."""
    first = "agent_only" if participant_number(participant) % 2 == 1 else "explanation"
    second = "explanation" if first == "agent_only" else "agent_only"
    return [(1, first, "A"), (2, second, "B")]


class PilotSession:
    """The logic of one pilot session, independent of any window or keyboard.

    One agent step is on screen at a time. While it is shown, the participant may press
    GOOD or BAD (the last key pressed counts). When the next tick arrives the step is
    closed: the agent learns from r_env + beta * h, the step is logged, and the agent
    takes its next action. The session ends when the agent has learned the task (its
    greedy path from the start is optimal) or after `max_steps` steps."""

    LOG_FIELDS = ["participant", "session", "condition", "layout", "step", "episode",
                  "row", "col", "action", "next_row", "next_col", "env_reward",
                  "terminated", "truncated", "optimal_action", "feedback",
                  "feedback_correct", "rt_ms", "policy_accuracy", "route_accuracy",
                  "greedy_path_optimal", "learned"]

    def __init__(self, participant, session, condition, layout="A", seed=0, max_steps=300,
                 alpha=0.5, gamma=0.95, epsilon=0.1, beta=1.0, env_kwargs=None):
        if condition not in CONDITIONS + ("practice",):
            raise ValueError(f"unknown condition {condition!r}")
        self.participant = re.sub(r"[^A-Za-z0-9_-]", "", str(participant)) or "P0"
        self.session, self.condition, self.layout = int(session), condition, layout
        self.env = GridWorld(layout, **(env_kwargs or {}))
        self.oracle = Oracle(self.env)
        self.agent = QLearningAgent(self.env.n_states, alpha=alpha, gamma=gamma,
                                    epsilon=epsilon, seed=seed)
        self.beta, self.max_steps, self.seed = beta, int(max_steps), seed
        self.config = {"alpha": alpha, "gamma": gamma, "epsilon": epsilon, "beta": beta,
                       "max_steps": int(max_steps), "seed": seed,
                       "env": dict(env_kwargs or {})}
        self.s = self.env.reset()
        self.t, self.episode = 0, 1
        self.pending = None
        self.rows = []
        self.learned_at = None
        self.finished = self.aborted = False
        self.n_feedback = self.n_good = self.n_bad = self.n_correct = 0
        self.fb_at_learn = self.correct_at_learn = None
        self._h, self._rt, self._shown_at = 0, None, None
        self.started_at = self.ended_at = None

    # -- driving the session ---------------------------------------------------------
    def start(self, now=None):
        """Show the first step."""
        self.started_at = time.time()
        self._take_action(now)

    def give_feedback(self, h, now=None):
        """Record a key press for the step on screen: +1 = GOOD, -1 = BAD."""
        if self.pending is None or self.finished or h not in (1, -1):
            return False
        now = time.perf_counter() if now is None else now
        self._h = h
        self._rt = None if self._shown_at is None else 1000.0 * (now - self._shown_at)
        return True

    def advance(self, now=None):
        """Close the step on screen and show the next one. Returns False once finished."""
        if self.finished:
            return False
        if self.pending is not None:
            self._close_step()
        if self.finished:
            self.ended_at = time.time()
            return False
        self._take_action(now)
        return True

    def abort(self):
        """Stop early (e.g. the participant pressed Esc); the step on screen is kept."""
        if not self.finished:
            if self.pending is not None:
                self._close_step()
            self.finished = self.aborted = True
            self.ended_at = time.time()

    # -- internals ---------------------------------------------------------------------
    def _take_action(self, now):
        s = self.s
        a = self.agent.act(s)
        s2, r, terminated, truncated = self.env.step(a)
        self.pending = {"step": self.t + 1, "episode": self.episode, "s": s, "a": a,
                        "r": r, "s2": s2, "terminated": terminated, "truncated": truncated}
        self._h, self._rt = 0, None
        self._shown_at = time.perf_counter() if now is None else now

    def _close_step(self):
        p, h = self.pending, self._h
        s, a = p["s"], p["a"]
        self.agent.update(s, a, p["r"] + self.beta * h, p["s2"], p["terminated"])
        self.t = p["step"]
        optimal = bool(self.oracle.optimal[s, a])
        correct = None
        if h != 0:
            correct = (h > 0) == optimal
            self.n_feedback += 1
            self.n_good += int(h > 0)
            self.n_bad += int(h < 0)
            self.n_correct += int(correct)
        path_ok = greedy_path_optimal(self.agent.Q, self.env, self.oracle)
        if path_ok and self.learned_at is None:
            self.learned_at = self.t
            self.fb_at_learn, self.correct_at_learn = self.n_feedback, self.n_correct
        r0, c0 = self.env.rc(s)
        r1, c1 = self.env.rc(p["s2"])
        self.rows.append({
            "participant": self.participant, "session": self.session,
            "condition": self.condition, "layout": self.layout, "step": self.t,
            "episode": p["episode"], "row": r0, "col": c0, "action": ACTION_NAMES[a],
            "next_row": r1, "next_col": c1, "env_reward": p["r"],
            "terminated": int(p["terminated"]), "truncated": int(p["truncated"]),
            "optimal_action": int(optimal), "feedback": h,
            "feedback_correct": "" if correct is None else int(correct),
            "rt_ms": "" if (h == 0 or self._rt is None) else round(self._rt, 1),
            "policy_accuracy": round(policy_accuracy(self.agent.Q, self.oracle), 4),
            "route_accuracy": round(route_accuracy(self.agent.Q, self.oracle), 4),
            "greedy_path_optimal": int(path_ok), "learned": int(self.learned_at is not None),
        })
        if p["terminated"] or p["truncated"]:
            self.s = self.env.reset()
            self.episode += 1
        else:
            self.s = p["s2"]
        self.pending = None
        if self.learned_at is not None or self.t >= self.max_steps:
            self.finished = True

    # -- results -----------------------------------------------------------------------
    def summary(self):
        rts = [row["rt_ms"] for row in self.rows if row["rt_ms"] != ""]
        learned = self.learned_at is not None
        return {
            "participant": self.participant, "session": self.session,
            "condition": self.condition, "layout": self.layout, "seed": self.seed,
            "learned": learned, "steps_to_learn": self.learned_at, "steps_run": self.t,
            "max_steps": self.max_steps, "aborted": self.aborted,
            "feedback_to_learn": self.fb_at_learn if learned else self.n_feedback,
            "correct_to_learn": self.correct_at_learn if learned else self.n_correct,
            "feedback_total": self.n_feedback, "good_total": self.n_good,
            "bad_total": self.n_bad, "correct_total": self.n_correct,
            "accuracy": (self.n_correct / self.n_feedback) if self.n_feedback else None,
            "feedback_rate": (self.n_feedback / self.t) if self.t else None,
            "median_rt_ms": float(np.median(rts)) if rts else None,
            "episodes": self.episode,
            "final_policy_accuracy": policy_accuracy(self.agent.Q, self.oracle),
            "started_at": self.started_at, "ended_at": self.ended_at,
            "config": self.config, "version": __version__,
        }

    def save(self, out_dir, extra=None):
        """Write <id>_s<k>_<condition>_<map>_steps.csv and ..._summary.json."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = f"{self.participant}_s{self.session}_{self.condition}_{self.layout}"
        steps_path = out / f"{stem}_steps.csv"
        with open(steps_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.LOG_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        summary = self.summary()
        summary.update(extra or {})
        summary_path = out / f"{stem}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        return steps_path, summary_path


class SimulatedResponder:
    """A stand-in for a participant, used ONLY to test the pilot pipeline end to end.
    It presses a key with probability `feedback_prob`; the key is wrong with
    probability `error_rate`; reaction times are log-normal around `rt_ms`."""

    def __init__(self, oracle, feedback_prob=0.6, error_rate=0.15, rt_ms=450.0, seed=None):
        self.oracle, self.feedback_prob, self.error_rate = oracle, feedback_prob, error_rate
        self.rt_ms = rt_ms
        self.rng = np.random.default_rng(seed)

    def respond(self, pending):
        if pending is None or self.rng.random() >= self.feedback_prob:
            return 0, None
        h = 1 if self.oracle.optimal[pending["s"], pending["a"]] else -1
        if self.rng.random() < self.error_rate:
            h = -h
        return h, float(self.rt_ms * np.exp(self.rng.normal(0.0, 0.25)))


def run_session_headless(session, responder, step_ms=800.0):
    """Drive a PilotSession with a SimulatedResponder on a virtual clock (no window).
    Responses slower than one step are dropped, as a real late key press would be
    attributed to the next step."""
    clock = 0.0
    session.start(now=clock)
    while not session.finished:
        h, rt = responder.respond(session.pending)
        if h and rt is not None and rt < step_ms:
            session.give_feedback(h, now=clock + rt / 1000.0)
        clock += step_ms / 1000.0
        session.advance(now=clock)
    return session


# %% [analysis]
PILOT_METRICS = [
    ("feedback_to_learn", "Feedback needed", "key presses until the agent learned"),
    ("accuracy", "Feedback that was correct", "share of key presses that were right"),
    ("steps_to_learn_c", "Agent learning speed", "agent steps until learned (fewer = faster)"),
]


def load_pilot(data_dir):
    """Load every session saved by the pilot app in `data_dir`.
    Returns (sessions, steps) DataFrames without practice rounds, or (None, None)."""
    d = Path(data_dir)
    summaries = sorted(d.glob("*_summary.json"))
    if not summaries:
        return None, None
    sessions = pd.DataFrame([json.loads(p.read_text()) for p in summaries])
    sessions = sessions[sessions["condition"].isin(CONDITIONS)].reset_index(drop=True)
    if sessions.empty:
        return None, None
    sessions["aborted"] = sessions["aborted"].astype(bool)
    sessions["steps_to_learn_c"] = sessions["steps_to_learn"].astype(float).fillna(
        sessions["max_steps"].astype(float))
    frames = [pd.read_csv(p) for p in sorted(d.glob("*_steps.csv"))]
    steps = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not steps.empty:
        steps = steps[steps["condition"].isin(CONDITIONS)].reset_index(drop=True)
    return sessions, steps


def pilot_table(sessions):
    """One row per participant x condition with the three pre-specified measures."""
    cols = ["participant", "condition", "layout", "session", "learned", "steps_to_learn_c",
            "feedback_to_learn", "correct_to_learn", "accuracy", "feedback_rate",
            "median_rt_ms", "aborted"]
    t = sessions[cols].copy()
    t["condition"] = t["condition"].map(CONDITION_LABELS)
    return t.sort_values(["participant", "session"]).reset_index(drop=True)


def pilot_medians(sessions):
    """Median of each measure per condition (descriptive only)."""
    keep = sessions[~sessions["aborted"]]
    out = keep.groupby("condition")[[m for m, _, _ in PILOT_METRICS] + ["feedback_rate"]].median()
    out.index = [CONDITION_LABELS[c] for c in out.index]
    return out


def plot_pilot_paired(sessions, banner=None):
    """Each participant's value in both conditions (grey line = same person).
    Hollow points mark sessions where the agent had not learned by the step limit."""
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9))
    keep = sessions[~sessions["aborted"]]
    labels = {}
    for ax, (col, title, ylabel) in zip(axes, PILOT_METRICS):
        labels[ax] = []
        for pid, g in keep.groupby("participant"):
            vals = [g.loc[g["condition"] == c, col] for c in CONDITIONS]
            if all(len(v) for v in vals):
                y0, y1 = float(vals[0].iloc[0]), float(vals[1].iloc[0])
                ax.plot([0, 1], [y0, y1], color=INK["axis"], lw=1.3, zorder=1)
                labels[ax].append((1.03, y1, pid))
        if col == "accuracy":
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        for i, c in enumerate(CONDITIONS):
            g = keep[keep["condition"] == c]
            v = g[col].astype(float)
            learned = g["learned"].astype(bool).values
            ax.scatter(np.full(learned.sum(), i), v[learned], s=46, color=CAT[i], zorder=3,
                       edgecolor=INK["surface"], lw=1.5)
            ax.scatter(np.full((~learned).sum(), i), v[~learned], s=46, facecolor="none",
                       edgecolor=CAT[i], lw=1.8, zorder=3)
            if len(v):
                ax.plot([i - 0.2, i + 0.2], [v.median()] * 2, color=INK["primary"], lw=2.2,
                        zorder=4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS])
        ax.set_xlim(-0.45, 1.6)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)
    if banner:
        fig.suptitle(banner, color=STATUS["critical"], fontsize=11, x=0.01, ha="left")
    fig.text(0.01, -0.03, "Each point is one session; grey lines join the same person; "
             "black bar = median; hollow = not learned by the step limit.",
             fontsize=8.5, color=INK["secondary"])
    fig.tight_layout()
    for ax, items in labels.items():
        label_ends(ax, items, min_gap=10.0, dx=2.0)
    return fig


def plot_pilot_curves(steps, banner=None):
    """How much of the shortest route the agent has right over the session (1 = learned),
    one small panel per participant."""
    ycol = "route_accuracy" if "route_accuracy" in steps.columns else "policy_accuracy"
    pids = sorted(steps["participant"].unique(), key=lambda p: (participant_number(p), p))
    n = len(pids)
    ncols = min(4, max(1, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.8 * nrows), squeeze=False,
                             sharey=True)
    for k, pid in enumerate(pids):
        ax = axes[k // ncols][k % ncols]
        for i, c in enumerate(CONDITIONS):
            g = steps[(steps["participant"] == pid) & (steps["condition"] == c)]
            if len(g):
                ax.plot(g["step"], g[ycol], color=CAT[i], lw=1.8, label=CONDITION_LABELS[c])
                done = g[g["learned"] == 1]
                if len(done):
                    ax.scatter([done["step"].iloc[0]], [done[ycol].iloc[0]],
                               s=40, color=CAT[i], edgecolor=INK["surface"], zorder=3)
        ax.set_title(pid)
        ax.set_ylim(-0.02, 1.05)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_xlabel("agent step")
        if k % ncols == 0:
            ax.set_ylabel("route cells right\n(dot = learned)" if ycol == "route_accuracy"
                          else "policy accuracy")
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.02))
    if banner:
        fig.suptitle(banner, color=STATUS["critical"], fontsize=11, x=0.01, ha="left", y=1.02)
    fig.tight_layout()
    return fig


def canonical_col(layout, col, width=5):
    """Map B is the mirror of map A; express its columns in map-A coordinates."""
    return width - 1 - col if layout == "B" else col


def feedback_error_map(steps):
    """Per condition: for every cell (map-A coordinates) the number of key presses and
    the share that were wrong according to the shortest-path oracle."""
    fb = steps[steps["feedback"] != 0].copy()
    fb["c_col"] = [canonical_col(l, c) for l, c in zip(fb["layout"], fb["col"])]
    fb["wrong"] = 1 - fb["feedback_correct"].astype(int)
    out = {}
    for c in CONDITIONS:
        g = fb[fb["condition"] == c]
        n = np.zeros((5, 5))
        w = np.zeros((5, 5))
        for (r, cc), grp in g.groupby(["row", "c_col"]):
            n[int(r), int(cc)] = len(grp)
            w[int(r), int(cc)] = grp["wrong"].sum()
        out[c] = (n, w)
    return out


def plot_error_map(steps, banner=None):
    """Where were key presses wrong? Shade = share of wrong presses in that cell."""
    maps = feedback_error_map(steps)
    env = GridWorld("A")
    oracle = Oracle(env)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.4))
    cmap = seq_cmap()
    for ax, c in zip(axes, CONDITIONS):
        n, w = maps[c]
        share = np.full(env.n_states, np.nan)
        for s in range(env.n_states):
            r, cc = env.rc(s)
            if n[r, cc] > 0 and env.cell_type[s] == FREE:
                share[s] = w[r, cc] / n[r, cc]
        draw_grid(ax, env, values=share, cmap=cmap, vmin=0.0, vmax=1.0)
        for s in oracle.states:
            r, cc = env.rc(s)
            if n[r, cc] > 0:
                txt_color = "white" if share[s] > 0.55 else INK["primary"]
                ax.text(cc, r, f"{int(w[r, cc])}/{int(n[r, cc])}", ha="center", va="center",
                        fontsize=8.5, color=txt_color, zorder=3)
        ax.set_title(f"{CONDITION_LABELS[c]}")
    sm = plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap)
    cb = fig.colorbar(sm, ax=list(axes), shrink=0.75, pad=0.02)
    cb.set_label("share of key presses that were wrong", color=INK["secondary"])
    cb.outline.set_visible(False)
    fig.text(0.01, 0.02, "Numbers = wrong / all key presses in that cell (map B is mirrored "
             "onto map A).", fontsize=8.5, color=INK["secondary"])
    if banner:
        fig.suptitle(banner, color=STATUS["critical"], fontsize=11, x=0.01, ha="left")
    return fig


def predict_steps(sweep_df, error_rate, feedback_prob):
    """Simulation-predicted median time-to-learn for a trainer with the given error rate
    and feedback rate: bilinear interpolation of log(median) over the Part 2 sweep."""
    med = (sweep_df.groupby(["feedback_prob", "error_rate"])["steps_to_learn_c"].median()
           .unstack("error_rate"))
    ps = np.array(med.index, dtype=float)
    es = np.array(med.columns, dtype=float)
    z = np.log(med.values)
    e = float(np.clip(error_rate, es.min(), es.max()))
    p = float(np.clip(feedback_prob, ps.min(), ps.max()))
    i = int(np.clip(np.searchsorted(ps, p) - 1, 0, len(ps) - 2))
    j = int(np.clip(np.searchsorted(es, e) - 1, 0, len(es) - 2))
    tp = (p - ps[i]) / (ps[i + 1] - ps[i])
    te = (e - es[j]) / (es[j + 1] - es[j])
    top = z[i, j] * (1 - te) + z[i, j + 1] * te
    bottom = z[i + 1, j] * (1 - te) + z[i + 1, j + 1] * te
    return float(np.exp(top * (1 - tp) + bottom * tp))


def plot_observed_vs_predicted(sessions, sweep_df, banner=None):
    """Does the simulation predict how fast each session's agent learned, given only that
    person's feedback accuracy and feedback rate?"""
    keep = sessions[(~sessions["aborted"]) & sessions["accuracy"].notna()]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    lo, hi = np.inf, 0.0
    for i, c in enumerate(CONDITIONS):
        g = keep[keep["condition"] == c]
        pred = [predict_steps(sweep_df, 1 - a, f) for a, f in zip(g["accuracy"], g["feedback_rate"])]
        obs = g["steps_to_learn_c"].astype(float).values
        ax.scatter(pred, obs, s=50, color=CAT[i], edgecolor=INK["surface"], lw=1.5,
                   label=CONDITION_LABELS[c], zorder=3)
        for x, y, pid in zip(pred, obs, g["participant"]):
            ax.annotate(pid, (x, y), xytext=(5, 3), textcoords="offset points", fontsize=8,
                        color=INK["secondary"])
        if len(pred):
            lo, hi = min(lo, min(pred), obs.min()), max(hi, max(pred), obs.max())
    if np.isfinite(lo):
        ax.plot([lo * 0.8, hi * 1.25], [lo * 0.8, hi * 1.25], color=INK["muted"], ls="--",
                lw=1.2)
        ax.annotate("prediction = observation", xy=(hi * 1.25, hi * 1.25), xytext=(0, -12),
                    textcoords="offset points", fontsize=8, color=INK["muted"], ha="right",
                    va="top", rotation=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    plain = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(plain)
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("predicted steps to learn (simulation, Part 2)")
    ax.set_ylabel("observed steps to learn")
    ax.set_title("Observed vs simulated learning speed")
    ax.legend(loc="upper left")
    if banner:
        fig.suptitle(banner, color=STATUS["critical"], fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    return fig


def pilot_markdown(sessions):
    """Markdown block for the README: per-participant table and per-condition medians."""
    t = pilot_table(sessions)
    lines = [f"**Participants:** {sessions['participant'].nunique()} "
             f"(sessions: {len(sessions)}). Descriptive results only - see Limitations.", "",
             "| Participant | Session | Display | Map | Learned | Agent steps to learn "
             "| Key presses until learned | Correct presses | Accuracy | Presses per step |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        acc = "-" if pd.isna(r["accuracy"]) else f"{r['accuracy']:.0%}"
        rate = "-" if pd.isna(r["feedback_rate"]) else f"{r['feedback_rate']:.2f}"
        steps_txt = f"{int(r['steps_to_learn_c'])}" + ("" if r["learned"] else "+")
        lines.append(f"| {r['participant']} | {int(r['session'])} | {r['condition']} | "
                     f"{r['layout']} | {'yes' if r['learned'] else 'no'} | {steps_txt} | "
                     f"{int(r['feedback_to_learn'])} | {int(r['correct_to_learn'])} | "
                     f"{acc} | {rate} |")
    med = pilot_medians(sessions)
    lines += ["", "| Median per display | Key presses until learned | Accuracy "
              "| Agent steps to learn | Presses per step |", "|---|---|---|---|---|"]
    for name, r in med.iterrows():
        lines.append(f"| {name} | {r['feedback_to_learn']:.0f} | {r['accuracy']:.0%} | "
                     f"{r['steps_to_learn_c']:.0f} | {r['feedback_rate']:.2f} |")
    lines += ["", "`+` = the agent had not learned by the step limit (value is a lower bound).",
              "", "![Pilot: paired results](figures/pilot_paired.png)",
              "", "![Pilot: learning curves](figures/pilot_curves.png)",
              "", "![Pilot: where feedback was wrong](figures/pilot_error_map.png)"]
    return "\n".join(lines)


def update_readme_block(readme_path, block, start="<!-- PILOT-RESULTS:START -->",
                        end="<!-- PILOT-RESULTS:END -->"):
    """Replace the text between the two markers in the README with `block`."""
    p = Path(readme_path)
    text = p.read_text()
    i, j = text.find(start), text.find(end)
    if i < 0 or j < 0 or j < i:
        raise ValueError("README markers not found")
    p.write_text(text[: i + len(start)] + "\n" + block.strip() + "\n" + text[j:])


# %% [power]
def sign_flip_pvalue(d, n_perm=4000, rng=None):
    """Two-sided paired sign-flip (permutation) test of mean(d) = 0.
    Exact enumeration for n <= 12, Monte Carlo otherwise."""
    d = np.asarray(d, dtype=float)
    n = d.size
    if n == 0:
        return float("nan")
    obs = abs(d.mean())
    if n <= 12:
        signs = 1.0 - 2.0 * ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1)
        null = np.abs((signs * d).mean(axis=1))
        return float(np.mean(null >= obs - 1e-12))
    rng = np.random.default_rng(rng)
    signs = rng.choice((-1.0, 1.0), size=(n_perm, n))
    null = np.abs((signs * d).mean(axis=1))
    return float((1 + np.sum(null >= obs - 1e-12)) / (n_perm + 1))


def power_by_simulation(outcomes, errors, base_error=0.25, deltas=(0.05, 0.10, 0.20),
                        sd_between=0.07, ns=(4, 8, 12, 16, 24, 32), reps=400, alpha=0.05,
                        seed=0):
    """Power of a within-subject study (each person trains once with and once without
    explanations) to detect that explanations lower the feedback error rate by `delta`.

    outcomes[i]  simulated time-to-learn samples for error rate errors[i]
                 (one fixed feedback rate - see the notebook).
    Each simulated person has their own error rate e ~ Normal(base_error, sd_between)
    without explanations and e - delta with them; each of their two sessions is a fresh
    simulated run at the nearest simulated error rate. Test: sign-flip test on the
    per-person difference in log time-to-learn, two-sided, at level alpha."""
    rng = np.random.default_rng(seed)
    errors = np.asarray(errors, dtype=float)
    pools = [np.asarray(o, dtype=float) for o in outcomes]
    power = {}
    for delta in deltas:
        row = []
        for n in ns:
            hits = 0
            for _ in range(reps):
                e0 = np.clip(rng.normal(base_error, sd_between, n), 0.0, errors.max())
                e1 = np.clip(e0 - delta, 0.0, errors.max())
                i0 = np.abs(errors[None, :] - e0[:, None]).argmin(axis=1)
                i1 = np.abs(errors[None, :] - e1[:, None]).argmin(axis=1)
                y0 = np.array([pools[k][rng.integers(len(pools[k]))] for k in i0])
                y1 = np.array([pools[k][rng.integers(len(pools[k]))] for k in i1])
                p = sign_flip_pvalue(np.log(y0) - np.log(y1), n_perm=2000, rng=rng)
                hits += p < alpha
            row.append(hits / reps)
        power[delta] = np.array(row)
    return power
