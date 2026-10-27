# Source of explanation_feedback_rl.ipynb ("percent" format).
# Build the notebook with:  python tools/build_notebook.py
#   "# %% [markdown]"   -> markdown cell
#   "# %%"              -> code cell
#   "# %% [core:NAME]"  -> section NAME of src/hitl_gridworld.py, copied in verbatim

# %% [markdown]
# # Does seeing the agent's plan make human feedback more useful?
# ### A first test with Q-learning in a 5×5 gridworld
#
# **Pranto Bosu** · October 2026 · repository: [github.com/bosu0077/explanations-human-feedback-rl](https://github.com/bosu0077/explanations-human-feedback-rl)
#
# **Question.** When a person trains a reinforcement-learning (RL) agent by pressing *good* or *bad*, does an explanation of the agent's behaviour — here, arrows that show what the agent currently plans to do in every cell — make the person's feedback more useful?
#
# | Part | What is done | Main output |
# |---|---|---|
# | 1 | Tabular Q-learning (Sutton & Barto, 2018, Ch. 6) in a 5×5 gridworld, checked against the book's Cliff Walking example | learning curves |
# | 2 | A simulated trainer: +1 for a correct move, −1 for a wrong one — but wrong 10–30 % of the time, and sometimes silent | how much slower the agent learns as feedback gets worse |
# | 3 | A human pilot: 3–4 people press good/bad, once seeing only the agent and once seeing the agent plus its policy arrows | feedback needed, feedback correct, learning speed |
# | 4 | Next step: how many people a real study would need | power curves |
#
# > **Status of the human pilot.** The app (`pilot/pilot_app.py`), the protocol and the analysis are ready, but **no human data has been collected yet**. Until real logs are placed in `pilot/data/`, Part 3 runs the *same* analysis on **simulated participants**, only to check that the pipeline works. Those figures carry a red banner — *PIPELINE CHECK — simulated participants, not human data* — and are not results.
#
# The notebook runs top to bottom in a few minutes on a laptop (set the environment variable `NB_FAST=1` for a quick smoke test). All code lives in `src/hitl_gridworld.py`; its sections are embedded below so the notebook is self-contained.

# %% [markdown]
# ## 0. Setup

# %% [core:imports]

# %%
# ---- settings used by the whole notebook ------------------------------------------
import os
from IPython.display import Image, Markdown, display

FAST = os.environ.get("NB_FAST", "0") == "1"       # True: quick smoke test with fewer runs
N_SEEDS = 12 if FAST else 50                        # independent runs per condition
MAX_STEPS = 4000                                    # step budget per run in Part 2
ROOT = Path.cwd()
FIG, RES = ROOT / "figures", ROOT / "results"
PILOT_DATA = ROOT / "pilot" / "data"                # real pilot logs (pilot/pilot_app.py)
PIPELINE_CHECK = ROOT / "pilot" / "pipeline_check"  # simulated participants (pipeline test only)
for d in (FIG, RES):
    d.mkdir(parents=True, exist_ok=True)
T_START = time.time()
print(f"FAST={FAST} | runs per condition: {N_SEEDS} | numpy {np.__version__}, "
      f"matplotlib {matplotlib.__version__}, pandas {pd.__version__}")

# %% [markdown]
# ## Part 1 — Q-learning in a 5×5 gridworld (Sutton & Barto, Ch. 6)
#
# **Temporal-difference control.** Chapter 6 of Sutton & Barto (2018) learns action values $Q(s,a)$ from experience and updates them after every step using the *TD error*: the gap between the current estimate and a one-step, bootstrapped target. **Q-learning** (Sec. 6.5, Eq. 6.8) puts the best next action into the target,
#
# $$Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha\big[R_{t+1} + \gamma \max_a Q(S_{t+1},a) - Q(S_t,A_t)\big],$$
#
# so it learns the greedy policy while it *behaves* ε-greedily — it is **off-policy**. **Sarsa** (Sec. 6.4, Eq. 6.7) uses the action it will actually take next, $Q(S_{t+1},A_{t+1})$, so it learns the value of the ε-greedy policy it follows (**on-policy**). The book's Cliff Walking example (Example 6.6) shows the difference; it is reproduced below as a check on the code.
#
# **The task.** The agent starts bottom-left and must reach the goal top-right. Walls block moves (bumping into one wastes a step); stepping into lava ends the episode. The shortest route is 8 moves along the bottom row and up the right column; going up the left column leads to a 12-move detour.
#
# | | setting |
# |---|---|
# | rewards | +1 goal, −1 lava, −0.01 for every other step; episodes are cut off after 50 steps |
# | agent | Q-learning, α = 0.5, γ = 0.95, ε = 0.1, Q₀ = 0, random tie-breaking |
# | *learned* | the greedy policy from the start follows a shortest path (8 moves) |
# | *policy accuracy* | share of the 17 non-terminal cells whose greedy action is a shortest-path move |

# %% [core:env]

# %% [markdown]
# The **oracle** below knows the shortest path from every cell. It is used to score the agent (policy accuracy, *learned*), to drive the simulated trainer in Part 2, and to score human key presses in Part 3.

# %% [core:oracle]

# %% [markdown]
# Plotting helpers. Colours come from a colour-blind-safe palette that was checked with a contrast/colour-vision validator: blue and orange for the two pilot displays, one blue ramp for ordered levels (light = low, dark = high), grey for reference lines, and green/red only where something means good/bad — always together with a text label.

# %% [core:plotting]

# %%
set_style()


def save(fig, name, folder=None):
    """Save a figure to figures/ (or `folder`) and show it."""
    fig.savefig((folder or FIG) / name)
    plt.show()
    plt.close(fig)


def err_color(e, emax=0.5):
    """Error rate -> one-hue blue ramp (light = few errors, dark = many)."""
    return seq_cmap(RAMP4, "err")(min(e, emax) / emax)


env_A = GridWorld("A")
oracle_A = Oracle(env_A)
fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.7))
draw_grid(axes[0], env_A, fontsize=9)
axes[0].set_title("Map A — what the agent faces")
draw_grid(axes[1], env_A, fontsize=9, show_start=False)
arrows = {"middle": ([], [], [], []), "tail": ([], [], [], [])}
for s in oracle_A.states:
    r, c = env_A.rc(s)
    axes[1].text(c - 0.42, r - 0.30, f"{int(oracle_A.dist[s])}", fontsize=8,
                 color=INK["secondary"], ha="left", va="center")
    best_moves = np.flatnonzero(oracle_A.optimal[s])
    pivot = "middle" if len(best_moves) == 1 else "tail"   # two correct moves: both from centre
    for a in best_moves:
        dr, dc = ACTIONS[a]
        scale = 0.5 if pivot == "middle" else 0.36
        for lst, val in zip(arrows[pivot], (c, r, dc * scale, dr * scale)):
            lst.append(val)
for pivot, (X, Y, U, V) in arrows.items():
    if X:
        axes[1].quiver(X, Y, U, V, angles="xy", scale_units="xy", scale=1, pivot=pivot,
                       color=INK["primary"], width=0.012, headwidth=4.0, headlength=4.0,
                       headaxislength=3.6, zorder=4)
axes[1].set_title("Oracle — correct move(s); number = moves to goal")
fig.suptitle("Figure 1 — The 5×5 gridworld", x=0.01, ha="left")
fig.tight_layout()
save(fig, "fig01_gridworld.png")
print(f"Shortest path from start: {oracle_A.start_dist} moves; "
      f"{len(oracle_A.states)} non-terminal cells; cells with two correct moves: "
      f"{int((oracle_A.optimal.sum(axis=1) > 1).sum())}")

# %% [core:agents]

# %% [markdown]
# ### 1a. Sanity check — Cliff Walking (Sutton & Barto, Example 6.6)
#
# Reward −1 per step, −100 for stepping off the cliff (which also sends the agent back to the start), ε = 0.1, α = 0.5, γ = 1. The book reports that Q-learning learns the optimal path along the cliff edge but, because of ε-greedy exploration, keeps falling off, so its *online* reward is worse than Sarsa's, which learns the longer, safer path. Matching this pattern is a check that the agents are implemented correctly.

# %%
CLIFF_ENV = dict(step_reward=-1.0, goal_reward=-1.0, lava_reward=-100.0, lava_resets=True,
                 max_episode_steps=None)
N_CLIFF_RUNS, N_CLIFF_EPISODES = (10, 300) if FAST else (50, 500)
cliff = {}
for name, cls in (("Sarsa", SarsaAgent), ("Q-learning", QLearningAgent)):
    returns, final_Q = [], []
    for seed in range(N_CLIFF_RUNS):
        env_c = GridWorld("cliff", **CLIFF_ENV)
        agent = cls(env_c.n_states, alpha=0.5, gamma=1.0, epsilon=0.1, seed=seed)
        returns.append(run_episodes(env_c, agent, N_CLIFF_EPISODES)["return"])
        final_Q.append(agent.Q)
    cliff[name] = (np.array(returns), final_Q)


def typical_route(Qs, env):
    """Greedy route of the first run whose greedy policy reaches the goal."""
    for Q in Qs:
        route = greedy_route(Q, env)
        if env.cell_type[route[-1]] == GOAL:
            return route
    return greedy_route(Qs[-1], env)


fig = plt.figure(figsize=(12, 4.6))
gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], hspace=0.55)
ax = fig.add_subplot(gs[:, 0])
episodes = np.arange(1, N_CLIFF_EPISODES + 1)
for i, name in enumerate(cliff):
    curve = smooth(cliff[name][0].mean(axis=0), 10)
    ax.plot(episodes, curve, color=CAT[i], label=name)
    label_end(ax, N_CLIFF_EPISODES, curve[-1], f"{name}: {cliff[name][0][:, -100:].mean():.0f}")
ax.set_ylim(-100, -10)
ax.set_xlabel("episode")
ax.set_ylabel("sum of rewards during episode")
ax.set_title(f"Online reward (mean of {N_CLIFF_RUNS} runs, 10-episode moving average)")
ax.legend(loc="lower right")
env_c = GridWorld("cliff", **CLIFF_ENV)
for k, name in enumerate(cliff):
    axm = fig.add_subplot(gs[k, 1])
    draw_grid(axm, env_c, labels=False)
    axm.text(5.5, 3, "the cliff", ha="center", va="center", color="white", fontsize=9,
             weight="bold", zorder=3)
    axm.text(0, 3.28, "S", ha="center", va="center", color=INK["primary"], fontsize=8, zorder=5)
    axm.text(11, 3.28, "G", ha="center", va="center", color="white", fontsize=8,
             weight="bold", zorder=5)
    route = [env_c.rc(s) for s in typical_route(cliff[name][1], env_c)]
    axm.plot([c for r, c in route], [r for r, c in route], color=CAT[k], lw=2.6, zorder=4)
    axm.set_title(f"{name}: greedy route after training", fontsize=10)
fig.suptitle("Figure 2 — Sanity check: Cliff Walking (Sutton & Barto, Example 6.6)",
             x=0.01, ha="left")
fig.subplots_adjust(left=0.06, right=0.98, top=0.84, bottom=0.12, wspace=0.18)
save(fig, "fig02_cliff_walking_check.png")
for name in cliff:
    print(f"{name:>10}: mean reward per episode over the last 100 episodes = "
          f"{cliff[name][0][:, -100:].mean():.1f}")

# %% [markdown]
# ### 1b. Q-learning on the 5×5 map — learning curves
#
# 30 independent runs of 300 episodes each, learning only from the environment reward. Shaded bands are 95 % confidence intervals of the mean.

# %%
P1_EPISODES, P1_SEEDS = (150, 12) if FAST else (300, 30)
SNAP_EPISODES = (0, 5, 20, 50, P1_EPISODES)
part1 = []
for seed in range(P1_SEEDS):
    env = GridWorld("A")
    agent = QLearningAgent(env.n_states, alpha=0.5, gamma=0.95, epsilon=0.1, seed=seed)
    part1.append(run_episodes(env, agent, P1_EPISODES, oracle=oracle_A,
                              snapshot_at=SNAP_EPISODES if seed == 0 else ()))
L1 = np.array([r["length"] for r in part1])
R1 = np.array([r["return"] for r in part1])
A1 = np.array([r["accuracy"] for r in part1])
P1 = np.array([r["path_optimal"] for r in part1])
ep = np.arange(1, P1_EPISODES + 1)

fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
m, h = mean_ci(L1)
axes[0].fill_between(ep, m - h, m + h, color=CAT[0], alpha=0.18, lw=0)
axes[0].plot(ep, m, color=CAT[0], label="mean of runs (95% CI)")
axes[0].axhline(8, color=INK["muted"], ls="--", lw=1.2, label="shortest path (8 steps)")
axes[0].set_ylim(0, 52)
axes[0].set_ylabel("steps in the episode")
axes[0].set_title("Episode length")
axes[0].legend(loc="upper right")
m, h = mean_ci(R1)
axes[1].fill_between(ep, m - h, m + h, color=CAT[0], alpha=0.18, lw=0)
axes[1].plot(ep, m, color=CAT[0], label="mean of runs (95% CI)")
axes[1].axhline(0.93, color=INK["muted"], ls="--", lw=1.2, label="best possible (0.93)")
axes[1].set_ylabel("return (sum of environment rewards)")
axes[1].set_title("Return per episode")
axes[1].legend(loc="lower right")
m, h = mean_ci(A1)
axes[2].fill_between(ep, m - h, m + h, color=CAT[0], alpha=0.18, lw=0)
axes[2].plot(ep, m, color=CAT[0], label="policy accuracy (all 17 cells)")
axes[2].plot(ep, P1.mean(axis=0), color=CAT[1], label="share of runs that have learned")
axes[2].set_ylim(-0.02, 1.02)
axes[2].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
axes[2].set_ylabel("share")
axes[2].set_title("What has been learned")
axes[2].legend(loc="lower right")
for ax in axes:
    ax.set_xlabel("episode")
fig.suptitle(f"Figure 3 — Q-learning from the environment reward alone ({P1_SEEDS} runs)",
             x=0.01, ha="left")
fig.tight_layout()
label_ends(axes[2], [(P1_EPISODES, m[-1], "accuracy"),
                     (P1_EPISODES, P1.mean(axis=0)[-1], "learned")])
save(fig, "fig03_qlearning_learning_curves.png")

first_ep, first_steps = [], []
for r in part1:
    idx = np.flatnonzero(r["path_optimal"])
    if idx.size:
        first_ep.append(idx[0] + 1)
        first_steps.append(int(r["length"][: idx[0] + 1].sum()))
pd.DataFrame({"episode": ep, "length_mean": L1.mean(0), "return_mean": R1.mean(0),
              "accuracy_mean": A1.mean(0), "share_learned": P1.mean(0)}).to_csv(
    RES / "part1_learning_curves.csv", index=False)
display(Markdown(
    f"**Part 1 result.** Learning from the environment reward alone, the agent first follows a "
    f"shortest path after a median of **{np.median(first_ep):.0f} episodes** "
    f"(≈ **{np.median(first_steps):.0f} environment steps**; {len(first_ep)}/{P1_SEEDS} runs got there). "
    f"After {P1_EPISODES} episodes the greedy action is a shortest-path move in "
    f"**{A1[:, -1].mean():.0%}** of cells on average; cells the agent rarely visits "
    f"(the top-left detour) are learned last."))

# %% [markdown]
# ### 1c. How the policy develops (one run)
#
# Arrows show the greedy action in each cell (**red** = not a shortest-path move, grey dot = actions still tied); the shading is the state value $V(s)=\max_a Q(s,a)$.

# %%
snaps = part1[0]["snapshots"]
fig, axes = plt.subplots(1, len(SNAP_EPISODES), figsize=(3.0 * len(SNAP_EPISODES), 3.5))
values = {k: snaps[k].max(axis=1).astype(float) for k in SNAP_EPISODES}
not_free = env_A.cell_type != FREE
vmin = min(v[~not_free].min() for v in values.values())
vmax = max(v[~not_free].max() for v in values.values())
cmap_v = seq_cmap(BLUE_SEQ[:5], "v")
for ax, k in zip(axes, SNAP_EPISODES):
    v = values[k].copy()
    v[not_free] = np.nan
    draw_grid(ax, env_A, values=v, cmap=cmap_v, vmin=vmin, vmax=vmax, fontsize=7)
    draw_policy(ax, snaps[k], env_A, oracle_A, color=INK["primary"], mark_wrong=True)
    ax.set_title(f"after {k} episodes\npolicy accuracy {policy_accuracy(snaps[k], oracle_A):.0%}",
                 fontsize=10, loc="center")
sm = plt.cm.ScalarMappable(norm=Normalize(vmin, vmax), cmap=cmap_v)
cb = fig.colorbar(sm, ax=list(axes), shrink=0.8, pad=0.01)
cb.set_label("V(s) = max_a Q(s, a)", color=INK["secondary"])
cb.outline.set_visible(False)
fig.suptitle("Figure 4 — The greedy policy and state values during learning (run 1)",
             x=0.01, ha="left", y=1.04)
save(fig, "fig04_policy_snapshots.png")

# %% [markdown]
# ### 1d. The learned Q-table
#
# One panel per action: the learned value of taking that action in each cell (end of run 1).

# %%
Q_final = snaps[P1_EPISODES]
free_states = oracle_A.states
qmin, qmax = Q_final[free_states].min(), Q_final[free_states].max()
cmap_q = seq_cmap(BLUE_SEQ, "q")
fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
for a, ax in enumerate(axes):
    vals = np.full(env_A.n_states, np.nan)
    vals[free_states] = Q_final[free_states, a]
    draw_grid(ax, env_A, values=vals, cmap=cmap_q, vmin=qmin, vmax=qmax, fontsize=7)
    for s in free_states:
        r, c = env_A.rc(s)
        shade = (Q_final[s, a] - qmin) / (qmax - qmin + 1e-12)
        best = oracle_A.optimal[s, a]
        ax.text(c, r, f"{Q_final[s, a]:.2f}", ha="center", va="center", fontsize=7.5,
                color="white" if shade > 0.6 else INK["primary"],
                weight="bold" if best else "normal", zorder=3)
    ax.set_title(f"Q(s, {ACTION_NAMES[a]})", loc="center")
sm = plt.cm.ScalarMappable(norm=Normalize(qmin, qmax), cmap=cmap_q)
cb = fig.colorbar(sm, ax=list(axes), shrink=0.8, pad=0.01)
cb.outline.set_visible(False)
fig.suptitle("Figure 5 — Learned action values (bold = a shortest-path move from that cell)",
             x=0.01, ha="left", y=1.02)
save(fig, "fig05_q_values.png")

# %% [markdown]
# ## Part 2 — A simulated trainer who is sometimes wrong and sometimes silent
#
# Real people are imperfect trainers: they press the wrong key, misjudge a move, or simply say nothing. Following the simulated-teacher method of Griffith et al. (2013) — whose *Advise* model describes feedback by its **consistency** $C$ and its **likelihood** $L$ — and the "mistake" teacher of the B-Pref benchmark (Lee et al., 2021), the simulated trainer here
#
# * decides at each move whether to speak at all: it gives feedback with probability **p** and otherwise stays **silent** ($h = 0$);
# * when it speaks, it says **+1** for a correct move (one that shortens the path to the goal) and **−1** otherwise — but with probability **e** it gives the *wrong* sign.
#
# So $C = 1 - e$ and $L = p$. The agent simply treats feedback as extra reward (*reward shaping* with human feedback; cf. Knox & Stone, 2010):
#
# $$r_{\text{total}} = r_{\text{env}} + \beta\,h,\qquad h \in \{+1, -1, 0\},\quad \beta = 1 .$$
#
# This is the simplest way to use feedback and it has known weaknesses — e.g. feedback treated as reward can create positive reward cycles (MacGlashan et al., 2017) — which is part of why errors matter. Every condition uses the same 50 random seeds. **Time to learn** = environment steps until the greedy path from the start is optimal; runs that never get there within 4 000 steps are counted as 4 000 (shown as *did not learn*).

# %% [core:trainer]

# %%
# Check: does the simulated trainer speak and err at the rates we asked for?
check = SimulatedTrainer(oracle_A, error_rate=0.3, feedback_prob=0.5, seed=0)
rng = np.random.default_rng(0)
calls = [(int(rng.choice(oracle_A.states)), int(rng.integers(N_ACTIONS))) for _ in range(100_000)]
h = np.array([check(s, a) for s, a in calls])
print(f"asked p = 0.50 -> spoke on {np.mean(h != 0):.3f} of steps; "
      f"asked e = 0.30 -> wrong in {1 - check.n_correct / check.n_given:.3f} of its feedback")

# %% [markdown]
# ### 2a. The baseline: plain Q-learning, and its sensitivity to α and ε
#
# Before adding a trainer: how many environment steps does the agent need on its own (median of 20 runs per cell)? The box marks the setting used everywhere else (α = 0.5, ε = 0.1).

# %%
ALPHAS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
EPSILONS = [0.01, 0.05, 0.1, 0.2, 0.3]
N_SENS = 6 if FAST else 20
sens = np.zeros((len(EPSILONS), len(ALPHAS)))
for i, eps in enumerate(EPSILONS):
    for j, al in enumerate(ALPHAS):
        runs = run_many(N_SENS, feedback_prob=0.0, alpha=al, epsilon=eps, max_steps=8000,
                        record_every=0, stop_when_learned=True)
        sens[i, j] = np.median(censored_steps(runs, 8000))
pd.DataFrame(sens, index=[f"eps={e}" for e in EPSILONS],
             columns=[f"alpha={a}" for a in ALPHAS]).to_csv(RES / "part1_sensitivity.csv")

fig, ax = plt.subplots(figsize=(8.6, 3.9))
norm = matplotlib.colors.LogNorm(vmin=sens.min(), vmax=sens.max())
cmap_s = seq_cmap(BLUE_SEQ, "s")
ax.imshow(sens, cmap=cmap_s, norm=norm, aspect="auto")
for i in range(len(EPSILONS)):
    for j in range(len(ALPHAS)):
        dark = norm(sens[i, j]) > 0.6
        ax.text(j, i, f"{sens[i, j]:.0f}", ha="center", va="center", fontsize=9,
                color="white" if dark else INK["primary"])
j0, i0 = ALPHAS.index(0.5), EPSILONS.index(0.1)
ax.add_patch(Rectangle((j0 - 0.5, i0 - 0.5), 1, 1, fill=False, edgecolor=MOVE_COLOR, lw=2.5))
ax.set_xticks(range(len(ALPHAS)))
ax.set_xticklabels([str(a) for a in ALPHAS])
ax.set_yticks(range(len(EPSILONS)))
ax.set_yticklabels([str(e) for e in EPSILONS])
ax.set_xlabel("step size α")
ax.set_ylabel("exploration ε")
ax.grid(False)
ax.set_title("Median steps until learned (darker = slower)")
fig.suptitle("Figure 6 — Sensitivity of plain Q-learning to α and ε", x=0.01, ha="left")
fig.tight_layout()
save(fig, "fig06_sensitivity_alpha_epsilon.png")

# %% [markdown]
# ### 2b. Learning curves with a noisy trainer
#
# Left: feedback on every step. Right: the trainer is silent half the time. The dashed grey line is the agent learning from the environment alone. **Top row:** the 8 cells of the shortest route — the part of the policy that the *learned* criterion (and the task) depends on. **Bottom row:** all 17 cells.

# %%
CURVE_STEPS = 1500
CURVE_CONDS = [(0.0, 0.0)] + [(e, p) for p in (1.0, 0.5) for e in (0.0, 0.1, 0.3)]
curves = {(e, p): run_many(N_SEEDS, error_rate=e, feedback_prob=p, max_steps=CURVE_STEPS,
                           record_every=10) for e, p in CURVE_CONDS}
t_axis = np.arange(10, CURVE_STEPS + 1, 10)


def cond_label(e, p):
    return "no trainer" if p == 0 else f"{e:.0%} wrong"


fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.6), sharex=True, sharey=True)
end_labels = {}
for row, (key, what) in enumerate((("route_curve", "cells on the shortest route"),
                                   ("accuracy_curve", "all 17 cells"))):
    for col, p in enumerate((1.0, 0.5)):
        ax = axes[row, col]
        end_labels[ax] = []
        m, hw = mean_ci(np.array([r[key] for r in curves[(0.0, 0.0)]]))
        ax.plot(t_axis, m, color=INK["muted"], ls="--", lw=1.6, label="no trainer")
        end_labels[ax].append((CURVE_STEPS, m[-1], "no trainer"))
        for e in (0.0, 0.1, 0.3):
            m, hw = mean_ci(np.array([r[key] for r in curves[(e, p)]]))
            ax.fill_between(t_axis, m - hw, m + hw, color=err_color(e), alpha=0.15, lw=0)
            ax.plot(t_axis, m, color=err_color(e), label=cond_label(e, p))
            end_labels[ax].append((CURVE_STEPS, m[-1], cond_label(e, p)))
        ax.set_xscale("log")
        ax.set_xlim(10, CURVE_STEPS)
        ax.set_ylim(-0.02, 1.04)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        feedback = "feedback on every step" if p == 1 else "trainer silent half the time"
        ax.set_title(f"{feedback} — {what}")
        if row == 1:
            ax.set_xlabel("environment steps (log scale)")
        if col == 0:
            ax.set_ylabel(f"policy accuracy, {what}\n(mean, 95% CI)")
axes[0, 0].legend(loc="upper left")
fig.suptitle("Figure 7 — Feedback speeds up learning the route; wrong feedback and silence "
             "slow it down", x=0.01, ha="left")
fig.tight_layout()
for ax in axes[:, 1]:                     # left column: the legend names the lines
    label_ends(ax, end_labels[ax])
save(fig, "fig07_trainer_learning_curves.png")
late = {k: np.mean([r["accuracy_curve"][-1] for r in v]) for k, v in curves.items()}
print("policy accuracy over all 17 cells after "
      f"{CURVE_STEPS} steps: no trainer {late[(0.0, 0.0)]:.0%}; perfect trainer "
      f"{late[(0.0, 1.0)]:.0%}; 10% wrong {late[(0.1, 1.0)]:.0%}; 30% wrong {late[(0.3, 1.0)]:.0%}")

# %% [markdown]
# **Top row:** with wrong feedback the route is not only learned later, it also does not *stay* learned — each wrong signal can flip an action back (with α = 0.5 the agent reacts strongly to single signals), so the curves sag after their peak.
#
# **A side effect worth noticing (bottom row).** Over the whole map, the agent without a trainer ends up knowing *more* cells than the trained agents. Once the trainer has rewarded the route, the agent keeps to it and rarely explores cells it does not need, so their greedy actions stay unlearned or wrong. This matters for the pilot: the policy-arrow display will show some wrong arrows in cells the agent hardly visits, and a person cannot fix them — feedback only reaches moves the agent actually makes.

# %% [markdown]
# ### 2c. When has the agent learned?
#
# The share of runs whose greedy path is already optimal at each step (a curve further right = slower learning).

# %%
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3), sharey=True)
for ax, p in zip(axes, (1.0, 0.5)):
    for (e, pp), style in [((0.0, 0.0), "base")] + [((e, p), "t") for e in (0.0, 0.1, 0.3)]:
        x = censored_steps(curves[(e, pp)], CURVE_STEPS * 10)
        grid = np.unique(np.concatenate([[1], np.sort(x), [CURVE_STEPS]]))
        share = [(x <= g).mean() for g in grid]
        color = INK["muted"] if style == "base" else err_color(e)
        median = np.median(x)
        med_txt = f"median {median:.0f}" if median <= CURVE_STEPS else f"median > {CURVE_STEPS}"
        ax.step(grid, share, where="post", color=color, ls="--" if style == "base" else "-",
                lw=1.8, label=f"{cond_label(e, pp)} ({med_txt})")
        if median <= CURVE_STEPS:
            ax.scatter([median], [0.5], s=36, color=color, edgecolor=INK["surface"], lw=1.2,
                       zorder=4)
    ax.axhline(0.5, color=INK["axis"], lw=0.9, ls=":")
    ax.set_xscale("log")
    ax.set_xlim(10, CURVE_STEPS)
    ax.set_ylim(-0.02, 1.04)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("environment steps (log scale)")
    ax.set_title("feedback on every step" if p == 1 else "trainer silent half the time")
    ax.legend(loc="upper left", title="steps to learn", title_fontsize=9)
axes[0].set_ylabel("share of runs that have learned")
fig.suptitle("Figure 8 — Time until the agent has learned the task (dot = median run)",
             x=0.01, ha="left")
fig.tight_layout()
save(fig, "fig08_time_to_learn_cdf.png")

# %% [markdown]
# ### 2d. The full picture: error rate × silence
#
# Every combination of 9 error rates and 5 feedback rates, 50 runs each (≈ 2 250 runs).

# %%
ERRORS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
PROBS = [1.0, 0.75, 0.5, 0.25, 0.1]
t0 = time.time()
sw = sweep(ERRORS, PROBS, N_SEEDS, max_steps=MAX_STEPS)
base = sweep([0.0], [0.0], N_SEEDS, max_steps=MAX_STEPS)      # learning from the environment alone
sw.to_csv(RES / "part2_sweep.csv", index=False)
base.to_csv(RES / "part2_no_trainer.csv", index=False)
BASE_MED = float(base["steps_to_learn_c"].median())
summary = (sw.groupby(["feedback_prob", "error_rate"])
             .agg(median=("steps_to_learn_c", "median"),
                  q25=("steps_to_learn_c", lambda x: x.quantile(0.25)),
                  q75=("steps_to_learn_c", lambda x: x.quantile(0.75)),
                  learned=("learned", "mean"),
                  feedback=("feedback_to_learn", "mean"),
                  correct=("correct_to_learn", "mean"))
             .reset_index())
summary["speedup_vs_no_trainer"] = BASE_MED / summary["median"]
summary.to_csv(RES / "part2_summary.csv", index=False)
print(f"{len(sw) + len(base)} runs in {time.time() - t0:.0f} s; "
      f"no trainer: median {BASE_MED:.0f} steps to learn")
display(summary.pivot(index="feedback_prob", columns="error_rate", values="median")
        .sort_index(ascending=False).round(0))

# %%
med_grid = (summary.pivot(index="feedback_prob", columns="error_rate", values="median")
            .sort_index(ascending=False))
learn_grid = (summary.pivot(index="feedback_prob", columns="error_rate", values="learned")
              .sort_index(ascending=False))
speed = np.log2(BASE_MED / med_grid.values)
fig, ax = plt.subplots(figsize=(10.5, 4.4))
lim = np.nanmax(np.abs(speed))
norm = matplotlib.colors.TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
im = ax.imshow(speed, cmap=div_cmap(), norm=norm, aspect="auto")
for i in range(speed.shape[0]):
    for j in range(speed.shape[1]):
        if learn_grid.values[i, j] < 0.5:
            txt = "did not\nlearn"
        else:
            txt = f"{2 ** speed[i, j]:.1f}×"
        ax.text(j, i, txt, ha="center", va="center", fontsize=8.5,
                color="white" if abs(norm(speed[i, j]) - 0.5) > 0.33 else INK["primary"])
ax.set_xticks(range(len(ERRORS)))
ax.set_xticklabels([f"{e:.0%}" for e in ERRORS])
ax.set_yticks(range(len(med_grid.index)))
ax.set_yticklabels([f"{p:.0%}" for p in med_grid.index])
ax.set_xlabel("share of feedback that is wrong (e)")
ax.set_ylabel("share of steps with feedback (p)")
ax.grid(False)
cb = fig.colorbar(im, ax=ax, pad=0.01)
cb.set_label("speed vs. no trainer (log scale)", color=INK["secondary"])
ticks = [t for t in (-3, -2, -1, 0, 1, 2, 3, 4) if -lim <= t <= lim]
cb.set_ticks(ticks)
cb.set_ticklabels([f"{2 ** t:g}×" for t in ticks])
cb.outline.set_visible(False)
ax.set_title(f"How many times faster than learning alone ({BASE_MED:.0f} steps)? "
             "Blue = faster, red = slower")
fig.suptitle("Figure 9 — Speed-up from feedback, for every error rate and amount of silence",
             x=0.01, ha="left")
fig.tight_layout()
save(fig, "fig09_speedup_heatmap.png")

# %%
every_step = summary[summary["feedback_prob"] == 1.0].sort_values("error_rate")
no_gain = every_step[every_step["median"] >= BASE_MED]["error_rate"]
CROSSOVER = float(no_gain.iloc[0]) if len(no_gain) else float("nan")

fig, ax = plt.subplots(figsize=(10, 5.0))
p_colors = dict(zip([0.1, 0.25, 0.5, 1.0], RAMP4))
ends = []
for p in [1.0, 0.5, 0.25, 0.1]:
    g = summary[summary["feedback_prob"] == p].sort_values("error_rate")
    if p == 1.0:
        ax.fill_between(g["error_rate"], g["q25"], g["q75"], color=p_colors[p], alpha=0.12,
                        lw=0, label="middle 50% of runs (100% feedback)")
    ax.plot(g["error_rate"], g["median"], marker="o", ms=5, color=p_colors[p],
            label=f"feedback on {p:.0%} of steps")
    ends.append((g["error_rate"].iloc[-1], g["median"].iloc[-1], f"{p:.0%} of steps"))
ax.axhline(BASE_MED, color=INK["muted"], ls="--", lw=1.4)
ax.annotate(f"no trainer: {BASE_MED:.0f} steps", xy=(0.0, BASE_MED), xytext=(2, 6),
            textcoords="offset points", fontsize=8.5, color=INK["secondary"])
ax.axhline(MAX_STEPS, color=INK["axis"], ls=":", lw=1.2)
ax.annotate(f"step budget ({MAX_STEPS} steps) = did not learn", xy=(0.3, MAX_STEPS),
            xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8.5,
            color=INK["secondary"])
ax.axvline(0.5, color=INK["axis"], ls=":", lw=1.2)
ax.annotate("coin flip:\nfeedback carries\nno information", xy=(0.5, 40), xytext=(-6, 0),
            textcoords="offset points", ha="right", fontsize=8.5, color=INK["secondary"])
ax.set_yscale("log")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_xticks(ERRORS)
ax.set_xticklabels([f"{e:.0%}" for e in ERRORS])
ax.set_xlim(-0.01, 0.56)
ax.set_ylim(top=MAX_STEPS * 1.8)
ax.set_xlabel("share of feedback that is wrong (e)")
ax.set_ylabel("steps until the agent has learned (median, log scale)")
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93))
ax.set_title(f"With feedback on every step, the benefit is gone at about {CROSSOVER:.0%} wrong")
fig.suptitle("Figure 10 — How much slower the agent learns as feedback gets worse",
             x=0.01, ha="left")
fig.tight_layout()
label_ends(ax, ends)
save(fig, "fig10_slowdown_vs_error.png")

# %% [markdown]
# ### 2e. How much feedback is needed — and how much of it is wrong
#
# Mean number of key presses the simulated trainer gave before the agent learned, split into correct and wrong ones.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1), sharey=True)
show_errors = [0.0, 0.1, 0.2, 0.3]
for ax, p in zip(axes, (1.0, 0.5)):
    g = summary[(summary["feedback_prob"] == p) & summary["error_rate"].isin(show_errors)]
    g = g.sort_values("error_rate")
    x = np.arange(len(g))
    correct = g["correct"].values
    wrong = g["feedback"].values - correct
    ax.bar(x, correct, width=0.62, color=STATUS["good"], edgecolor=INK["surface"], lw=1.5,
           label="correct feedback")
    ax.bar(x, wrong, width=0.62, bottom=correct, color=STATUS["critical"],
           edgecolor=INK["surface"], lw=1.5, label="wrong feedback")
    for xi, c_, w_ in zip(x, correct, wrong):
        ax.text(xi, c_ + w_, f"{c_ + w_:.0f}", ha="center", va="bottom", fontsize=8.5,
                color=INK["secondary"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{e:.0%} wrong" for e in g["error_rate"]])
    ax.set_title("feedback on every step" if p == 1 else "trainer silent half the time")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
axes[0].set_ylabel("feedback given until learned (mean)")
fig.suptitle("Figure 11 — With more errors the trainer must give far more feedback",
             x=0.01, ha="left")
fig.tight_layout()
save(fig, "fig11_feedback_needed.png")

# %% [markdown]
# ### 2f. Does a more cautious learner tolerate errors better?
#
# Same experiment (trainer silent half the time) for four step sizes α. The y-axis is the speed-up over the same agent learning alone; below 1× the feedback hurts.

# %%
ALPHA_SET = [0.1, 0.25, 0.5, 0.9]
ERR_SET = [0.0, 0.1, 0.2, 0.3, 0.4]
N_ALPHA = 8 if FAST else 30
rows = []
for al in ALPHA_SET:
    alone = np.median(censored_steps(run_many(N_ALPHA, feedback_prob=0.0, alpha=al,
                                              max_steps=MAX_STEPS, record_every=0,
                                              stop_when_learned=True), MAX_STEPS))
    for e in ERR_SET:
        runs = run_many(N_ALPHA, error_rate=e, feedback_prob=0.5, alpha=al, max_steps=MAX_STEPS,
                        record_every=0, stop_when_learned=True)
        med_steps = np.median(censored_steps(runs, MAX_STEPS))
        rows.append({"alpha": al, "error_rate": e, "median": med_steps, "alone": alone,
                     "speedup": alone / med_steps})
alpha_df = pd.DataFrame(rows)
alpha_df.to_csv(RES / "part2_alpha_sensitivity.csv", index=False)

fig, ax = plt.subplots(figsize=(9.5, 4.6))
for k, al in enumerate(ALPHA_SET):
    g = alpha_df[alpha_df["alpha"] == al]
    ax.plot(g["error_rate"], g["speedup"], marker="o", ms=5, color=RAMP4[k], label=f"α = {al}")
    label_end(ax, g["error_rate"].iloc[-1], g["speedup"].iloc[-1],
              f"α = {al} (alone: {g['alone'].iloc[0]:.0f} steps)")
ax.axhline(1.0, color=INK["muted"], ls="--", lw=1.3)
ax.annotate("no benefit", xy=(0.0, 1.0), xytext=(2, -11), textcoords="offset points",
            fontsize=8.5, color=INK["secondary"])
ax.set_yscale("log", base=2)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}×"))
ax.set_xticks(ERR_SET)
ax.set_xticklabels([f"{e:.0%}" for e in ERR_SET])
ax.set_xlim(-0.02, 0.5)
ax.set_xlabel("share of feedback that is wrong (e); feedback on 50% of steps")
ax.set_ylabel("speed-up over learning alone (log scale)")
ax.legend(loc="lower left")
ax.set_title("A smaller step size averages out wrong feedback")
fig.suptitle("Figure 12 — How harmful errors are depends on the learner", x=0.01, ha="left")
fig.tight_layout()
save(fig, "fig12_alpha_vs_error.png")

# %%
med = summary.set_index(["feedback_prob", "error_rate"])["median"]
lrn = summary.set_index(["feedback_prob", "error_rate"])["learned"]


def m(e, p):
    return float(med.loc[(p, e)])


def faster_or_slower(x):
    return f"{BASE_MED / x:.1f}× faster than" if x < BASE_MED else f"{x / BASE_MED:.1f}× slower than"


def silence_effect(every, half):
    ratio = half / every
    if ratio < 0.9:
        return f"actually *speeds learning up* ({every:.0f} → {half:.0f} steps)"
    if ratio > 1.1:
        return f"slows it down further ({every:.0f} → {half:.0f} steps)"
    return f"makes little difference ({every:.0f} → {half:.0f} steps)"


a01 = alpha_df[(alpha_df["alpha"] == 0.1) & (alpha_df["error_rate"] == 0.3)]["speedup"].iloc[0]
a05 = alpha_df[(alpha_df["alpha"] == 0.5) & (alpha_df["error_rate"] == 0.3)]["speedup"].iloc[0]
KEY = {
    "no_trainer_median_steps": BASE_MED,
    "perfect_every_step": m(0.0, 1.0), "err10_every_step": m(0.1, 1.0),
    "err30_every_step": m(0.3, 1.0), "err50_every_step_learned_share": float(lrn.loc[(1.0, 0.5)]),
    "perfect_half": m(0.0, 0.5), "err10_half": m(0.1, 0.5), "err30_half": m(0.3, 0.5),
    "speedup_err30_half_alpha0.1": float(a01), "speedup_err30_half_alpha0.5": float(a05),
}
display(Markdown(f"""**Part 2 results** (median steps until learned, {N_SEEDS} runs per condition)

* **No trainer:** {BASE_MED:.0f} steps.
* **Perfect trainer, feedback on every step:** {m(0.0, 1.0):.0f} steps — **{BASE_MED / m(0.0, 1.0):.0f}× faster** than learning alone.
* **10 % wrong:** {m(0.1, 1.0):.0f} steps — {m(0.1, 1.0) / m(0.0, 1.0):.1f}× slower than a perfect trainer, still {faster_or_slower(m(0.1, 1.0))} no trainer.
* **30 % wrong:** {m(0.3, 1.0):.0f} steps — {faster_or_slower(m(0.3, 1.0))} no trainer: the benefit is essentially gone.
* **50 % wrong** (a coin flip): the agent learned in only {lrn.loc[(1.0, 0.5)]:.0%} of runs within {MAX_STEPS} steps — pure noise is worse than no feedback.
* **Silent half the time:** perfect {m(0.0, 0.5):.0f}, 10 % wrong {m(0.1, 0.5):.0f}, 30 % wrong {m(0.3, 0.5):.0f} steps. Silence slows a perfect trainer down {m(0.0, 0.5) / m(0.0, 1.0):.1f}×; with 30 % wrong feedback, silence {silence_effect(m(0.3, 1.0), m(0.3, 0.5))} — fewer signals also means fewer wrong ones.
* **The learner matters:** with 30 % wrong feedback (silent half the time) the speed-up over learning alone is {a01:.1f}× for α = 0.1 but {a05:.1f}× for α = 0.5 — a smaller step size averages wrong feedback out.

So for this agent the *quality* of feedback matters a great deal: with feedback on every step, a trainer who is right 90 % of the time gets the agent there **{m(0.3, 1.0) / m(0.1, 1.0):.1f}× faster** than one who is right 70 % of the time. That is the lever an explanation could pull."""))

# %% [markdown]
# ## Part 3 — Human pilot: does seeing the agent's plan change the feedback?
#
# **Design.** Within-subject, two displays, 3–4 participants (friends; no RL background needed):
#
# * **Agent only** — the map, the agent (blue) and an orange arrow for the move it has just made.
# * **Agent + policy arrows** — the same, plus a purple arrow in every cell showing the agent's current greedy action, i.e. its *plan* (faded = unsure, grey dot = no preference yet). This is the **explanation**. The arrows show what the agent intends, never whether that is right.
#
# Participants press **G** (good) or **B** (bad) after any move they want to judge, or say nothing. The agent is the same Q-learner as in Part 2 and learns from their key presses as extra reward. One move is shown every 0.75 s. A session ends when the agent has learned (greedy path from the start is optimal) or after **300 steps** (under 4 minutes).
#
# **Counterbalancing.** Odd-numbered participants see *agent only* first, even-numbered participants see *agent + arrows* first. Session 1 always uses map A and session 2 its mirror image (map B), so display and map are crossed. A 30-step practice round on an open map comes first.
#
# **Measures, fixed before any data is collected** (per session):
# 1. **Feedback needed** — key presses until the agent learned;
# 2. **Feedback correct** — share of key presses that agree with the shortest-path oracle;
# 3. **Learning speed** — agent steps until it learned (censored at 300).
#
# With 3–4 people this is a **pilot**: its job is to test the procedure, see the direction and rough size of differences and estimate their spread — not to test a hypothesis (Part 4 shows why).

# %%
alone = base["steps_to_learn_c"]
print(f"Without any feedback the agent learned within 300 steps in {np.mean(alone <= 300):.0%} of "
      f"{len(alone)} simulated runs (median {alone.median():.0f} steps), so learning within a "
      f"pilot session is due to the person's feedback.")

# %% [core:pilot]

# %% [markdown]
# **What participants see** (a mid-session frame). The app is `pilot/pilot_app.py`; it draws exactly this figure in a window and listens to the keyboard.

# %%
# A mid-session moment: 55 steps into a simulated session (same agent state in both panels).
demo = PilotSession("demo", 1, "explanation", "A", seed=0, max_steps=55)
run_session_headless(demo, SimulatedResponder(demo.oracle, feedback_prob=0.7, error_rate=0.1,
                                              seed=1))
last = demo.rows[-1]
s_prev = last["row"] * demo.env.W + last["col"]
s_now = last["next_row"] * demo.env.W + last["next_col"]
a_last = ACTION_NAMES.index(last["action"])
fig, axes = plt.subplots(1, 2, figsize=(10, 6.0))
for ax, cond in zip(axes, CONDITIONS):
    renderer = GridRenderer(ax, demo.env, show_policy=(cond == "explanation"), oracle=demo.oracle)
    renderer.update(demo.agent.Q, s_prev, a_last, s_now)
    ax.set_title(CONDITION_LABELS[cond], loc="center", pad=10)
    ax.set_xlabel("G = good     B = bad     (or say nothing)", color=INK["secondary"],
                  fontsize=10, labelpad=8)
fig.suptitle("Figure 13 — The two pilot displays", x=0.01, ha="left")
fig.text(0.5, 0.015, "Blue dot = agent (faint dot = where it was); orange arrow = the move to "
         "judge; purple arrows = the agent's current plan in each cell (faded = unsure, grey "
         "dot = no plan yet).", ha="center", fontsize=8.8, color=INK["secondary"], wrap=True)
fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.12, wspace=0.06)
save(fig, "fig13_pilot_interface.png")

# %% [core:analysis]

# %% [markdown]
# ### 3a. Pilot analysis (pre-specified)
#
# The cell below loads real logs from `pilot/data/`. **If there are none yet, it creates four simulated participants instead** — with the *same* error rate and feedback rate in both displays, i.e. no true effect — purely to check that logging, loading, tables and figures work end to end. Their figures are saved to `figures/pipeline_check/`, never to the README.

# %%
sessions, steps = load_pilot(PILOT_DATA)
REAL_PILOT = sessions is not None
if REAL_PILOT:
    BANNER, PILOT_FIGS = None, FIG
    print(f"Real pilot data: {sessions['participant'].nunique()} participants, "
          f"{len(sessions)} sessions.")
else:
    import shutil
    BANNER, PILOT_FIGS = "PIPELINE CHECK — simulated participants, not human data", FIG / "pipeline_check"
    shutil.rmtree(PIPELINE_CHECK, ignore_errors=True)
    rng = np.random.default_rng(2026)
    for k in range(1, 5):
        pid = f"P{k}"
        fb_prob, err = rng.uniform(0.35, 0.8), rng.uniform(0.05, 0.2)   # same in both displays
        for session_no, condition, layout in counterbalanced_order(pid):
            sess = PilotSession(pid, session_no, condition, layout, seed=100 * k + session_no)
            run_session_headless(sess, SimulatedResponder(sess.oracle, fb_prob, err,
                                                          seed=10 * k + session_no))
            sess.save(PIPELINE_CHECK, extra={"simulated": True})
    sessions, steps = load_pilot(PIPELINE_CHECK)
    print("No real pilot data in pilot/data/ yet -> analysis below runs on 4 SIMULATED "
          "participants (no true effect) to test the pipeline. These are NOT results.")
PILOT_FIGS.mkdir(parents=True, exist_ok=True)
display(pilot_table(sessions))
display(pilot_medians(sessions).round(2))

# %%
fig = plot_pilot_paired(sessions, banner=BANNER)
save(fig, "pilot_paired.png", folder=PILOT_FIGS)
fig = plot_pilot_curves(steps, banner=BANNER)
save(fig, "pilot_curves.png", folder=PILOT_FIGS)

# %%
fig = plot_error_map(steps, banner=BANNER)
save(fig, "pilot_error_map.png", folder=PILOT_FIGS)
fig = plot_observed_vs_predicted(sessions, sw, banner=BANNER)
save(fig, "pilot_observed_vs_predicted.png", folder=PILOT_FIGS)
if not REAL_PILOT:
    med_sim = pilot_medians(sessions)
    a_row, e_row = med_sim.loc["Agent only"], med_sim.loc["Agent + policy arrows"]
    display(Markdown(
        "**A lesson from the pipeline check.** These four simulated people have *exactly the "
        "same* error rate and feedback rate in both displays, yet the medians still differ: "
        f"{a_row['feedback_to_learn']:.0f} vs {e_row['feedback_to_learn']:.0f} key presses and "
        f"{a_row['steps_to_learn_c']:.0f} vs {e_row['steps_to_learn_c']:.0f} agent steps "
        "(agent only vs. arrows). That is chance alone. With four real people, differences "
        "of this size can also be chance — which is why the pilot is reported descriptively "
        "and why Part 4 sizes a real study."))

# %% [markdown]
# **How to read these once real data exists.** The first figure answers the three questions directly (feedback needed, feedback correct, learning speed) for each person in both displays. The error map shows *where* people misjudge moves — e.g. whether the arrows reduce mistakes on the detour cells. The last figure asks whether Part 2's simulation, given only a person's accuracy and feedback rate, predicts how fast their agent learned; if it does, accuracy and amount of feedback are enough to explain the effect of the display.

# %% [markdown]
# ## Part 4 — Next step: how many people would a real study need?
#
# A paired test with 4 people cannot reach the usual 5 % significance level: even if all four improve, a sign-flip (permutation) test gives p = 2/16 = 0.125. To plan a real study, we simulate one: each person has their own error rate (about 20 % without explanations, i.e. $C = 0.8$; person-to-person spread 7 points), explanations lower it by Δ, the person trains once in each display (feedback on half the steps), and each session's learning time is drawn from the Part 2 simulations at that error rate. **This rests on assumptions — the pilot's job is to replace them with measured values.**

# %% [core:power]

# %%
print("smallest possible two-sided p-value with 4 participants:", sign_flip_pvalue([1, 1, 1, 1]))
half = sw[sw["feedback_prob"] == 0.5]
outcomes = [half.loc[np.isclose(half["error_rate"], e), "steps_to_learn_c"].values for e in ERRORS]
NS = [4, 8, 12, 16, 24, 32, 48]
DELTAS = (0.05, 0.10, 0.15)
power = power_by_simulation(outcomes, ERRORS, base_error=0.20, deltas=DELTAS, sd_between=0.07,
                            ns=NS, reps=60 if FAST else 300, seed=1)
power_df = pd.DataFrame({f"delta={d}": power[d] for d in DELTAS}, index=NS)
power_df.index.name = "participants"
power_df.to_csv(RES / "part4_power.csv")

fig, ax = plt.subplots(figsize=(9, 4.6))
ends = []
for k, d in zip((0, 2, 3), DELTAS):
    ax.plot(NS, power[d], marker="o", ms=5, color=RAMP4[k],
            label=f"explanations cut errors by {d * 100:.0f} points")
    ends.append((NS[-1], power[d][-1], f"−{d * 100:.0f} points"))
ax.axhline(0.8, color=INK["muted"], ls="--", lw=1.3)
ax.annotate("80% power", xy=(NS[-1], 0.8), xytext=(0, -12), textcoords="offset points",
            ha="right", fontsize=8.5, color=INK["secondary"])
ax.axvline(4, color=INK["axis"], ls=":", lw=1.3)
ax.annotate("this pilot\n(n = 4)", xy=(4, 0.5), xytext=(5, 0), textcoords="offset points",
            fontsize=8.5, color=INK["secondary"])
ax.set_xticks(NS)
ax.set_ylim(-0.02, 1.04)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set_xlabel("participants (each uses both displays)")
ax.set_ylabel("chance of detecting the effect (p < 0.05)")
ax.legend(loc="lower right")
ax.set_title("Assumed: 20% errors without explanations, feedback on half the steps")
fig.suptitle("Figure 14 — How many participants a real study would need", x=0.01, ha="left")
fig.tight_layout()
label_ends(ax, ends)
save(fig, "fig14_power_analysis.png")


def n_needed(curve):
    hit = [n for n, pw in zip(NS, curve) if pw >= 0.8]
    return f"{hit[0]}" if hit else f"> {NS[-1]}"


display(power_df.round(2))
display(Markdown("**Participants needed for 80 % power** (under the assumptions above): " +
                 ", ".join(f"Δ = {d * 100:.0f} points → **{n_needed(power[d])}**" for d in DELTAS)))
KEY.update({f"n_for_80pct_power_delta_{d}": n_needed(power[d]) for d in DELTAS})

# %% [markdown]
# ## Animation — watching the plan form
#
# One simulated session in the explanation display (feedback on 60 % of steps, 10 % wrong). The purple arrows are the explanation a participant would see; GOOD/BAD shows the feedback given on each move.

# %%
from matplotlib.animation import FuncAnimation, PillowWriter

anim = PilotSession("anim", 1, "explanation", "A", seed=11, max_steps=300)
responder = SimulatedResponder(anim.oracle, feedback_prob=0.6, error_rate=0.1, seed=5)
frames, clock = [], 0.0
anim.start(now=clock)
while not anim.finished:
    p = anim.pending
    h, rt = responder.respond(p)
    if h and rt < 800:
        anim.give_feedback(h, now=clock + rt / 1000)
    else:
        h = 0
    frames.append((anim.agent.Q.copy(), p["s"], p["a"], p["s2"], p["step"], h))
    clock += 0.8
    anim.advance(now=clock)
keep = frames[::2] + [frames[-1]] * 8          # every 2nd step, then hold the last frame
fig, ax = plt.subplots(figsize=(4.4, 4.9), dpi=72)
renderer = GridRenderer(ax, anim.env, show_policy=True, oracle=anim.oracle)
title = ax.set_title("", loc="left", fontsize=10)
fb_text = ax.text(4.45, -0.6, "", ha="right", va="bottom", fontsize=11, weight="bold")


def draw_frame(i):
    Q, s, a, s2, step, h = keep[i]
    renderer.update(Q, s, a, s2)
    done = anim.learned_at is not None and step >= anim.learned_at
    title.set_text(f"step {step} · " + ("route learned" if done else
                   f"route cells right: {route_accuracy(Q, anim.oracle):.0%}"))
    fb_text.set_text({1: "GOOD", -1: "BAD", 0: ""}[h])
    fb_text.set_color(STATUS["good_text"] if h > 0 else STATUS["critical"])
    return []


animation = FuncAnimation(fig, draw_frame, frames=len(keep), interval=140, blit=False)
animation.save(FIG / "policy_learning.gif", writer=PillowWriter(fps=7), dpi=72)
plt.close(fig)
print(f"{len(frames)} steps; learned at step {anim.learned_at}; "
      f"GIF size {(FIG / 'policy_learning.gif').stat().st_size / 1e3:.0f} kB")
display(Image(filename=str(FIG / "policy_learning.gif")))

# %%
(RES / "key_numbers.json").write_text(json.dumps(KEY, indent=2))
print(json.dumps(KEY, indent=2))
print(f"Notebook finished in {(time.time() - T_START) / 60:.1f} minutes.")

# %% [markdown]
# ## Summary, limitations and next steps
#
# **What this first test shows.** (1) The Q-learning implementation reproduces Sutton & Barto's Cliff Walking result and learns the 5×5 task from reward alone. (2) Good/bad feedback can make it learn many times faster — but only if the feedback is mostly right: the speed-up shrinks quickly as errors rise and is gone at around 30 % wrong feedback for this learner (Part 2). (3) So if an explanation makes people's feedback even 10 points more accurate, the simulation predicts a large change in learning speed — which is what the human pilot is built to probe.
#
# **Limitations.**
# * **The human pilot is tiny** (3–4 friends, not naive users) and, at the time of writing, still to be run; with n = 4 no difference can be statistically significant (Part 4). Results will be descriptive only.
# * **One toy task.** A 5×5 deterministic gridworld; people can see the whole map, which makes judging moves easier than in most real tasks.
# * **"Correct" is defined by a shortest-path oracle.** A person may reasonably judge a move differently (e.g. a safe detour), which the oracle counts as wrong.
# * **One simple learner.** Feedback is used as shaping reward with a fixed weight; how harmful errors are depends strongly on the learner (Figure 12). Learners built for noisy or silent feedback (Advise, SABL, COACH) might change the picture.
# * **The simulated trainer is simplistic.** Its errors are independent and its silence random. Real people taper their feedback over time, use silence strategically (Loftin et al., 2016) and judge relative to the agent's current behaviour (MacGlashan et al., 2017).
# * **Order and learning effects.** Each person uses both displays; counterbalancing and a mirrored map reduce, but do not remove, practice effects. Key presses are credited to the move on screen, so very late presses can be misattributed.
# * **The explanation is one specific display** (greedy-policy arrows with a confidence fade); other explanations (Q-value maps, policy summaries such as HIGHLIGHTS) may act differently — and showing uncertainty can also distract trainers (Li et al., 2015).
#
# **Next steps.** Run the pilot (≈ 15 minutes per person); use it to estimate real error and feedback rates and their spread; update the power analysis with those numbers and pre-register a study of about the size it indicates; compare a learner that models feedback noise (e.g. Advise) with plain reward shaping; and test richer explanations in a larger domain where the person cannot see everything at once.
#
# ## References
#
# * Amir, D., & Amir, O. (2018). HIGHLIGHTS: Summarizing agent behavior to people. *AAMAS*, 1168–1176.
# * Griffith, S., Subramanian, K., Scholz, J., Isbell, C. L., & Thomaz, A. L. (2013). Policy shaping: Integrating human feedback with reinforcement learning. *NeurIPS*, 2625–2633.
# * Knox, W. B., & Stone, P. (2009). Interactively shaping agents via human reinforcement: The TAMER framework. *K-CAP*.
# * Knox, W. B., & Stone, P. (2010). Combining manual feedback with subsequent MDP reward signals for reinforcement learning. *AAMAS*.
# * Lee, K., Smith, L., Dragan, A., & Abbeel, P. (2021). B-Pref: Benchmarking preference-based reinforcement learning. *NeurIPS Datasets and Benchmarks*.
# * Li, G., Hung, H., Whiteson, S., & Knox, W. B. (2013). Using informative behavior to increase engagement in the TAMER framework. *AAMAS*. Extended version: Li, Whiteson, Knox & Hung (2015), *Autonomous Agents and Multi-Agent Systems*, doi:10.1007/s10458-015-9308-2.
# * Loftin, R., Peng, B., MacGlashan, J., Littman, M. L., Taylor, M. E., Huang, J., & Roberts, D. L. (2016). Learning behaviors via human-delivered discrete feedback: Modeling implicit feedback strategies to speed up learning. *Autonomous Agents and Multi-Agent Systems*, 30(1), 30–59.
# * MacGlashan, J., Ho, M. K., Loftin, R., Peng, B., Wang, G., Roberts, D. L., Taylor, M. E., & Littman, M. L. (2017). Interactive learning from policy-dependent human feedback. *ICML*, PMLR 70, 2285–2294.
# * Peng, B., MacGlashan, J., Loftin, R., Littman, M. L., Roberts, D. L., & Taylor, M. E. (2016). A need for speed: Adapting agent action speed to improve task learning from non-expert humans. *AAMAS*, 957–965.
# * Retzlaff, C. O., Das, S., Wayllace, C., Mousavi, P., Afshari, M., Yang, T., Saranti, A., Angerschmid, A., Taylor, M. E., & Holzinger, A. (2024). Human-in-the-loop reinforcement learning: A survey and position on requirements, challenges, and opportunities. *JAIR*, 79, 359–415.
# * Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.), Ch. 6. MIT Press.
# * Teso, S., & Kersting, K. (2019). Explanatory interactive machine learning. *AIES*, 239–245.
# * Thomaz, A. L., & Breazeal, C. (2008). Teachable robots: Understanding human teaching behavior to build more effective robot learners. *Artificial Intelligence*, 172(6–7), 716–737.
