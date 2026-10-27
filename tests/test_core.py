"""Tests for src/hitl_gridworld.py and the pilot scripts.  Run:  python -m pytest -q"""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import hitl_gridworld as hg


# ---------------------------------------------------------------- environment & oracle
def test_walls_edges_and_terminals():
    env = hg.GridWorld("A")
    start = env.start                                   # bottom-left corner (4, 0)
    assert env.rc(start) == (4, 0)
    env.reset()
    s2, r, term, trunc = env.step(3)                    # left: off the grid -> stay
    assert s2 == start and r == env.step_reward and not term
    env.reset()
    env.s = 2 * env.W + 0                               # (2, 0); right is a wall
    s2, *_ = env.step(1)
    assert s2 == 2 * env.W + 0
    env.reset()
    env.s = 4 * env.W + 2                               # (4, 2); up is lava
    s2, r, term, _ = env.step(0)
    assert env.cell_type[s2] == hg.LAVA and r == -1.0 and term
    env.s = 1 * env.W + 4                               # (1, 4); up is the goal
    s2, r, term, _ = env.step(0)
    assert env.cell_type[s2] == hg.GOAL and r == 1.0 and term


def test_truncation_after_max_episode_steps():
    env = hg.GridWorld("A", max_episode_steps=3)
    env.reset()
    flags = [env.step(3)[3] for _ in range(3)]          # bump left three times
    assert flags == [False, False, True]


def test_cliff_resets_to_start():
    env = hg.GridWorld("cliff", step_reward=-1, goal_reward=-1, lava_reward=-100,
                       lava_resets=True, max_episode_steps=None)
    env.reset()
    s2, r, term, trunc = env.step(1)                    # right from start = into the cliff
    assert s2 == env.start and r == -100 and not term and not trunc


def test_oracle_distances_and_route():
    for name in ("A", "B"):
        env = hg.GridWorld(name)
        oracle = hg.Oracle(env)
        assert oracle.start_dist == 8
        assert len(oracle.states) == 17
        assert len(oracle.route) == 8
        # every non-terminal cell has at least one correct move
        assert oracle.optimal[oracle.states].any(axis=1).all()
    env = hg.GridWorld("A")
    oracle = hg.Oracle(env)
    s = 3 * env.W + 0                                   # (3, 0): going DOWN is correct
    assert oracle.optimal[s].tolist() == [False, False, True, False]
    s = 2 * env.W + 0                                   # (2, 0): up and down are both correct
    assert oracle.optimal[s].tolist() == [True, False, True, False]


def test_mirror_layout():
    assert hg.LAYOUTS["B"] == [row[::-1] for row in hg.LAYOUTS["A"]]
    assert hg.canonical_col("B", 0) == 4 and hg.canonical_col("A", 0) == 0


# ---------------------------------------------------------------- agents & metrics
def test_q_learning_update_matches_equation_6_8():
    agent = hg.QLearningAgent(4, alpha=0.5, gamma=0.9, epsilon=0.0, seed=0)
    agent.Q[1] = [0.0, 2.0, 1.0, 0.0]
    td = agent.update(0, 3, 1.0, 1, terminated=False)
    assert td == pytest.approx(1.0 + 0.9 * 2.0 - 0.0)
    assert agent.Q[0, 3] == pytest.approx(0.5 * 2.8)
    agent.update(0, 3, 1.0, 1, terminated=True)         # terminal: no bootstrap
    assert agent.Q[0, 3] == pytest.approx(1.4 + 0.5 * (1.0 - 1.4))


def test_sarsa_update_uses_next_action():
    agent = hg.SarsaAgent(4, alpha=1.0, gamma=1.0, epsilon=0.0, seed=0)
    agent.Q[1] = [0.0, 5.0, -3.0, 0.0]
    agent.update(0, 0, -1.0, 1, terminated=False, a2=2)
    assert agent.Q[0, 0] == pytest.approx(-4.0)


def test_strict_greedy_and_learned_criterion():
    env = hg.GridWorld("A")
    oracle = hg.Oracle(env)
    Q = np.zeros((env.n_states, 4))
    assert (hg.strict_greedy(Q) == -1).all()
    assert hg.policy_accuracy(Q, oracle) == 0.0
    assert not hg.greedy_path_optimal(Q, env, oracle)
    Q[oracle.optimal] = 1.0                             # all correct moves preferred
    Q[2 * env.W + 0, 0] = 1.5                           # break the tie at (2, 0)
    assert hg.policy_accuracy(Q, oracle) == 1.0
    assert hg.route_accuracy(Q, oracle) == 1.0
    assert hg.greedy_path_optimal(Q, env, oracle)


# ---------------------------------------------------------------- simulated trainer
def test_trainer_rates_match_settings():
    env = hg.GridWorld("A")
    oracle = hg.Oracle(env)
    trainer = hg.SimulatedTrainer(oracle, error_rate=0.3, feedback_prob=0.4, seed=1)
    rng = np.random.default_rng(0)
    h = np.array([trainer(int(rng.choice(oracle.states)), int(rng.integers(4)))
                  for _ in range(40_000)])
    assert np.mean(h != 0) == pytest.approx(0.4, abs=0.01)
    assert 1 - trainer.n_correct / trainer.n_given == pytest.approx(0.3, abs=0.012)
    silent = hg.SimulatedTrainer(oracle, feedback_prob=0.0)
    assert silent(env.start, 1) == 0


def test_good_feedback_speeds_learning_and_bad_feedback_hurts():
    def median_steps(**kw):
        runs = hg.run_many(12, max_steps=4000, record_every=0, stop_when_learned=True, **kw)
        return np.median(hg.censored_steps(runs, 4000))
    alone = median_steps(feedback_prob=0.0)
    perfect = median_steps(error_rate=0.0, feedback_prob=1.0)
    noisy = median_steps(error_rate=0.5, feedback_prob=1.0)
    assert perfect < alone / 4
    assert noisy > alone


def test_run_is_reproducible():
    a = hg.run_with_trainer(error_rate=0.1, feedback_prob=0.5, seed=3, max_steps=500)
    b = hg.run_with_trainer(error_rate=0.1, feedback_prob=0.5, seed=3, max_steps=500)
    assert a["steps_to_learn"] == b["steps_to_learn"]
    assert np.array_equal(a["Q"], b["Q"])


def test_sweep_shape():
    df = hg.sweep([0.0, 0.3], [1.0], n_seeds=2, max_steps=300)
    assert len(df) == 4 and {"steps_to_learn_c", "learned"} <= set(df.columns)
    assert (df["steps_to_learn_c"] <= 300).all()


# ---------------------------------------------------------------- pilot session & analysis
def test_counterbalancing():
    assert hg.counterbalanced_order("P1") == [(1, "agent_only", "A"), (2, "explanation", "B")]
    assert hg.counterbalanced_order("P2") == [(1, "explanation", "A"), (2, "agent_only", "B")]
    with pytest.raises(ValueError):
        hg.participant_number("Alice")


def test_pilot_session_logic_and_feedback_attribution():
    s = hg.PilotSession("P9", 1, "explanation", "A", seed=0, max_steps=5)
    s.start(now=0.0)
    first = s.pending
    assert s.give_feedback(+1, now=0.3)
    assert s.give_feedback(-1, now=0.5)                 # last key press wins
    assert not s.give_feedback(7)
    s.advance(now=0.75)
    row = s.rows[0]
    assert row["feedback"] == -1 and row["rt_ms"] == pytest.approx(500.0)
    optimal = bool(s.oracle.optimal[first["s"], first["a"]])
    assert row["feedback_correct"] == int(not optimal)
    while s.advance():
        pass
    assert s.finished and len(s.rows) == 5
    assert all(r["feedback"] == 0 for r in s.rows[1:])
    summary = s.summary()
    assert summary["feedback_total"] == 1 and summary["steps_run"] == 5


def test_headless_sessions_save_load_and_plot(tmp_path):
    rng = np.random.default_rng(0)
    for k in (1, 2):
        pid = f"P{k}"
        for session_no, condition, layout in hg.counterbalanced_order(pid):
            sess = hg.PilotSession(pid, session_no, condition, layout, seed=10 * k + session_no)
            hg.run_session_headless(sess, hg.SimulatedResponder(
                sess.oracle, rng.uniform(0.4, 0.8), 0.1, seed=k + session_no))
            steps_path, summary_path = sess.save(tmp_path, extra={"simulated": True})
            log = pd.read_csv(steps_path)
            summ = json.loads(summary_path.read_text())
            assert len(log) == summ["steps_run"]
            assert (log["feedback"] != 0).sum() == summ["feedback_total"]
            assert log["feedback_correct"].dropna().sum() == summ["correct_total"]
    practice = hg.PilotSession("P1", 0, "practice", "practice", max_steps=10)
    hg.run_session_headless(practice, hg.SimulatedResponder(practice.oracle, seed=0))
    practice.save(tmp_path)
    sessions, steps = hg.load_pilot(tmp_path)
    assert len(sessions) == 4 and set(sessions["condition"]) == set(hg.CONDITIONS)
    assert hg.pilot_table(sessions).shape[0] == 4
    for fig in (hg.plot_pilot_paired(sessions, banner="test"), hg.plot_pilot_curves(steps),
                hg.plot_error_map(steps)):
        fig.savefig(tmp_path / "fig.png")
    md = hg.pilot_markdown(sessions)
    assert "| P1 |" in md and "Median per display" in md
    assert hg.load_pilot(tmp_path / "empty") == (None, None)


def test_readme_block_update(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("a\n<!-- PILOT-RESULTS:START -->\nold\n<!-- PILOT-RESULTS:END -->\nb\n")
    hg.update_readme_block(readme, "new table")
    text = readme.read_text()
    assert "new table" in text and "old" not in text and text.endswith("b\n")


def test_prediction_from_sweep_interpolates():
    df = pd.DataFrame({"feedback_prob": [0.5, 0.5, 1.0, 1.0], "error_rate": [0.0, 0.2, 0.0, 0.2],
                       "steps_to_learn_c": [100.0, 400.0, 50.0, 200.0]})
    assert hg.predict_steps(df, 0.0, 1.0) == pytest.approx(50.0)
    assert hg.predict_steps(df, 0.1, 0.5) == pytest.approx(200.0)    # geometric mean


# ---------------------------------------------------------------- statistics
def test_sign_flip_test():
    assert hg.sign_flip_pvalue([1, 1, 1, 1]) == pytest.approx(0.125)
    assert hg.sign_flip_pvalue([1, -1, 1, -1]) == pytest.approx(1.0)
    assert hg.sign_flip_pvalue(np.ones(20), rng=0) < 0.01


def test_power_simulation_runs():
    outcomes = [np.full(10, 100.0), np.full(10, 300.0)]
    power = hg.power_by_simulation(outcomes, [0.0, 0.3], base_error=0.3, deltas=(0.3,),
                                   sd_between=0.0, ns=(4, 8), reps=20, seed=0)
    assert power[0.3][0] == 0.0          # n = 4 can never reach p < 0.05
    assert power[0.3][1] == 1.0          # a huge, noise-free effect is found with n = 8


# ---------------------------------------------------------------- pilot app (no window)
def test_pilot_window_handles_keys_and_ticks(tmp_path):
    import pilot_app
    sess = hg.PilotSession("P1", 1, "explanation", "A", seed=0, max_steps=6)
    window = pilot_app.PilotWindow(sess, step_ms=750, intro_text="intro", title="test")
    window.on_key(SimpleNamespace(key=" "))              # start
    assert window.state == "running" and sess.pending is not None
    window.on_key(SimpleNamespace(key="g"))
    for _ in range(10):
        window.on_tick()
    assert sess.finished and window.state == "done"
    assert sess.rows[0]["feedback"] == 1
    window.fig.savefig(tmp_path / "window.png")
    window.on_close()


@pytest.mark.parametrize("backend, ok", [("TkAgg", True), ("QtAgg", True), ("MacOSX", True),
                                         ("GTK4Agg", True), ("agg", False), ("pdf", False),
                                         ("module://matplotlib_inline.backend_inline", False)])
def test_backend_check(monkeypatch, backend, ok):
    import pilot_app
    monkeypatch.setattr(pilot_app.matplotlib, "get_backend", lambda: backend)
    assert pilot_app.interactive_backend_available() is ok


def test_pilot_app_simulate_and_screenshot(tmp_path):
    import pilot_app
    assert pilot_app.main(["--simulate", "--data-dir", str(tmp_path), "--max-steps", "60"]) == 0
    assert len(list(tmp_path.glob("SIM1_*_summary.json"))) == 2
    assert pilot_app.main(["--screenshot", str(tmp_path / "shot.png")]) == 0
    assert (tmp_path / "shot.png").exists()


def test_analyze_script_on_simulated_data(tmp_path):
    import analyze_pilot
    import pilot_app
    data = tmp_path / "data"
    pilot_app.main(["--simulate", "--data-dir", str(data), "--max-steps", "60"])
    assert analyze_pilot.main(["--data-dir", str(data), "--fig-dir", str(tmp_path / "figs")]) == 0
    assert (tmp_path / "figs" / "pipeline_check" / "pilot_paired.png").exists()
    # simulated data must never be written into the README
    assert analyze_pilot.main(["--data-dir", str(data), "--fig-dir", str(tmp_path / "figs"),
                               "--update-readme"]) == 1
