"""Human pilot: teach a Q-learning agent with GOOD / BAD key presses.

Run one participant (practice round + two counterbalanced sessions) from the
repository root:

    python pilot/pilot_app.py --participant P1

Keys while a session runs:   G = good    B = bad    (or press nothing)    Esc = stop
Keys on the start screens:   SPACE = start / continue

Every session is saved to pilot/data/ as
    <id>_s<k>_<display>_<map>_steps.csv     one row per agent step
    <id>_s<k>_<display>_<map>_summary.json  totals + questionnaire answers

Other modes (no window needed):
    python pilot/pilot_app.py --simulate                 # check the install with a fake participant
    python pilot/pilot_app.py --screenshot shots.png     # save both displays as an image
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

if "--simulate" in sys.argv or "--screenshot" in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from hitl_gridworld import (ACTION_NAMES, CONDITION_LABELS, CONDITIONS, INK, STATUS,  # noqa: E402
                            GridRenderer, PilotSession, SimulatedResponder, counterbalanced_order,
                            participant_number, run_session_headless, set_style)

GOOD_KEYS = {"g", "up"}
BAD_KEYS = {"b", "down"}
START_KEYS = {" ", "enter"}

INTRO = {
    "practice": (
        "PRACTICE ROUND\n\n"
        "The blue dot is a learning agent. It should reach\n"
        "the green GOAL quickly and never step into the red LAVA.\n\n"
        "After a move you like, press  G  (good).\n"
        "After a move you do not like, press  B  (bad).\n"
        "You do not have to judge every move.\n\n"
        "Click on this window, then press SPACE to start."),
    "agent_only": (
        "SESSION {k} OF 2\n\n"
        "Same task on a new map: teach the agent to reach\n"
        "the GOAL as fast as possible and to avoid the LAVA.\n\n"
        "G = good      B = bad      (or say nothing)\n\n"
        "Press SPACE to start."),
    "explanation": (
        "SESSION {k} OF 2\n\n"
        "Same task on a new map. This time the PURPLE ARROWS\n"
        "show what the agent currently plans to do in every square.\n"
        "Faded arrow = unsure;  grey dot = no plan yet.\n"
        "The arrows show its plan - not whether the plan is right.\n\n"
        "G = good      B = bad      (or say nothing)\n\n"
        "Press SPACE to start."),
}

QUESTIONS = [
    ("ease", "How easy was it to judge the agent's moves?  1 = very hard ... 5 = very easy: ",
     ["1", "2", "3", "4", "5"]),
    ("confidence", "How sure are you that your key presses were right?  1 = not at all ... "
     "5 = completely: ", ["1", "2", "3", "4", "5"]),
    ("comment", "Anything you noticed? (optional - press Enter to skip): ", None),
]


def clear_default_keymaps():
    """matplotlib binds keys such as 'g' (grid), 'q' (quit) and 's' (save) by default;
    switch them all off so only the pilot keys do anything."""
    for key in list(plt.rcParams.keys()):
        if key.startswith("keymap."):
            plt.rcParams[key] = []


NON_INTERACTIVE_BACKENDS = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}


def interactive_backend_available():
    """True for window backends such as TkAgg, QtAgg or MacOSX; False for file-only
    backends (Agg, PDF, ...) and for Jupyter's inline backend."""
    backend = matplotlib.get_backend().lower()
    return backend not in NON_INTERACTIVE_BACKENDS and "inline" not in backend


class PilotWindow:
    """One window = one session. Shows an intro screen, runs the session on a timer,
    listens to the keyboard, and shows a short end screen."""

    def __init__(self, session, step_ms, intro_text, title):
        self.session, self.step_ms = session, step_ms
        self.fig, self.ax = plt.subplots(figsize=(6.4, 7.2))
        try:
            self.fig.canvas.manager.set_window_title(title)
        except Exception:
            pass
        show_policy = session.condition == "explanation"
        self.renderer = GridRenderer(self.ax, session.env, show_policy, oracle=session.oracle)
        self.renderer.update(session.agent.Q)
        self.status = self.fig.text(0.04, 0.955, "", fontsize=11, color=INK["secondary"],
                                    va="top")
        self.flash = self.fig.text(0.96, 0.955, "", fontsize=15, weight="bold", ha="right",
                                   va="top")
        self.fig.text(0.5, 0.035, "G = good      B = bad      Esc = stop", ha="center",
                      fontsize=11, color=INK["secondary"])
        self.overlay = self.ax.text(2, 2, intro_text, ha="center", va="center", fontsize=10.5,
                                    color=INK["primary"], zorder=30, linespacing=1.4,
                                    bbox=dict(boxstyle="round,pad=0.9", facecolor="white",
                                              edgecolor=INK["axis"], alpha=0.97))
        self.state = "intro"
        self.timer = self.fig.canvas.new_timer(interval=step_ms)
        self.timer.add_callback(self.on_tick)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("close_event", self.on_close)
        self.fig.subplots_adjust(left=0.03, right=0.97, top=0.9, bottom=0.08)

    # -- events --------------------------------------------------------------------------
    def on_key(self, event):
        key = (event.key or "").lower()
        if self.state == "intro" and key in START_KEYS:
            self.overlay.set_visible(False)
            self.state = "running"
            self.session.start(now=time.perf_counter())
            self._draw_step()
            self.timer.start()
        elif self.state == "running":
            if key in GOOD_KEYS:
                self._feedback(+1)
            elif key in BAD_KEYS:
                self._feedback(-1)
            elif key == "escape":
                self.session.abort()
                self._finish()
        elif self.state == "done" and (key in START_KEYS or key == "escape"):
            plt.close(self.fig)

    def on_tick(self):
        if self.state != "running":
            return
        if not self.session.advance(now=time.perf_counter()):
            self._finish()
            return
        self._draw_step()

    def on_close(self, event=None):
        self.timer.stop()
        if self.state == "running":
            self.session.abort()
        self.state = "closed"

    # -- drawing -------------------------------------------------------------------------
    def _feedback(self, h):
        if self.session.give_feedback(h, now=time.perf_counter()):
            self.flash.set_text("GOOD" if h > 0 else "BAD")
            self.flash.set_color(STATUS["good_text"] if h > 0 else STATUS["critical"])
            self.fig.canvas.draw_idle()

    def _draw_step(self):
        p = self.session.pending
        self.renderer.update(self.session.agent.Q, p["s"], p["a"], p["s2"])
        note = ""
        if p["terminated"]:
            note = "   -  reached the GOAL" if p["r"] > 0 else "   -  fell into the LAVA"
        elif p["truncated"]:
            note = "   -  time out, back to start"
        self.status.set_text(f"Episode {p['episode']}   ·   step {p['step']} of "
                             f"{self.session.max_steps}{note}")
        self.flash.set_text("")
        self.fig.canvas.draw_idle()

    def _finish(self):
        self.timer.stop()
        self.state = "done"
        s = self.session
        if s.learned_at is not None:
            msg = f"The agent has learned the task\n(after {s.learned_at} steps). Thank you!"
        elif s.aborted:
            msg = "Session stopped."
        else:
            msg = "Time is up - thank you!"
        self.overlay.set_text(msg + "\n\nPress SPACE to continue.")
        self.overlay.set_visible(True)
        self.fig.canvas.draw_idle()


def run_window(session, step_ms, intro_text, title):
    """Show a session window and block until it is closed."""
    window = PilotWindow(session, step_ms, intro_text, title)
    plt.show()
    if not session.finished:           # window closed before the session ended
        session.abort()
    return window


def ask(prompt, choices):
    while True:
        try:
            answer = input(prompt).strip()
        except EOFError:
            return ""
        if choices is None or answer in choices:
            return answer
        print("   please type one of: " + ", ".join(choices))


def run_participant(args):
    pid = args.participant
    participant_number(pid)                         # validates the id (needs a number)
    data_dir = Path(args.data_dir)
    existing = sorted(data_dir.glob(f"{pid}_s[12]_*_summary.json"))
    if existing and not args.overwrite:
        print(f"Data for {pid} already exists in {data_dir}. Use a new id or --overwrite.")
        return 1
    clear_default_keymaps()
    set_style()
    base_seed = 1000 * participant_number(pid) + args.seed
    if not args.no_practice:
        practice = PilotSession(pid, 0, "practice", "practice", seed=base_seed, max_steps=30)
        run_window(practice, args.step_ms, INTRO["practice"], "Pilot - practice")
        practice.save(data_dir, extra={"step_ms": args.step_ms})
    for session_no, condition, layout in counterbalanced_order(pid):
        session = PilotSession(pid, session_no, condition, layout, seed=base_seed + session_no,
                               max_steps=args.max_steps)
        run_window(session, args.step_ms, INTRO[condition].format(k=session_no),
                   f"Pilot - session {session_no} of 2")
        answers = {}
        if not args.no_questions:
            print(f"\n--- {pid}, session {session_no} ({CONDITION_LABELS[condition]}) ---")
            answers = {name: ask(prompt, choices) for name, prompt, choices in QUESTIONS}
        steps_path, summary_path = session.save(
            data_dir, extra={"questionnaire": answers, "step_ms": args.step_ms})
        s = session.summary()
        print(f"saved {summary_path.name}: learned={s['learned']} "
              f"steps={s['steps_to_learn'] or s['steps_run']} presses={s['feedback_total']} "
              f"accuracy={s['accuracy'] if s['accuracy'] is None else round(s['accuracy'], 2)}")
    if not args.no_questions:
        final = {"participant": pid, "preferred_display": ask(
            "\nWhich display made teaching easier?  1 = agent only, 2 = agent + arrows, "
            "3 = no difference: ", ["1", "2", "3"]),
            "comment": ask("Any last comment? (optional - Enter to skip): ", None)}
        (data_dir / f"{pid}_final.json").write_text(json.dumps(final, indent=2))
    print(f"\nDone. Files are in {data_dir}. Next: python pilot/analyze_pilot.py")
    return 0


def run_simulation(args):
    """Install check: one fake participant, no window, written to pilot/pipeline_check/."""
    out = Path(args.data_dir if args.data_dir != DEFAULT_DATA else ROOT / "pilot" / "pipeline_check")
    pid = args.participant or "SIM1"
    for session_no, condition, layout in counterbalanced_order(pid):
        session = PilotSession(pid, session_no, condition, layout, seed=args.seed + session_no,
                               max_steps=args.max_steps)
        responder = SimulatedResponder(session.oracle, feedback_prob=0.6, error_rate=0.1,
                                       seed=args.seed + 10 * session_no)
        run_session_headless(session, responder, step_ms=args.step_ms)
        _, summary_path = session.save(out, extra={"simulated": True})
        s = session.summary()
        acc = "n/a" if s["accuracy"] is None else f"{s['accuracy']:.2f}"
        print(f"{summary_path.name}: learned={s['learned']} at step {s['steps_to_learn']}, "
              f"{s['feedback_total']} presses, accuracy {acc}")
    print(f"Simulation OK - files in {out} (these are NOT human data).")
    return 0


def save_screenshot(path):
    """Both displays side by side at a mid-session moment (for documentation)."""
    set_style()
    demo = PilotSession("demo", 1, "explanation", "A", seed=0, max_steps=55)
    run_session_headless(demo, SimulatedResponder(demo.oracle, 0.7, 0.1, seed=1))
    last = demo.rows[-1]
    w = demo.env.W
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.6))
    for ax, cond in zip(axes, CONDITIONS):
        GridRenderer(ax, demo.env, cond == "explanation", demo.oracle).update(
            demo.agent.Q, last["row"] * w + last["col"],
            ACTION_NAMES.index(last["action"]),
            last["next_row"] * w + last["next_col"])
        ax.set_title(CONDITION_LABELS[cond], loc="center")
    fig.savefig(path)
    print(f"saved {path}")
    return 0


DEFAULT_DATA = str(ROOT / "pilot" / "data")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--participant", help="participant id with a number, e.g. P1")
    parser.add_argument("--data-dir", default=DEFAULT_DATA, help="where to save the logs")
    parser.add_argument("--step-ms", type=int, default=750, help="time per agent step (ms)")
    parser.add_argument("--max-steps", type=int, default=300, help="step limit per session")
    parser.add_argument("--seed", type=int, default=0, help="base random seed")
    parser.add_argument("--no-practice", action="store_true", help="skip the practice round")
    parser.add_argument("--no-questions", action="store_true", help="skip the questionnaire")
    parser.add_argument("--overwrite", action="store_true", help="replace existing data")
    parser.add_argument("--simulate", action="store_true",
                        help="run headless with a simulated participant (install check)")
    parser.add_argument("--screenshot", metavar="PNG", help="save both displays to an image")
    args = parser.parse_args(argv)
    if args.screenshot:
        return save_screenshot(args.screenshot)
    if args.simulate:
        return run_simulation(args)
    if not args.participant:
        parser.error("--participant is required (e.g. --participant P1)")
    if not interactive_backend_available():
        print("No interactive matplotlib window is available (backend: "
              f"{matplotlib.get_backend()}). Run this from a normal terminal on a laptop, not "
              "inside Jupyter. On Linux you may need: sudo apt install python3-tk")
        return 2
    return run_participant(args)


if __name__ == "__main__":
    sys.exit(main())
