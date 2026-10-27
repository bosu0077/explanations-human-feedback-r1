# Human pilot — protocol

**Purpose.** Check whether showing the agent's current plan (policy arrows) changes how people give good/bad feedback, and whether the procedure works. This is a pilot with 3–4 people: results are reported descriptively; no hypothesis test is planned (with n = 4 the smallest possible two-sided sign-flip p-value is 0.125).

## Participants and consent

* 3–4 adults (friends are fine); no RL background needed. IDs **P1, P2, P3, P4** — never names.
* Read aloud before starting:

  > "This is a short study about teaching a computer agent with key presses. It takes about 15 minutes. You can stop at any time without giving a reason. I will save only your key presses, their timing and your answers to three short questions, under an ID such as P1 — no name or other personal information. Is it OK to continue?"

* Only continue after a clear "yes". (A full study would need approval from a research ethics board.)

## Equipment

A laptop with Python 3.11+ and the packages in `requirements.txt` (`pip install -r requirements.txt`), a keyboard, and a quiet place. On Linux, matplotlib windows need `sudo apt install python3-tk`. Check the install first:

```bash
python pilot/pilot_app.py --simulate      # runs a fake participant without a window
```

## Procedure (≈ 15 minutes per person)

```bash
python pilot/pilot_app.py --participant P1
```

1. **Practice round** (30 steps, open map, no arrows) — learn the keys. Not analysed.
2. **Session 1** (≤ 300 steps ≈ 4 minutes) — then 3 questions in the terminal.
3. **Session 2** with the other display — then 3 questions, and one final question on which display was easier.
4. **Debrief:** explain what the two displays were and why; ask which they preferred and why (write notes).

During a session: **G** = good, **B** = bad, nothing = no feedback, **Esc** = stop. One agent move is shown every 0.75 s; a key press counts for the move on screen (the last press counts). The session ends when the agent has learned the task — its greedy path from the start is a shortest path — or after 300 steps. Without feedback the agent almost never learns within 300 steps (2 % of simulated runs), so learning in a session comes from the person's feedback.

## Counterbalancing

| Participant | Session 1 | Session 2 |
|---|---|---|
| P1 | agent only — map A | agent + policy arrows — map B |
| P2 | agent + policy arrows — map A | agent only — map B |
| P3 | agent only — map A | agent + policy arrows — map B |
| P4 | agent + policy arrows — map A | agent only — map B |

Map B is the mirror image of map A, so people cannot reuse what they learned about the first map. The app chooses the order automatically from the participant number.

## Measures (fixed before data collection)

Per session:

1. **Feedback needed** — key presses until the agent learned (all presses if it never learned).
2. **Feedback correct** — share of key presses that agree with the shortest-path oracle (*good* on a shortest-path move, *bad* on any other move).
3. **Learning speed** — agent steps until learned; sessions that hit the 300-step limit are censored at 300.

Also recorded: presses per step, reaction times, where (which cell) wrong presses happen, and the questionnaire (ease of judging 1–5, confidence 1–5, comments, preferred display).

## Analysis

```bash
python pilot/analyze_pilot.py                  # tables + figures/pilot_*.png
python pilot/analyze_pilot.py --update-readme  # also writes the results block in README.md
```

The analysis is the one in Part 3 of the notebook: a per-person paired plot for each measure, medians per display, per-session learning curves, a map of where feedback was wrong, and a check of whether the Part 2 simulation predicts each session's learning speed from its accuracy and feedback rate. Descriptive only.

## Data files

Each session writes two files to `pilot/data/`:

**`<id>_s<k>_<display>_<map>_steps.csv`** — one row per agent step

| column | meaning |
|---|---|
| participant, session, condition, layout | ID, 1 or 2 (0 = practice), `agent_only` / `explanation`, `A` / `B` |
| step, episode | step number in the session; episode number (an episode ends at the goal, in lava, or after 50 steps) |
| row, col, action, next_row, next_col | the move shown on screen (row 0 = top) |
| env_reward, terminated, truncated | environment reward of the move; episode end flags |
| optimal_action | 1 if the move is on a shortest path (oracle) |
| feedback | +1 good, −1 bad, 0 no key |
| feedback_correct | 1 / 0 if a key was pressed (empty otherwise) |
| rt_ms | time from the move appearing to the key press (ms) |
| policy_accuracy, route_accuracy | after learning from this step: share of all 17 cells / of the 8 route cells whose greedy action is correct |
| greedy_path_optimal, learned | greedy path from the start optimal right now; agent has learned at or before this step |

**`<id>_s<k>_<display>_<map>_summary.json`** — totals (`steps_to_learn`, `feedback_to_learn`, `correct_to_learn`, `accuracy`, `feedback_rate`, `median_rt_ms`, …), the agent settings, the app version, and the questionnaire answers. `<id>_final.json` holds the final preference question.
