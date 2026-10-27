"""Analyse the human pilot: tables + figures from the logs written by pilot_app.py.

    python pilot/analyze_pilot.py                  # print tables, write figures/pilot_*.png
    python pilot/analyze_pilot.py --update-readme  # also refresh the results block in README.md

The analysis is the one fixed in advance in the notebook (Part 3): per participant and
display - key presses until the agent learned, share of presses that were correct,
and agent steps until learned. With 3-4 people it is descriptive only.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import pandas as pd  # noqa: E402

from hitl_gridworld import (load_pilot, pilot_markdown, pilot_medians, pilot_table,  # noqa: E402
                            plot_error_map, plot_observed_vs_predicted, plot_pilot_curves,
                            plot_pilot_paired, set_style, update_readme_block)

BANNER = "PIPELINE CHECK — simulated participants, not human data"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(ROOT / "pilot" / "data"))
    parser.add_argument("--fig-dir", default=str(ROOT / "figures"))
    parser.add_argument("--update-readme", action="store_true",
                        help="write the results table into README.md (real data only)")
    args = parser.parse_args(argv)

    sessions, steps = load_pilot(args.data_dir)
    if sessions is None:
        print(f"No pilot sessions found in {args.data_dir}. Run pilot/pilot_app.py first.")
        return 1
    simulated = "simulated" in sessions.columns and bool(sessions["simulated"].eq(True).any())
    banner = BANNER if simulated else None
    fig_dir = Path(args.fig_dir) / ("pipeline_check" if simulated else "")
    fig_dir.mkdir(parents=True, exist_ok=True)

    set_style()
    pd.set_option("display.width", 160)
    print(pilot_table(sessions).to_string(index=False))
    print("\nMedians per display:")
    print(pilot_medians(sessions).round(2).to_string())

    plot_pilot_paired(sessions, banner=banner).savefig(fig_dir / "pilot_paired.png")
    plot_pilot_curves(steps, banner=banner).savefig(fig_dir / "pilot_curves.png")
    plot_error_map(steps, banner=banner).savefig(fig_dir / "pilot_error_map.png")
    sweep_csv = ROOT / "results" / "part2_sweep.csv"
    if sweep_csv.exists():
        plot_observed_vs_predicted(sessions, pd.read_csv(sweep_csv), banner=banner).savefig(
            fig_dir / "pilot_observed_vs_predicted.png")
    table_path = fig_dir / "pilot_summary.csv" if simulated else ROOT / "results" / "pilot_summary.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    pilot_table(sessions).to_csv(table_path, index=False)
    print(f"\nFigures written to {fig_dir}; table written to {table_path}")

    if args.update_readme:
        if simulated:
            print("Refusing to put simulated participants into the README.")
            return 1
        update_readme_block(ROOT / "README.md", pilot_markdown(sessions))
        print("README.md pilot section updated - review it, then commit and push.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
