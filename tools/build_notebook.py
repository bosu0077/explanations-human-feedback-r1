"""Build explanation_feedback_rl.ipynb from tools/notebook_source.py and src/hitl_gridworld.py.

The notebook source is a Python file in "percent" format:
  # %% [markdown]   -> a markdown cell (each line starts with "# ")
  # %%              -> a code cell
  # %% [core:NAME]  -> the code of section NAME of src/hitl_gridworld.py, copied verbatim

Usage (from the repository root):
  python tools/build_notebook.py            # write the notebook (no outputs)
  jupyter nbconvert --to notebook --execute --inplace explanation_feedback_rl.ipynb
"""
import re
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "src" / "hitl_gridworld.py"
SOURCE = ROOT / "tools" / "notebook_source.py"
OUT = ROOT / "explanation_feedback_rl.ipynb"


def core_sections():
    """{section name: code} for every '# %% [name]' section of the core module."""
    parts = re.split(r"^# %% \[(\w+)\][ \t]*$", CORE.read_text(), flags=re.M)
    return {parts[i]: parts[i + 1].strip("\n") for i in range(1, len(parts), 2)}


def build():
    sections = core_sections()
    chunks = re.split(r"^(# %%.*)$", SOURCE.read_text(), flags=re.M)
    cells, used = [], set()
    for header, body in zip(chunks[1::2], chunks[2::2]):
        body = body.strip("\n")
        if header.startswith("# %% [markdown]"):
            lines = []
            for line in body.splitlines():
                if line.startswith("# "):
                    lines.append(line[2:])
                elif line.startswith("#"):
                    lines.append(line[1:])
                else:
                    lines.append(line)
            cells.append(nbformat.v4.new_markdown_cell("\n".join(lines).strip()))
            continue
        core = re.match(r"# %% \[core:(\w+)\]", header)
        if core:
            name = core.group(1)
            if name not in sections:
                raise KeyError(f"core section {name!r} not found in {CORE.name}")
            used.add(name)
            code = (f"# ---- from src/hitl_gridworld.py, section [{name}] ----\n"
                    + sections[name])
            cells.append(nbformat.v4.new_code_cell(code))
        else:
            cells.append(nbformat.v4.new_code_cell(body))
    missing = set(sections) - used
    if missing:
        raise ValueError(f"core sections not embedded in the notebook: {sorted(missing)}")
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python",
                                 "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    nbformat.validate(nb)
    nbformat.write(nb, OUT)
    print(f"wrote {OUT.name}: {len(cells)} cells ({len(used)} core sections)")


if __name__ == "__main__":
    build()
