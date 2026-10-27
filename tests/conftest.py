import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")          # no windows during tests
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "pilot"))
