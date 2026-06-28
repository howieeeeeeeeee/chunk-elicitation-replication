"""Compatibility entrypoint for the replication analysis pipeline.

Run from the replication repository root with:

    uv run python scripts/05_Run_Analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path


REPLICATION_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPLICATION_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from main_analysis.pipeline import main


if __name__ == "__main__":
    main()
