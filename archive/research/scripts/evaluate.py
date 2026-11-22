"""CLI entry point for puzzle evaluation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nyt_connections_solver.evaluation import main


if __name__ == "__main__":
    main()
