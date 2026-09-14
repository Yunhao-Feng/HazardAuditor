#!/usr/bin/env python3
"""Launch HazardAuditor GuardPO training with VERL."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.guardpo.cispo.launcher import main


if __name__ == "__main__":
    raise SystemExit(main())
