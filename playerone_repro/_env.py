from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_diffsynth() -> Path:
    root = Path(__file__).resolve().parents[1]
    diffsynth_root = root / "diffsynth-studio"
    if diffsynth_root.exists():
        diffsynth_root_str = str(diffsynth_root)
        if diffsynth_root_str not in sys.path:
            sys.path.insert(0, diffsynth_root_str)
    return root


PROJECT_ROOT = bootstrap_diffsynth()

