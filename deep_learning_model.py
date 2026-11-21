"""Optional deep-learning deliverable; delegates to the shared reproducible pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

from train_models import main

if __name__ == "__main__":
    # Reuse the same input validation and exact held-out split as the ML baseline.
    sys.argv.append("--deep-learning")
    main()
