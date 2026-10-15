# Ensure tests can import top-level modules when run by CI
# This prepends the repository root to sys.path so `from train_model import ...` works
# regardless of how pytest is invoked in the workflow.
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
