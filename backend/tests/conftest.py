"""Pytest configuration — ensure src/ is importable."""

import sys
from pathlib import Path

# Add src/ to sys.path so tests can import smartnest without installation.
_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
