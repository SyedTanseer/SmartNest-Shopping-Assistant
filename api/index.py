"""Vercel entry point: serves the SmartNest FastAPI app (every /api/* request is rewritten here, see vercel.json)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend" / "src"))

from smartnest.api.app import get_app  # noqa: E402

app = get_app()
