#!/usr/bin/env python3
"""Run the SmartNest API:  python src/serve.py  [--port 8000] [--atlas]

The API makes no SerpApi request; it serves saved real snapshots (see capture_catalog.py).
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--atlas", action="store_true", help="persist sessions in MongoDB Atlas")
    args = ap.parse_args()
    if args.atlas:
        os.environ["SMARTNEST_USE_ATLAS"] = "1"
    import uvicorn
    uvicorn.run("smartnest.api.app:get_app", factory=True, host=args.host, port=args.port, log_level="info")
