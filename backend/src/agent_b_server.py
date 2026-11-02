#!/usr/bin/env python3
"""SmartNest — Assignment 2.2: Agent B (Remote Space Evaluation Server).

Run in Terminal 1:

    python src/agent_b_server.py

Agent B listens on 127.0.0.1:8765 for incoming TCP connections from
Agent A, synchronizes apartment state, and handles task requests for
furniture fit evaluation.

Usage:
    python src/agent_b_server.py [--host 127.0.0.1] [--port 8765]
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time

from smartnest.communication.message_router import MessageRouter
from smartnest.communication.remote_agent_server import RemoteAgentServer
from smartnest.communication.task_handlers import (
    TaskHandlerRegistry,
    evaluate_furniture_fit,
)
from smartnest.core.models import MessageType


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] [%(threadName)-20s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def _print_separator(title: str) -> None:
    print(f"\n{'=' * 70}", file=sys.stderr)
    print(f"  {title}", file=sys.stderr)
    print(f"{'=' * 70}", file=sys.stderr)


def run(host: str = "127.0.0.1", port: int = 8765) -> None:
    """Start the Agent B server."""
    _configure_logging()
    logger = logging.getLogger("smartnest.agent_b")

    _print_separator("SmartNest — Agent B: Remote Space Evaluation Server")
    print(f"  Listening on {host}:{port}", file=sys.stderr)
    print(f"  Press Ctrl+C to shut down.", file=sys.stderr)

    # ── Build handler registry ───────────────────────────────────────────
    registry = TaskHandlerRegistry(agent_id="agent_b")
    registry.register_evaluator("evaluate_furniture_fit", evaluate_furniture_fit)

    # ── Build router ──────────────────────────────────────────────────────
    router = MessageRouter(agent_id="agent_b")
    router.register(MessageType.STATE_SYNC, registry.handle_state_sync)
    router.register(MessageType.TASK_REQUEST, registry.handle_task_request)

    # ── Start server ──────────────────────────────────────────────────────
    server = RemoteAgentServer(
        router=router,
        host=host,
        port=port,
        agent_id="agent_b",
    )

    # ── Handle Ctrl+C ─────────────────────────────────────────────────────
    def _signal_handler(sig, frame):
        logger.info("Shutdown signal received")
        server.stop(timeout=3.0)
        print(f"\n  Agent B server stopped.", file=sys.stderr)
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)

    server.start()

    _print_separator("Server Ready")
    print(f"  Registered task types: {registry.registered_task_types}", file=sys.stderr)
    print(f"  Registered message types: {router.registered_types}", file=sys.stderr)
    print(f"  Waiting for Agent A to connect...\n", file=sys.stderr)

    # Keep main thread alive
    try:
        while server.is_running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        server.stop(timeout=3.0)
        print(f"\n  Agent B server stopped.", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartNest Agent B Server")
    parser.add_argument("--host", default="127.0.0.1", help="Listen address")
    parser.add_argument("--port", type=int, default=8765, help="Listen port")
    args = parser.parse_args()
    run(host=args.host, port=args.port)
