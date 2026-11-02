#!/usr/bin/env python3
"""SmartNest — Assignment 2.2: Agent A (Home Configuration Client).

Run in Terminal 2 (after Agent B server is running):

    python src/agent_a_client.py

Agent A connects to Agent B over TCP, transfers its apartment
configuration, receives an acknowledgment, then sends a furniture
evaluation task request and displays the result.

Usage:
    python src/agent_a_client.py [--host 127.0.0.1] [--port 8765]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid

from smartnest.communication.socket_transport import SocketTransport, TransportError
from smartnest.core.models import MessageEnvelope, MessageType


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def _print_separator(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def _print_message(label: str, msg: MessageEnvelope) -> None:
    print(f"\n  {label}:")
    print(f"     Message ID:    {msg.message_id}")
    print(f"     Message type:  {msg.message_type}")
    print(f"     Sender:        {msg.sender}")
    print(f"     Receiver:      {msg.receiver}")
    print(f"     Task ID:       {msg.task_id}")
    print(f"     Timestamp:     {msg.timestamp.isoformat()}")
    print(f"     Payload:")
    for line in json.dumps(msg.payload, indent=6, ensure_ascii=False).splitlines():
        print(f"       {line}")


# ── Agent A apartment state ─────────────────────────────────────────────────

APARTMENT_STATE = {
    "project_id": "HOME_001",
    "budget": 200000,
    "remaining_budget": 30000,
    "rooms": {
        "living_room": {"width_cm": 210, "length_cm": 350},
        "bedroom": {"width_cm": 300, "length_cm": 350},
    },
    "preferences": {
        "style": "minimalist",
        "color": "neutral",
    },
}

# ── Task request payload ───────────────────────────────────────────────────

TASK_PAYLOAD = {
    "task_type": "evaluate_furniture_fit",
    "product": {
        "product_id": "SOFA_001",
        "width_cm": 230,
        "price": 24999,
        "target_room": "living_room",
    },
}


def run(host: str = "127.0.0.1", port: int = 8765) -> None:
    """Run the Agent A client demonstration."""
    _configure_logging()
    logger = logging.getLogger("smartnest.agent_a")

    _print_separator("SmartNest — Agent A: Home Configuration Client")
    print(f"  Connecting to Agent B at {host}:{port}")
    print(f"  Apartment: {APARTMENT_STATE['project_id']} "
          f"(budget ₹{APARTMENT_STATE['budget']:,}, "
          f"remaining ₹{APARTMENT_STATE['remaining_budget']:,})")

    # ── Connect to Agent B ───────────────────────────────────────────────
    try:
        transport = SocketTransport.connect(host, port, timeout=10.0)
    except TransportError as exc:
        print(f"\n  ✗ Failed to connect: {exc}")
        print(f"  Make sure Agent B server is running on {host}:{port}")
        sys.exit(1)

    print(f"\n  ✓ Connected to Agent B at {transport.peer_address}")

    # ── Step 1: State synchronization ─────────────────────────────────────
    _print_separator("Step 1: State Synchronization")
    print("  Sending apartment configuration to Agent B...")

    state_task_id = f"task_{uuid.uuid4().hex[:8]}"
    state_msg = MessageEnvelope(
        sender="agent_a",
        receiver="agent_b",
        message_type=MessageType.STATE_SYNC,
        task_id=state_task_id,
        payload=APARTMENT_STATE,
    )
    _print_message("Sending", state_msg)
    transport.send_message(state_msg)

    ack_msg = transport.receive_message()
    _print_message("Received ACK", ack_msg)

    if ack_msg.message_type == "ack":
        print("\n  ✓ State synchronization successful!")
    else:
        print(f"\n  ✗ Unexpected response: {ack_msg.message_type}")
        transport.close()
        return

    # ── Step 2: Task request ──────────────────────────────────────────────
    _print_separator("Step 2: Furniture Evaluation Task Request")
    print(f"  Requesting furniture fit evaluation for:")
    print(f"     Product:  {TASK_PAYLOAD['product']['product_id']}")
    print(f"     Width:    {TASK_PAYLOAD['product']['width_cm']} cm")
    print(f"     Price:    ₹{TASK_PAYLOAD['product']['price']:,}")
    print(f"     Room:     {TASK_PAYLOAD['product'].get('target_room', 'first available')}")

    task_id = f"task_{uuid.uuid4().hex[:8]}"
    task_msg = MessageEnvelope(
        sender="agent_a",
        receiver="agent_b",
        message_type=MessageType.TASK_REQUEST,
        task_id=task_id,
        payload=TASK_PAYLOAD,
    )
    _print_message("Sending", task_msg)
    transport.send_message(task_msg)

    result_msg = transport.receive_message()
    _print_message("Received Result", result_msg)

    if result_msg.message_type == "inform_result":
        result = result_msg.payload.get("result", {})
        _print_separator("Evaluation Result")
        print(f"     Fits:           {result.get('fits')}")
        print(f"     Width excess:   {result.get('width_excess_cm')} cm")
        print(f"     Room width:     {result.get('room_width_cm')} cm")
        print(f"     Product width:  {result.get('product_width_cm')} cm")
        print(f"     Within budget:  {result.get('within_budget')}")
        print(f"     Recommendation: {result.get('recommendation')}")
    else:
        print(f"\n  ✗ Unexpected response type: {result_msg.message_type}")

    # ── Verify task ID correlation ───────────────────────────────────────
    _print_separator("Task ID Correlation Check")
    if result_msg.task_id == task_id:
        print(f"  ✓ Task IDs match: {task_id}")
    else:
        print(f"  ✗ Task ID mismatch: sent {task_id}, received {result_msg.task_id}")

    # ── Done ──────────────────────────────────────────────────────────────
    transport.close()
    _print_separator("Agent A Complete")
    print("  State synchronization: ✓")
    print("  Task request:          ✓")
    print("  Result received:       ✓")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartNest Agent A Client")
    parser.add_argument("--host", default="127.0.0.1", help="Server address")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    args = parser.parse_args()
    run(host=args.host, port=args.port)
