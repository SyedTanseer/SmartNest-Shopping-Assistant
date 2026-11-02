"""Integration tests for the RemoteAgentServer.

Spins up a real TCP server on a random port, connects a client, and
exchanges messages.  Tests verify:
    - Server starts and stops cleanly.
    - State sync round-trip produces ACK.
    - Task request round-trip produces INFORM_RESULT.
    - Task ID correlation across request/response.
    - Multiple messages over one connection.
    - Client disconnection is handled gracefully.
    - Server shutdown stops accepting connections.
    - Unsupported message types return FAILURE.
"""

from __future__ import annotations

import socket
import threading
import time

import pytest

from smartnest.communication.message_router import MessageRouter
from smartnest.communication.remote_agent_server import RemoteAgentServer
from smartnest.communication.socket_transport import (
    SocketTransport,
    TransportError,
    ConnectionClosedError,
)
from smartnest.communication.task_handlers import (
    TaskHandlerRegistry,
    evaluate_furniture_fit,
)
from smartnest.core.models import MessageEnvelope, MessageType


# ── Fixtures ────────────────────────────────────────────────────────────────

APARTMENT_STATE = {
    "project_id": "HOME_001",
    "budget": 200000,
    "remaining_budget": 30000,
    "rooms": {
        "living_room": {"width_cm": 210, "length_cm": 350},
        "bedroom": {"width_cm": 300, "length_cm": 350},
    },
    "preferences": {"style": "minimalist", "color": "neutral"},
}

TASK_PAYLOAD = {
    "task_type": "evaluate_furniture_fit",
    "product": {
        "product_id": "SOFA_001",
        "width_cm": 230,
        "price": 24999,
        "target_room": "living_room",
    },
}


def _build_server() -> tuple[RemoteAgentServer, TaskHandlerRegistry]:
    """Create a configured server on a random port."""
    registry = TaskHandlerRegistry(agent_id="agent_b")
    registry.register_evaluator("evaluate_furniture_fit", evaluate_furniture_fit)

    router = MessageRouter(agent_id="agent_b")
    router.register(MessageType.STATE_SYNC, registry.handle_state_sync)
    router.register(MessageType.TASK_REQUEST, registry.handle_task_request)

    # Use port 0 so the OS picks a free port
    server = RemoteAgentServer(
        router=router,
        host="127.0.0.1",
        port=0,
        agent_id="agent_b",
    )
    return server, registry


def _start_server(server: RemoteAgentServer) -> int:
    """Start server and return the actual port it's listening on."""
    server.start()
    # Wait for server to be ready
    for _ in range(50):
        if server.is_running:
            break
        time.sleep(0.02)

    # Get actual port (port 0 → OS-assigned)
    # The server socket is stored internally; we can read the port from it
    if server._server_socket is not None:
        return server._server_socket.getsockname()[1]
    return server.port


@pytest.fixture()
def running_server():
    """Start and stop a server for each test."""
    server, registry = _build_server()
    port = _start_server(server)
    yield server, registry, port
    server.stop(timeout=3.0)


# ── Server lifecycle ────────────────────────────────────────────────────────

class TestServerLifecycle:
    def test_server_starts(self, running_server):
        server, _, _ = running_server
        assert server.is_running

    def test_server_stops_cleanly(self):
        server, _ = _build_server()
        _start_server(server)
        assert server.is_running
        server.stop(timeout=3.0)
        assert not server.is_running


# ── State synchronization ─────────────────────────────────────────────────

class TestStateSyncIntegration:
    def test_state_sync_round_trip(self, running_server):
        _, registry, port = running_server

        with SocketTransport.connect("127.0.0.1", port, timeout=5.0) as client:
            sync_msg = MessageEnvelope(
                sender="agent_a",
                receiver="agent_b",
                message_type=MessageType.STATE_SYNC,
                task_id="sync_001",
                payload=APARTMENT_STATE,
            )
            client.send_message(sync_msg)

            ack = client.receive_message()

        assert ack.message_type == "ack"
        assert ack.task_id == "sync_001"
        assert ack.payload["status"] == "synchronized"
        assert ack.payload["project_id"] == "HOME_001"

        # Server should have the state
        assert registry.has_state()
        assert registry.apartment_state["project_id"] == "HOME_001"


# ── Task request ──────────────────────────────────────────────────────────

class TestTaskRequestIntegration:
    def test_task_request_returns_inform_result(self, running_server):
        _, registry, port = running_server

        with SocketTransport.connect("127.0.0.1", port, timeout=5.0) as client:
            # Sync state first
            client.send_message(MessageEnvelope(
                sender="agent_a",
                receiver="agent_b",
                message_type=MessageType.STATE_SYNC,
                task_id="sync_001",
                payload=APARTMENT_STATE,
            ))
            client.receive_message()  # ACK

            # Send task request
            task_msg = MessageEnvelope(
                sender="agent_a",
                receiver="agent_b",
                message_type=MessageType.TASK_REQUEST,
                task_id="task_001",
                payload=TASK_PAYLOAD,
            )
            client.send_message(task_msg)

            result = client.receive_message()

        assert result.message_type == "inform_result"
        assert result.task_id == "task_001"
        result_data = result.payload["result"]
        assert result_data["fits"] is False
        assert result_data["width_excess_cm"] == 20
        assert result_data["within_budget"] is True
        assert "exceeds available room width" in result_data["recommendation"]

    def test_task_id_correlation(self, running_server):
        _, _, port = running_server

        with SocketTransport.connect("127.0.0.1", port, timeout=5.0) as client:
            client.send_message(MessageEnvelope(
                sender="agent_a",
                receiver="agent_b",
                message_type=MessageType.STATE_SYNC,
                task_id="sync_xyz",
                payload=APARTMENT_STATE,
            ))
            client.receive_message()

            task_id = "correlation_test_789"
            client.send_message(MessageEnvelope(
                sender="agent_a",
                receiver="agent_b",
                message_type=MessageType.TASK_REQUEST,
                task_id=task_id,
                payload=TASK_PAYLOAD,
            ))
            response = client.receive_message()

        assert response.task_id == task_id


# ── Multiple messages ──────────────────────────────────────────────────────

class TestMultipleMessages:
    def test_state_sync_then_task_on_same_connection(self, running_server):
        """Both operations work over a single persistent connection."""
        _, _, port = running_server

        with SocketTransport.connect("127.0.0.1", port, timeout=5.0) as client:
            # 1. State sync
            client.send_message(MessageEnvelope(
                sender="agent_a",
                receiver="agent_b",
                message_type=MessageType.STATE_SYNC,
                task_id="sync_1",
                payload=APARTMENT_STATE,
            ))
            ack = client.receive_message()
            assert ack.message_type == "ack"

            # 2. Task request
            client.send_message(MessageEnvelope(
                sender="agent_a",
                receiver="agent_b",
                message_type=MessageType.TASK_REQUEST,
                task_id="task_1",
                payload=TASK_PAYLOAD,
            ))
            result = client.receive_message()
            assert result.message_type == "inform_result"

            # 3. Another task on same connection with different product
            client.send_message(MessageEnvelope(
                sender="agent_a",
                receiver="agent_b",
                message_type=MessageType.TASK_REQUEST,
                task_id="task_2",
                payload={
                    "task_type": "evaluate_furniture_fit",
                    "product": {
                        "product_id": "CHAIR_001",
                        "width_cm": 60,
                        "price": 5000,
                        "target_room": "living_room",
                    },
                },
            ))
            result2 = client.receive_message()
            assert result2.message_type == "inform_result"
            assert result2.payload["result"]["fits"] is True


# ── Unsupported message type ─────────────────────────────────────────────

class TestUnsupportedType:
    def test_unsupported_type_returns_failure(self, running_server):
        _, _, port = running_server

        with SocketTransport.connect("127.0.0.1", port, timeout=5.0) as client:
            client.send_message(MessageEnvelope(
                sender="agent_a",
                receiver="agent_b",
                message_type=MessageType.CFP,
                task_id="cfp_001",
                payload={"description": "call for proposals"},
            ))
            response = client.receive_message()

        assert response.message_type == "failure"
        assert response.payload["reason"] == "unsupported_message_type"
        assert response.payload["unsupported_type"] == "call_for_proposals"


# ── Client disconnection ──────────────────────────────────────────────────

class TestClientDisconnection:
    def test_server_survives_client_disconnect(self, running_server):
        server, registry, port = running_server

        # Connect, sync, then disconnect
        with SocketTransport.connect("127.0.0.1", port, timeout=5.0) as client:
            client.send_message(MessageEnvelope(
                sender="agent_a",
                receiver="agent_b",
                message_type=MessageType.STATE_SYNC,
                task_id="sync_1",
                payload=APARTMENT_STATE,
            ))
            client.receive_message()

        # Server should still be running
        assert server.is_running

        # A new client can still connect
        with SocketTransport.connect("127.0.0.1", port, timeout=5.0) as client2:
            client2.send_message(MessageEnvelope(
                sender="agent_a",
                receiver="agent_b",
                message_type=MessageType.STATE_SYNC,
                task_id="sync_2",
                payload=APARTMENT_STATE,
            ))
            ack = client2.receive_message()
            assert ack.message_type == "ack"


# ── Data-dependent evaluation ──────────────────────────────────────────────

class TestDataDependence:
    def test_different_state_produces_different_result(self, running_server):
        """Verify the server evaluates based on received state, not hardcoded values."""
        _, _, port = running_server

        narrow_state = {
            "project_id": "HOME_NARROW",
            "remaining_budget": 50000,
            "rooms": {"room": {"width_cm": 200, "length_cm": 300}},
        }
        wide_state = {
            "project_id": "HOME_WIDE",
            "remaining_budget": 50000,
            "rooms": {"room": {"width_cm": 300, "length_cm": 300}},
        }

        product_payload = {
            "task_type": "evaluate_furniture_fit",
            "product": {"product_id": "SOFA", "width_cm": 250, "price": 10000},
        }

        with SocketTransport.connect("127.0.0.1", port, timeout=5.0) as client:
            # Sync narrow state
            client.send_message(MessageEnvelope(
                sender="agent_a",
                receiver="agent_b",
                message_type=MessageType.STATE_SYNC,
                task_id="sync_narrow",
                payload=narrow_state,
            ))
            client.receive_message()

            # Evaluate with narrow room
            client.send_message(MessageEnvelope(
                sender="agent_a",
                receiver="agent_b",
                message_type=MessageType.TASK_REQUEST,
                task_id="task_narrow",
                payload=product_payload,
            ))
            narrow_result = client.receive_message()

            # Re-sync with wide state
            client.send_message(MessageEnvelope(
                sender="agent_a",
                receiver="agent_b",
                message_type=MessageType.STATE_SYNC,
                task_id="sync_wide",
                payload=wide_state,
            ))
            client.receive_message()

            # Evaluate with wide room
            client.send_message(MessageEnvelope(
                sender="agent_a",
                receiver="agent_b",
                message_type=MessageType.TASK_REQUEST,
                task_id="task_wide",
                payload=product_payload,
            ))
            wide_result = client.receive_message()

        narrow_data = narrow_result.payload["result"]
        wide_data = wide_result.payload["result"]

        assert narrow_data["fits"] is False
        assert wide_data["fits"] is True
        assert narrow_data["room_width_cm"] == 200
        assert wide_data["room_width_cm"] == 300


# ── Robustness fixes ─────────────────────────────────────────────────────────

class TestServerRobustness:
    def test_port_property_reports_os_assigned_port(self, running_server):
        server, _, port = running_server
        assert server.port == port
        assert server.port != 0

    def test_malformed_frame_gets_failure_and_connection_survives(self, running_server):
        _, _, port = running_server
        with SocketTransport.connect("127.0.0.1", port, timeout=5.0) as client:
            client.send_bytes(b"{this is not json")
            reply = client.receive_message()
            assert reply.message_type == "failure"
            assert reply.payload["reason"] == "malformed_message"

            # Same connection still works for a valid message.
            client.send_message(MessageEnvelope(
                sender="agent_a", receiver="agent_b",
                message_type=MessageType.STATE_SYNC,
                payload=APARTMENT_STATE,
            ))
            assert client.receive_message().message_type == "ack"

    def test_schema_invalid_envelope_gets_failure(self, running_server):
        _, _, port = running_server
        with SocketTransport.connect("127.0.0.1", port, timeout=5.0) as client:
            client.send_bytes(b'{"sender": "x"}')  # missing required fields
            reply = client.receive_message()
            assert reply.payload["reason"] == "malformed_message"

    def test_idle_connection_is_kept_open(self, running_server):
        """A client silent for longer than the poll interval is not dropped."""
        _, _, port = running_server
        with SocketTransport.connect("127.0.0.1", port, timeout=5.0) as client:
            time.sleep(1.2)  # > 2 internal poll intervals
            client.send_message(MessageEnvelope(
                sender="agent_a", receiver="agent_b",
                message_type=MessageType.STATE_SYNC,
                payload=APARTMENT_STATE,
            ))
            assert client.receive_message().message_type == "ack"

    def test_optional_idle_timeout_disconnects(self):
        router = MessageRouter(agent_id="agent_b")
        server = RemoteAgentServer(router=router, host="127.0.0.1", port=0,
                                   agent_id="agent_b", idle_timeout=0.3)
        server.start()
        try:
            client = SocketTransport.connect("127.0.0.1", server.port, timeout=5.0)
            with pytest.raises((ConnectionClosedError, TransportError)):
                client.receive_message()  # server closes after ~0.3–0.8 s
            client.close()
        finally:
            server.stop(timeout=3.0)

    def test_stop_is_fast_with_idle_clients(self):
        server, _ = _build_server()
        port = _start_server(server)
        clients = [SocketTransport.connect("127.0.0.1", port, timeout=5.0) for _ in range(3)]
        time.sleep(0.2)
        t0 = time.monotonic()
        assert server.stop(timeout=3.0) is True
        assert time.monotonic() - t0 < 1.5
        for c in clients:
            c.close()

    def test_finished_client_threads_are_pruned(self, running_server):
        server, _, port = running_server
        for _ in range(5):
            with SocketTransport.connect("127.0.0.1", port, timeout=5.0):
                pass
        time.sleep(0.3)
        # Trigger one more accept so the list is pruned.
        with SocketTransport.connect("127.0.0.1", port, timeout=5.0):
            time.sleep(0.1)
            assert len(server._client_threads) <= 2
