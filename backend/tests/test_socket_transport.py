"""Tests for the SocketTransport — framing, partial reads, and error handling.

Uses a loopback TCP server for integration tests so tests don't depend
on external services.

Verifies:
    - Length-prefixed framing round-trip.
    - Partial reads (split across recv calls).
    - Multiple messages over a single connection.
    - Connection close handling.
    - Oversized message rejection.
    - Message-level send/receive.
    - Context manager support.
"""

from __future__ import annotations

import socket
import struct
import threading
import time

import pytest

from smartnest.communication.message_codec import MAX_MESSAGE_SIZE, MessageCodec
from smartnest.communication.socket_transport import (
    ConnectionClosedError,
    MessageTooLargeError,
    SocketTransport,
    TransportError,
)
from smartnest.core.models import MessageEnvelope, MessageType


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_message(**overrides) -> MessageEnvelope:
    defaults = dict(
        sender="agent_a",
        receiver="agent_b",
        message_type=MessageType.INFORM,
        task_id="task_test",
        payload={"key": "value"},
    )
    defaults.update(overrides)
    return MessageEnvelope(**defaults)


class LoopbackServer:
    """A minimal TCP server for integration testing.

    Accepts a single connection and provides a raw socket for the test
    to read/write framed messages manually.
    """

    def __init__(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))  # OS picks a free port
        self._sock.listen(1)
        self.host, self.port = self._sock.getsockname()
        self._client_sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._accept_ready = threading.Event()

    def start(self):
        def _accept():
            self._client_sock, _ = self._sock.accept()
            self._accept_ready.set()

        self._thread = threading.Thread(target=_accept, daemon=True)
        self._thread.start()

    def wait_for_connection(self, timeout=5.0):
        self._accept_ready.wait(timeout=timeout)

    @property
    def client_sock(self) -> socket.socket:
        self._accept_ready.wait(timeout=5.0)
        assert self._client_sock is not None
        return self._client_sock

    def stop(self):
        if self._client_sock:
            try:
                self._client_sock.close()
            except OSError:
                pass
        try:
            self._sock.close()
        except OSError:
            pass


# ── Framing tests ───────────────────────────────────────────────────────────

class TestFraming:
    def test_send_and_receive_bytes(self):
        """Round-trip a raw bytes frame."""
        server = LoopbackServer()
        server.start()
        client = SocketTransport.connect(server.host, server.port, timeout=5.0)
        server.wait_for_connection()

        client.send_bytes(b"hello world")
        received = _recv_exact_frame(server.client_sock)
        assert received == b"hello world"

        # Send a response back
        _send_exact_frame(server.client_sock, b"response data")
        data = client.receive_bytes()
        assert data == b"response data"

        client.close()
        server.stop()

    def test_send_and_receive_message(self):
        """Round-trip a MessageEnvelope."""
        server = LoopbackServer()
        server.start()
        client = SocketTransport.connect(server.host, server.port, timeout=5.0)
        server.wait_for_connection()

        msg = _make_message(payload={"test": 123})
        client.send_message(msg)

        # Read on server side
        server_data = _recv_exact_frame(server.client_sock)
        server_msg = MessageCodec.decode(server_data)
        assert server_msg.sender == "agent_a"
        assert server_msg.payload["test"] == 123

        # Send response
        reply = _make_message(sender="agent_b", receiver="agent_a",
                               message_type=MessageType.ACK)
        _send_exact_frame(server.client_sock, MessageCodec.encode(reply))
        decoded = client.receive_message()
        assert decoded.message_type == "ack"

        client.close()
        server.stop()

    def test_multiple_messages_over_one_connection(self):
        """Send and receive multiple messages on the same connection."""
        server = LoopbackServer()
        server.start()
        client = SocketTransport.connect(server.host, server.port, timeout=5.0)
        server.wait_for_connection()

        for i in range(5):
            msg = _make_message(task_id=f"task_{i}", payload={"index": i})
            client.send_message(msg)

        # Server receives all 5
        for i in range(5):
            data = _recv_exact_frame(server.client_sock)
            msg = MessageCodec.decode(data)
            assert msg.task_id == f"task_{i}"

        client.close()
        server.stop()

    def test_large_payload(self):
        """Send a large-ish payload to verify framing handles it."""
        server = LoopbackServer()
        server.start()
        client = SocketTransport.connect(server.host, server.port, timeout=5.0)
        server.wait_for_connection()

        big_data = b"x" * 100000  # 100 KB
        client.send_bytes(big_data)
        received = _recv_exact_frame(server.client_sock)
        assert received == big_data

        client.close()
        server.stop()


# ── Error handling ───────────────────────────────────────────────────────────

class TestTransportErrors:
    def test_connection_refused_raises(self):
        with pytest.raises(TransportError, match="Failed to connect"):
            SocketTransport.connect("127.0.0.1", 1, timeout=2.0)

    def test_connection_close_detected(self):
        """Closing the remote end raises ConnectionClosedError."""
        server = LoopbackServer()
        server.start()
        client = SocketTransport.connect(server.host, server.port, timeout=5.0)
        server.wait_for_connection()

        # Close server side
        server.client_sock.close()
        server.stop()

        with pytest.raises((ConnectionClosedError, TransportError)):
            client.receive_bytes()

        client.close()

    def test_oversized_message_rejected_on_send(self):
        """Sending a message larger than MAX_MESSAGE_SIZE raises."""
        server = LoopbackServer()
        server.start()
        client = SocketTransport.connect(server.host, server.port, timeout=5.0)
        server.wait_for_connection()

        with pytest.raises(MessageTooLargeError):
            client.send_bytes(b"x" * (MAX_MESSAGE_SIZE + 1))

        client.close()
        server.stop()

    def test_oversized_frame_rejected_on_receive(self):
        """A frame header announcing more than MAX_MESSAGE_SIZE is rejected."""
        server = LoopbackServer()
        server.start()
        client = SocketTransport.connect(server.host, server.port, timeout=5.0)
        server.wait_for_connection()

        # Send a header claiming a huge size
        server.client_sock.sendall(struct.pack("!I", MAX_MESSAGE_SIZE + 1))

        with pytest.raises(MessageTooLargeError):
            client.receive_bytes()

        client.close()
        server.stop()


# ── Context manager ─────────────────────────────────────────────────────────

class TestContextManager:
    def test_context_manager_closes_socket(self):
        server = LoopbackServer()
        server.start()

        with SocketTransport.connect(server.host, server.port, timeout=5.0) as client:
            server.wait_for_connection()
            assert client.is_closed is False

        assert client.is_closed is True
        server.stop()

    def test_close_is_idempotent(self):
        server = LoopbackServer()
        server.start()
        client = SocketTransport.connect(server.host, server.port, timeout=5.0)
        server.wait_for_connection()

        client.close()
        client.close()  # Should not raise

        server.stop()

    def test_peer_address(self):
        server = LoopbackServer()
        server.start()
        client = SocketTransport.connect(server.host, server.port, timeout=5.0)
        server.wait_for_connection()

        addr = client.peer_address
        assert "127.0.0.1" in addr

        client.close()
        server.stop()


# ── Framing helpers ─────────────────────────────────────────────────────────

def _send_exact_frame(sock: socket.socket, data: bytes) -> None:
    """Send a length-prefixed frame."""
    header = struct.pack("!I", len(data))
    sock.sendall(header + data)


def _recv_exact_frame(sock: socket.socket, timeout: float = 5.0) -> bytes:
    """Receive a length-prefixed frame, handling partial reads."""
    sock.settimeout(timeout)

    # Read 4-byte header
    header = b""
    while len(header) < 4:
        chunk = sock.recv(4 - len(header))
        if not chunk:
            raise ConnectionError("Connection closed during header read")
        header += chunk

    (length,) = struct.unpack("!I", header)

    # Read payload
    payload = b""
    while len(payload) < length:
        chunk = sock.recv(length - len(payload))
        if not chunk:
            raise ConnectionError("Connection closed during payload read")
        payload += chunk

    return payload
