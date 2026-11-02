"""TCP socket transport with length-prefixed framing.

Provides reliable message-oriented communication over a stream socket
using a simple framing protocol:

    [4-byte big-endian length] [UTF-8 JSON payload]

The framing ensures that ``recv()`` calls don't have to guess where one
message ends and the next begins — a common pitfall with raw TCP.

The transport is designed to work with ``MessageCodec`` for
serialization, but it operates on raw bytes so it can be used
independently of the message schema.

Thread safety:
    A single ``SocketTransport`` instance is not thread-safe for
    concurrent reads/writes on the same socket.  Each connection should
    have its own transport instance, or the caller must serialize access.
"""

from __future__ import annotations

import logging
import socket
import struct
from typing import Any

from smartnest.communication.message_codec import MAX_MESSAGE_SIZE, MessageCodec
from smartnest.core.models import MessageEnvelope

logger = logging.getLogger(__name__)

# Header is a 4-byte unsigned int (big-endian / network order).
_HEADER_FORMAT = "!I"
_HEADER_SIZE = 4


class TransportError(Exception):
    """Raised when a socket operation fails."""


class ConnectionClosedError(TransportError):
    """Raised when the remote end closes the connection."""


class MessageTooLargeError(TransportError):
    """Raised when a message exceeds ``MAX_MESSAGE_SIZE``."""


class TransportTimeoutError(TransportError):
    """Raised when a read times out *before any byte of a frame arrived*.

    The connection is still usable: no partial frame was consumed, so the
    caller may simply retry (e.g. a server polling its stop flag on an
    idle connection).  A timeout in the middle of a frame raises a plain
    ``TransportError`` instead, because the stream is then desynchronised.
    """


class SocketTransport:
    """Wraps a connected TCP socket with framed send/receive.

    Args:
        sock: A connected ``socket.socket`` instance.
        timeout: Socket timeout in seconds (default 10.0).  For a server
            this is effectively the idle-poll interval while waiting for
            the next frame header.
        frame_timeout: Timeout used once a frame header has arrived, while
            the rest of the frame (and replies) are transferred.  Defaults
            to ``max(timeout, 10.0)`` so a short idle-poll timeout never
            cuts a large message in half.
    """

    def __init__(
        self,
        sock: socket.socket,
        timeout: float = 10.0,
        frame_timeout: float | None = None,
    ) -> None:
        self._sock = sock
        self._timeout = timeout
        self._frame_timeout = frame_timeout if frame_timeout is not None else max(timeout, 10.0)
        self._sock.settimeout(timeout)
        self._closed = False

    # ── Factory methods ────────────────────────────────────────────────────

    @classmethod
    def connect(
        cls,
        host: str,
        port: int,
        timeout: float = 10.0,
    ) -> "SocketTransport":
        """Create a transport by connecting to a remote server.

        Args:
            host: Remote host (e.g. "127.0.0.1").
            port: Remote port.
            timeout: Connection and I/O timeout in seconds.

        Returns:
            A new ``SocketTransport`` over a connected socket.

        Raises:
            TransportError: If the connection fails.
        """
        try:
            sock = socket.create_connection((host, port), timeout=timeout)
            return cls(sock, timeout=timeout)
        except (socket.error, OSError) as exc:
            raise TransportError(f"Failed to connect to {host}:{port}: {exc}") from exc

    # ── Raw byte framing ─────────────────────────────────────────────────

    def send_bytes(self, data: bytes) -> None:
        """Send a length-prefixed frame of raw bytes.

        Args:
            data: The payload bytes.

        Raises:
            MessageTooLargeError: If ``data`` exceeds ``MAX_MESSAGE_SIZE``.
            TransportError: If the socket write fails.
        """
        if len(data) > MAX_MESSAGE_SIZE:
            raise MessageTooLargeError(
                f"Message size {len(data)} exceeds maximum {MAX_MESSAGE_SIZE}"
            )

        header = struct.pack(_HEADER_FORMAT, len(data))
        try:
            self._sock.settimeout(self._frame_timeout)
            self._sock.sendall(header + data)
        except (socket.error, OSError) as exc:
            raise TransportError(f"Failed to send data: {exc}") from exc
        finally:
            self._restore_timeout()

    def receive_bytes(self) -> bytes:
        """Receive one length-prefixed frame of raw bytes.

        Reads the 4-byte header first, then reads exactly that many
        payload bytes, handling partial reads correctly.

        Returns:
            The payload bytes.

        Raises:
            ConnectionClosedError: If the remote closes the connection.
            MessageTooLargeError: If the announced size exceeds the limit.
            TransportError: If the socket read fails.
        """
        header = self._recv_exact(_HEADER_SIZE, idle_timeout_ok=True)
        if header is None:
            raise ConnectionClosedError("Connection closed by remote")

        (length,) = struct.unpack(_HEADER_FORMAT, header)
        if length == 0:
            return b""
        if length > MAX_MESSAGE_SIZE:
            raise MessageTooLargeError(
                f"Announced message size {length} exceeds maximum {MAX_MESSAGE_SIZE}"
            )

        try:
            self._sock.settimeout(self._frame_timeout)
            payload = self._recv_exact(length)
        finally:
            self._restore_timeout()
        if payload is None:
            raise ConnectionClosedError("Connection closed during payload read")

        return payload

    def _restore_timeout(self) -> None:
        if self._closed:
            return
        try:
            self._sock.settimeout(self._timeout)
        except OSError:
            pass  # socket already closed

    def _recv_exact(self, n: int, idle_timeout_ok: bool = False) -> bytes | None:
        """Read exactly ``n`` bytes from the socket.

        Args:
            n: Number of bytes to read.
            idle_timeout_ok: If ``True`` and the read times out before any
                byte arrived, raise ``TransportTimeoutError`` (retryable).

        Returns ``None`` if the connection is closed before any data arrives.
        """
        chunks: list[bytes] = []
        remaining = n
        while remaining > 0:
            try:
                chunk = self._sock.recv(remaining)
            except socket.timeout as exc:
                if idle_timeout_ok and not chunks:
                    raise TransportTimeoutError(f"Socket read timed out: {exc}") from exc
                raise TransportError(f"Socket read timed out mid-frame: {exc}") from exc
            except (socket.error, OSError) as exc:
                raise TransportError(f"Socket read failed: {exc}") from exc

            if not chunk:
                if not chunks:
                    return None  # clean close before any data
                raise ConnectionClosedError(
                    "Connection closed mid-frame (expected "
                    f"{remaining} more bytes)"
                )
            chunks.append(chunk)
            remaining -= len(chunk)

        return b"".join(chunks)

    # ── Message-level convenience ─────────────────────────────────────────

    def send_message(self, message: MessageEnvelope) -> None:
        """Encode and send a ``MessageEnvelope`` over the socket."""
        data = MessageCodec.encode(message)
        self.send_bytes(data)

    def receive_message(self) -> MessageEnvelope:
        """Receive and decode a ``MessageEnvelope`` from the socket.

        Raises:
            TransportError: Subclass of TransportError for connection issues.
            CodecError: If the payload is not a valid MessageEnvelope.
        """
        data = self.receive_bytes()
        return MessageCodec.decode(data)

    # ── Connection management ─────────────────────────────────────────────

    def close(self) -> None:
        """Close the underlying socket cleanly."""
        if self._closed:
            return
        self._closed = True
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass  # Already closed or not connected
        try:
            self._sock.close()
        except OSError:
            pass

    def settimeout(self, timeout: float | None) -> None:
        """Change the socket I/O timeout used while waiting for a frame."""
        self._timeout = timeout
        self._sock.settimeout(timeout)

    @property
    def is_closed(self) -> bool:
        """Whether the transport has been closed."""
        return self._closed

    @property
    def peer_address(self) -> str:
        """Human-readable representation of the remote address."""
        try:
            addr = self._sock.getpeername()
            return f"{addr[0]}:{addr[1]}"
        except OSError:
            return "<unknown>"

    def __enter__(self) -> "SocketTransport":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
