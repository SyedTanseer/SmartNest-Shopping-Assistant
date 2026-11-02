"""TCP server that accepts connections and dispatches messages.

The ``RemoteAgentServer`` listens on a configurable host/port, accepts
incoming TCP connections, reads framed ``MessageEnvelope`` objects,
routes them through a ``MessageRouter``, and sends responses back.

The server runs in a background daemon thread so it doesn't block the
main application — this keeps it compatible with future FastAPI
integration (Assignment 5.2).

Design:
    - Each client connection is handled in its own daemon thread.
    - The server uses ``threading.Event`` for graceful shutdown.  Client
      sockets poll with a short read timeout so they notice shutdown
      quickly, and :meth:`stop` also closes them to unblock any read.
    - Idle connections are kept open by default (a coordinator may hold a
      connection across several negotiation rounds).  An optional
      ``idle_timeout`` drops clients that stay silent too long.
    - A frame that is not a valid ``MessageEnvelope`` gets a ``FAILURE``
      reply; framing is intact, so the connection stays usable.
    - Business logic is injected via the router; the server itself has
      no domain knowledge.
"""

from __future__ import annotations

import logging
import socket
import threading
from time import monotonic

from smartnest.communication.message_codec import CodecError
from smartnest.communication.message_router import MessageRouter
from smartnest.communication.socket_transport import (
    ConnectionClosedError,
    SocketTransport,
    TransportError,
    TransportTimeoutError,
)
from smartnest.core.models import MessageEnvelope, MessageType

logger = logging.getLogger(__name__)

# How often idle client threads wake to check the stop flag (seconds).
_CLIENT_POLL_INTERVAL = 0.5


class RemoteAgentServer:
    """A TCP server that accepts connections and dispatches messages.

    Args:
        router: The ``MessageRouter`` to dispatch incoming messages.
        host: Listen address (default "127.0.0.1").
        port: Listen port (default 8765; 0 lets the OS choose).
        agent_id: Server agent identity for logging and error replies.
        idle_timeout: Seconds a client may stay silent before it is
            disconnected.  ``None`` (default) keeps idle clients forever.
    """

    def __init__(
        self,
        router: MessageRouter,
        host: str = "127.0.0.1",
        port: int = 8765,
        agent_id: str = "agent_b_server",
        idle_timeout: float | None = None,
    ) -> None:
        self._router = router
        self._host = host
        self._port = port
        self._agent_id = agent_id
        self._idle_timeout = idle_timeout

        self._server_socket: socket.socket | None = None
        self._stop_event = threading.Event()
        self._accept_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._client_threads: list[threading.Thread] = []
        self._client_transports: set[SocketTransport] = set()
        self._running = False

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Bind, listen, and start accepting connections in a background thread.

        Binding happens synchronously, so once ``start`` returns the server
        is ready and :attr:`port` holds the real port (even when 0 was
        requested).
        """
        if self._running:
            logger.warning("Server is already running")
            return

        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self._host, self._port))
        self._server_socket.listen(5)
        self._server_socket.settimeout(0.5)  # Poll for stop event
        self._port = self._server_socket.getsockname()[1]
        self._running = True
        self._stop_event.clear()

        self._accept_thread = threading.Thread(
            target=self._accept_loop,
            name=f"server-{self._agent_id}",
            daemon=True,
        )
        self._accept_thread.start()
        logger.info("Server %s listening on %s:%d", self._agent_id, self._host, self._port)

    def stop(self, timeout: float = 3.0) -> bool:
        """Stop the server and close all connections.

        Returns:
            ``True`` if the accept thread and all client threads joined
            within the timeout.
        """
        if not self._running:
            return True

        self._running = False
        self._stop_event.set()

        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except OSError:
                pass
            self._server_socket = None

        joined = True
        if self._accept_thread is not None and self._accept_thread.is_alive():
            self._accept_thread.join(timeout=timeout)
            if self._accept_thread.is_alive():
                joined = False

        # Close client sockets so any blocked read returns immediately.
        with self._lock:
            transports = list(self._client_transports)
            threads = list(self._client_threads)
        for transport in transports:
            transport.close()

        deadline = monotonic() + timeout
        for t in threads:
            if t.is_alive():
                t.join(timeout=max(0.0, deadline - monotonic()))
                if t.is_alive():
                    joined = False

        with self._lock:
            self._client_threads.clear()
            self._client_transports.clear()
        logger.info("Server %s stopped", self._agent_id)
        return joined

    @property
    def is_running(self) -> bool:
        """Whether the server is accepting connections."""
        return self._running

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        """The listening port (the OS-assigned one if 0 was requested)."""
        return self._port

    @property
    def active_client_count(self) -> int:
        """Number of client connections currently being served."""
        with self._lock:
            return sum(1 for t in self._client_threads if t.is_alive())

    # ── Accept loop ───────────────────────────────────────────────────────

    def _accept_loop(self) -> None:
        """Main accept loop running in the background thread."""
        while not self._stop_event.is_set():
            try:
                if self._server_socket is None:
                    break
                client_sock, client_addr = self._server_socket.accept()
                logger.info("Connection accepted from %s:%s", *client_addr)

                t = threading.Thread(
                    target=self._handle_client,
                    args=(client_sock, client_addr),
                    name=f"client-{client_addr[0]}:{client_addr[1]}",
                    daemon=True,
                )
                with self._lock:
                    # Drop finished threads so the list doesn't grow forever.
                    self._client_threads = [x for x in self._client_threads if x.is_alive()]
                    self._client_threads.append(t)
                t.start()
            except socket.timeout:
                continue  # Check stop_event
            except OSError:
                if not self._stop_event.is_set():
                    logger.exception("Accept loop error")
                break

    # ── Client handling ─────────────────────────────────────────────────

    def _handle_client(self, client_sock: socket.socket, client_addr: tuple) -> None:
        """Handle a single client connection.

        Reads messages in a loop, dispatches them through the router,
        and sends back responses.  The loop ends when the client closes
        the connection, the server stops, the optional idle timeout
        expires, or an unrecoverable transport error occurs.
        """
        transport = SocketTransport(client_sock, timeout=_CLIENT_POLL_INTERVAL)
        peer = f"{client_addr[0]}:{client_addr[1]}"
        with self._lock:
            self._client_transports.add(transport)
        last_activity = monotonic()

        try:
            while not self._stop_event.is_set():
                try:
                    message = transport.receive_message()
                except TransportTimeoutError:
                    # Idle, no partial frame consumed — keep the connection
                    # unless the optional idle limit is exceeded.
                    if (
                        self._idle_timeout is not None
                        and monotonic() - last_activity > self._idle_timeout
                    ):
                        logger.info("Client %s idle for too long — disconnecting", peer)
                        break
                    continue
                except ConnectionClosedError:
                    logger.info("Client %s disconnected", peer)
                    break
                except CodecError as exc:
                    # The frame arrived intact but isn't a valid envelope.
                    last_activity = monotonic()
                    logger.warning("Malformed message from %s: %s", peer, exc)
                    transport.send_message(self._malformed_reply(str(exc)))
                    continue
                except TransportError as exc:
                    if not self._stop_event.is_set():
                        logger.warning("Transport error with %s: %s", peer, exc)
                    break

                last_activity = monotonic()
                logger.info(
                    "Received from %s: type=%s sender=%s task=%s",
                    peer,
                    message.message_type,
                    message.sender,
                    message.task_id,
                )

                response = self._router.route(message)

                if response is not None:
                    transport.send_message(response)
                    logger.info(
                        "Sent to %s: type=%s task=%s",
                        peer,
                        response.message_type,
                        response.task_id,
                    )
                else:
                    logger.debug("No response for message %s", message.message_id)

        except TransportError as exc:
            if not self._stop_event.is_set():
                logger.warning("Failed to reply to %s: %s", peer, exc)
        except Exception:
            logger.exception("Unexpected error handling client %s", peer)
        finally:
            transport.close()
            with self._lock:
                self._client_transports.discard(transport)

    def _malformed_reply(self, error: str) -> MessageEnvelope:
        """Build the FAILURE reply for a frame that failed to decode."""
        return MessageEnvelope(
            sender=self._agent_id,
            receiver="unknown",
            message_type=MessageType.FAILURE,
            payload={"reason": "malformed_message", "error": error[:500]},
        )
