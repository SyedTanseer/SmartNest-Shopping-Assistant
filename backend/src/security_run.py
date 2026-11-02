#!/usr/bin/env python3
"""SmartNest — Assignment 5.1: Agent authentication and security (Ed25519).

Run from the ``backend`` directory:

    python src/security_run.py

Agent B (a real TCP ``RemoteAgentServer`` wrapped in a ``SecureMessageRouter``) receives messages from several senders:

  1. a legitimate, trusted coordinator            -> accepted, handler runs, the signed reply is verified by the caller
  2. the same command, modified in transit        -> rejected: the signature no longer matches
  3. an unsigned command                          -> rejected
  4. an attacker signing as the coordinator       -> rejected: she does not hold the coordinator's private key
  5. an unknown agent with a valid signature      -> rejected: not a trusted agent
  6. a trusted worker issuing a command           -> rejected: authenticated but not authorized (and its report is accepted)
  7. a captured valid command replayed            -> rejected: replay
  8. a message addressed to another agent         -> rejected
  9. a worker whose key was then revoked          -> rejected

Every decision is written to the audit log.  Keys are generated in a temporary directory (private keys are saved
encrypted); nothing is written to the repository.  Signatures give authenticity and integrity, NOT encryption.
"""

from __future__ import annotations

import base64
import copy
import logging
import sys
import tempfile
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_a_client import APARTMENT_STATE, TASK_PAYLOAD  # noqa: E402
from smartnest.communication.message_router import MessageRouter  # noqa: E402
from smartnest.communication.remote_agent_server import RemoteAgentServer  # noqa: E402
from smartnest.communication.socket_transport import SocketTransport  # noqa: E402
from smartnest.communication.task_handlers import TaskHandlerRegistry, evaluate_furniture_fit  # noqa: E402
from smartnest.core.models import MessageEnvelope, MessageType  # noqa: E402
from smartnest.security import (  # noqa: E402
    AgentKeyPair,
    AuditLog,
    SecureChannel,
    SecureMessageRouter,
    TrustedAgentRegistry,
    canonical_bytes,
    sign_envelope,
    verify_envelope,
)


class Tee:
    """Copies everything printed or logged so the run can prove no private key leaked into its own output."""

    def __init__(self, stream) -> None:
        self.stream, self.text = stream, []

    def write(self, data: str) -> int:
        self.text.append(data)
        return self.stream.write(data)

    def flush(self) -> None:
        self.stream.flush()


def sep(title: str) -> None:
    print(f"\n{'=' * 78}\n  {title}\n{'=' * 78}")


def main() -> int:
    # Runtime log: only the security audit trail (INFO) and warnings from everything else.
    tee = Tee(sys.stdout)
    sys.stdout = tee
    logging.basicConfig(level=logging.WARNING, stream=tee, format="    [log] %(message)s")
    audit_logger = logging.getLogger("smartnest.security.audit")
    audit_logger.setLevel(logging.INFO)
    failures: list[str] = []

    def check(condition: bool, text: str) -> None:
        print(f"    [{'OK' if condition else 'FAIL'}] {text}")
        if not condition:
            failures.append(text)

    sep("SmartNest - Assignment 5.1: Agent authentication (Ed25519)")

    # ── Identities and key storage ────────────────────────────────────────────
    keydir = Path(tempfile.mkdtemp(prefix="smartnest_keys_"))
    agent_b = AgentKeyPair.generate("agent_b")
    coordinator = AgentKeyPair.generate("coordinator")
    worker = AgentKeyPair.generate("research_agent")
    mallory = AgentKeyPair.generate("mallory")           # the attacker: never registered anywhere
    for pair in (agent_b, coordinator, worker):
        pair.save(keydir / f"{pair.agent_id}.pem", passphrase="demo-passphrase")
    print(f"  Private keys saved ENCRYPTED in a temporary directory ({len(list(keydir.glob('*.pem')))} files); "
          "they are never printed or logged.")
    reloaded = AgentKeyPair.load("coordinator", keydir / "coordinator.pem", "demo-passphrase")
    for pair in (agent_b, coordinator, worker, mallory):
        print(f"  {pair.agent_id:15} public-key fingerprint {pair.fingerprint}")
    check(reloaded == coordinator, "an encrypted key reloads to the same identity")
    try:
        AgentKeyPair.load("coordinator", keydir / "coordinator.pem", "wrong")
        check(False, "a wrong passphrase must not load the key")
    except Exception:
        check(True, "a wrong passphrase cannot load the key")

    # Agent B trusts the coordinator (full rights) and one worker (reports only). It has never heard of mallory.
    registry_b = TrustedAgentRegistry()
    registry_b.add("coordinator", coordinator, roles=["coordinator"])
    registry_b.add("research_agent", worker, roles=["worker"])
    # Callers trust Agent B's public key, so they can verify its replies.
    registry_clients = TrustedAgentRegistry()
    registry_clients.add("agent_b", agent_b, roles=["worker"])
    registry_clients.add("coordinator", coordinator, roles=["coordinator"])

    # ── Agent B: real handlers behind the secure router, served over real TCP ──
    handlers = TaskHandlerRegistry(agent_id="agent_b")
    handlers.register_evaluator("evaluate_furniture_fit", evaluate_furniture_fit)
    calls: list[str] = []
    inner = MessageRouter(agent_id="agent_b")

    def counted(kind: str, fn):
        def run(message: MessageEnvelope):
            calls.append(f"{kind} from {message.sender}")
            return fn(message)
        return run

    inner.register(MessageType.STATE_SYNC, counted("state_sync", handlers.handle_state_sync))
    inner.register(MessageType.TASK_REQUEST, counted("task_request", handlers.handle_task_request))
    inner.register(MessageType.INFORM, counted("inform", lambda m: MessageEnvelope(
        sender="agent_b", receiver=m.sender, message_type=MessageType.ACK, task_id=m.task_id, payload={"ok": True})))
    audit = AuditLog()
    secure = SecureMessageRouter(agent_b, inner, registry_b, audit=audit)
    server = RemoteAgentServer(router=secure, host="127.0.0.1", port=0, agent_id="agent_b")
    server.start()
    host, port = server.host, server.port
    print(f"\n  Agent B is serving on {host}:{port} behind a SecureMessageRouter.")

    def message(sender: str, mtype: MessageType, payload: dict, *, receiver: str = "agent_b") -> MessageEnvelope:
        return MessageEnvelope(sender=sender, receiver=receiver, message_type=mtype,
                               task_id=f"task_{uuid.uuid4().hex[:8]}", payload=payload)

    def raw_send(msg: MessageEnvelope) -> MessageEnvelope:
        with SocketTransport.connect(host, port, timeout=5.0) as t:
            t.send_message(msg)
            return t.receive_message()

    def outcome(reply: MessageEnvelope) -> str:
        return reply.payload.get("reason") or reply.message_type

    try:
        # 1 ─ Legitimate ─────────────────────────────────────────────────────────
        sep("1. A legitimate, trusted coordinator sends signed commands")
        with SecureChannel(coordinator, registry_clients, host, port, expected_server="agent_b") as channel:
            sync = channel.request(message("coordinator", MessageType.STATE_SYNC, APARTMENT_STATE))
            task = channel.request(message("coordinator", MessageType.TASK_REQUEST, TASK_PAYLOAD))
        print(f"  state sync -> {sync.message.message_type}; task -> {task.message.message_type}: "
              f"{task.message.payload.get('result')}")
        check(sync.message.message_type == "ack" and sync.trusted, "the state sync was accepted and the signed ACK verified")
        check(task.message.message_type == "inform_result" and task.trusted, "the task ran and the signed result verified")
        check(task.message.payload["result"]["fits"] is False, "the furniture-fit evaluation itself ran (230 cm sofa vs 210 cm wall)")
        check(task.message.signature is not None and task.verification.agent.agent_id == "agent_b",
              "the reply was signed by Agent B's key")
        legit_task = sign_envelope(message("coordinator", MessageType.TASK_REQUEST, TASK_PAYLOAD), coordinator)

        # 2 ─ Tampering ──────────────────────────────────────────────────────────
        sep("2. The same command is modified in transit (sofa width 230 -> 150 cm)")
        original = sign_envelope(message("coordinator", MessageType.TASK_REQUEST, TASK_PAYLOAD), coordinator)
        forged_payload = copy.deepcopy(original.payload)
        forged_payload["product"]["width_cm"] = 150
        tampered = original.model_copy(update={"payload": forged_payload})   # the signature is left as it was
        before = len(calls)
        reply = raw_send(tampered)
        print(f"  Agent B answers: {reply.message_type} ({outcome(reply)})")
        check(verify_envelope(tampered, registry_b).code == "bad_signature", "verification pinpoints a modified message")
        check(reply.message_type == "failure" and outcome(reply) == "authentication_failed", "the tampered command was rejected")
        check(len(calls) == before, "no handler ran for it")

        # 3 ─ Unsigned ───────────────────────────────────────────────────────────
        sep("3. An unsigned command")
        before = len(calls)
        reply = raw_send(message("coordinator", MessageType.TASK_REQUEST, TASK_PAYLOAD))
        print(f"  Agent B answers: {reply.message_type} ({outcome(reply)})")
        check(outcome(reply) == "authentication_failed" and len(calls) == before, "rejected, handler not run")

        # 4 ─ Impersonation ──────────────────────────────────────────────────────
        sep("4. Mallory claims to be the coordinator and signs with her own key")
        fake = message("coordinator", MessageType.TASK_REQUEST, TASK_PAYLOAD)
        fake = fake.model_copy(update={"signature": base64.b64encode(mallory.sign(canonical_bytes(fake))).decode()})
        before = len(calls)
        reply = raw_send(fake)
        print(f"  Agent B answers: {reply.message_type} ({outcome(reply)})")
        check(outcome(reply) == "authentication_failed" and len(calls) == before,
              "rejected: a name is not an identity - only the coordinator's private key can sign as the coordinator")

        # 5 ─ Unknown agent ──────────────────────────────────────────────────────
        sep("5. Mallory signs honestly as 'mallory' - correctly signed, but not a trusted agent")
        before = len(calls)
        reply = raw_send(sign_envelope(message("mallory", MessageType.TASK_REQUEST, TASK_PAYLOAD), mallory))
        print(f"  Agent B answers: {reply.message_type} ({outcome(reply)})")
        check(outcome(reply) == "authentication_failed" and len(calls) == before, "rejected: an unregistered key is never believed")

        # 6 ─ Authorization ──────────────────────────────────────────────────────
        sep("6. A TRUSTED worker tries to issue a command (authenticated is not authorized)")
        before = len(calls)
        reply = raw_send(sign_envelope(message("research_agent", MessageType.TASK_REQUEST, TASK_PAYLOAD), worker))
        print(f"  command  -> {reply.message_type} ({outcome(reply)})")
        check(outcome(reply) == "not_authorized" and len(calls) == before, "its command was refused")
        reply2 = raw_send(sign_envelope(message("research_agent", MessageType.INFORM, {"status": "done"}), worker))
        print(f"  its report -> {reply2.message_type}")
        check(reply2.message_type == "ack", "the same worker's report (INFORM) is accepted: roles, not blanket denial")

        # 7 ─ Replay ─────────────────────────────────────────────────────────────
        sep("7. An attacker re-sends a captured, validly signed command")
        first = raw_send(legit_task)
        second = raw_send(legit_task)
        print(f"  first delivery -> {first.message_type}; the identical copy -> {second.message_type} ({outcome(second)})")
        check(first.message_type == "inform_result", "the original delivery was accepted")
        check(outcome(second) == "message_rejected", "the replayed copy was rejected (same message id)")

        # 8 ─ Wrong receiver ─────────────────────────────────────────────────────
        sep("8. A command meant for another agent is forwarded to Agent B")
        before = len(calls)
        reply = raw_send(sign_envelope(message("coordinator", MessageType.TASK_REQUEST, TASK_PAYLOAD, receiver="agent_c"), coordinator))
        print(f"  Agent B answers: {reply.message_type} ({outcome(reply)})")
        check(outcome(reply) == "message_rejected" and len(calls) == before, "rejected: not addressed to Agent B")

        # 9 ─ Revocation ─────────────────────────────────────────────────────────
        sep("9. The worker's key is reported stolen and revoked")
        registry_b.revoke("research_agent")
        reply = raw_send(sign_envelope(message("research_agent", MessageType.INFORM, {"status": "done"}), worker))
        print(f"  its previously accepted report now -> {reply.message_type} ({outcome(reply)})")
        check(outcome(reply) == "authentication_failed", "a revoked key is rejected immediately")

        # Summary ────────────────────────────────────────────────────────────────
        sep("Audit trail and result")
        for event in audit.events:
            print("  " + event.format())
        print(f"\n  Decisions: {audit.summary()}")
        print(f"  Handlers that actually ran: {calls}")
        check(all(c.endswith(("coordinator", "research_agent")) for c in calls),
              "handlers ran only for messages from trusted senders")
        check(not any("mallory" in c for c in calls), "nothing from the attacker ever reached a handler")
        from cryptography.hazmat.primitives import serialization
        output = "".join(tee.text)
        secrets = []
        for pair in (agent_b, coordinator, worker, mallory):
            raw = pair._private.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                                              serialization.NoEncryption())
            secrets += [raw.hex(), base64.b64encode(raw).decode(), base64.urlsafe_b64encode(raw).decode()]
        check(not any(secret in output for secret in secrets) and "PRIVATE KEY" not in output,
              "no private key (hex, base64 or PEM) appears anywhere in this run's output or logs")
        print("\n  Reminder: signatures prove WHO sent a message and that it is UNCHANGED. They do not encrypt it.")
    finally:
        server.stop(timeout=3.0)
        sys.stdout = tee.stream

    if failures:
        print(f"\n  {len(failures)} check(s) FAILED:")
        for text in failures:
            print(f"    - {text}")
        return 1
    print("\n  All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
