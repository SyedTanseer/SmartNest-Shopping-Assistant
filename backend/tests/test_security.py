"""Assignment 5.1 - agent authentication and message integrity (Ed25519).

Covers key handling, canonical signing, tamper detection, the trusted-agent registry, authorization, replay
protection, the secure router, real TCP sockets, reply verification and isolation of the security layer.
"""

from __future__ import annotations

import ast
import base64
import copy
import logging
import os
import pathlib
import pickle
import socket
import threading
from datetime import datetime, timedelta, timezone

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa

from smartnest.communication.message_codec import MessageCodec
from smartnest.communication.message_router import MessageRouter
from smartnest.communication.remote_agent_server import RemoteAgentServer
from smartnest.communication.socket_transport import SocketTransport
from smartnest.core.models import MessageEnvelope, MessageType
from smartnest.security import (
    COMMAND_TYPES,
    AgentKeyPair,
    AuditLog,
    AuthorizationPolicy,
    KeyManagementError,
    RegistryError,
    ReplayGuard,
    SecureChannel,
    SecureMessageRouter,
    SigningError,
    TrustedAgentRegistry,
    canonical_bytes,
    fingerprint_of,
    sign_envelope,
    verify_envelope,
)
from smartnest.security.keys import public_key_from_b64, public_key_from_bytes

NOW = datetime(2026, 11, 1, 12, 0, 0, tzinfo=timezone.utc)


def msg(sender="coordinator", receiver="agent_b", mtype=MessageType.TASK_REQUEST, payload=None, **over) -> MessageEnvelope:
    return MessageEnvelope(sender=sender, receiver=receiver, message_type=mtype, task_id="task_1",
                           payload=payload if payload is not None else {"task_type": "evaluate", "product": {"width_cm": 230, "tags": ["a", "b"]}},
                           timestamp=over.pop("timestamp", datetime.now(timezone.utc)), **over)


@pytest.fixture()
def keys():
    return {name: AgentKeyPair.generate(name) for name in ("agent_b", "coordinator", "worker", "observer", "mallory")}


@pytest.fixture()
def registry(keys):
    r = TrustedAgentRegistry()
    r.add("coordinator", keys["coordinator"], ["coordinator"])
    r.add("worker", keys["worker"], ["worker"])
    r.add("observer", keys["observer"], ["observer"])
    r.add("agent_b", keys["agent_b"], ["worker"])
    return r


# ══ Key management ═══════════════════════════════════════════════════════════

class TestKeys:
    def test_each_generated_key_is_unique(self):
        a, b = AgentKeyPair.generate("a"), AgentKeyPair.generate("b")
        assert a.public_key_bytes != b.public_key_bytes and len(a.public_key_bytes) == 32
        assert a.fingerprint != b.fingerprint and len(a.fingerprint) == 16 and fingerprint_of(a.public_key_bytes) == a.fingerprint

    @pytest.mark.parametrize("bad", ["", " ", "a b", "x" * 129, "../etc", "-start", ".dot", None, 5, "bad\nid", "a/b"])
    def test_invalid_agent_ids_are_refused(self, bad):
        with pytest.raises(KeyManagementError, match="agent_id"):
            AgentKeyPair.generate(bad)

    @pytest.mark.parametrize("good", ["a", "agent_b", "agent-b.1", "research_agent:eu", "X" * 128])
    def test_valid_agent_ids(self, good):
        assert AgentKeyPair.generate(good).agent_id == good

    def test_signatures_are_64_bytes_and_deterministic(self):
        k = AgentKeyPair.generate("a")
        assert len(k.sign(b"hello")) == 64 and k.sign(b"hello") == k.sign(b"hello") and k.sign(b"hello") != k.sign(b"world")

    def test_a_key_round_trips_through_an_unencrypted_file(self, tmp_path):
        k = AgentKeyPair.generate("a")
        path = k.save(tmp_path / "a.pem")
        assert b"BEGIN PRIVATE KEY" in path.read_bytes()
        assert AgentKeyPair.load("a", path) == k

    @pytest.mark.parametrize("pw", ["s3cret", b"s3cret"])
    def test_a_key_round_trips_through_an_encrypted_file(self, tmp_path, pw):
        k = AgentKeyPair.generate("a")
        path = k.save(tmp_path / "a.pem", passphrase=pw)
        assert b"ENCRYPTED PRIVATE KEY" in path.read_bytes()
        assert AgentKeyPair.load("a", path, pw) == k

    def test_a_wrong_or_missing_passphrase_cannot_load_the_key(self, tmp_path):
        path = AgentKeyPair.generate("a").save(tmp_path / "a.pem", passphrase="right")
        for pw in ("wrong", None, ""):
            with pytest.raises(KeyManagementError, match="wrong passphrase or corrupt"):
                AgentKeyPair.load("a", path, pw)

    def test_saving_never_overwrites_a_key_by_accident(self, tmp_path):
        k1, k2 = AgentKeyPair.generate("a"), AgentKeyPair.generate("a")
        path = k1.save(tmp_path / "a.pem")
        with pytest.raises(KeyManagementError, match="already exists"):
            k2.save(path)
        assert AgentKeyPair.load("a", path) == k1
        k2.save(path, overwrite=True)
        assert AgentKeyPair.load("a", path) == k2

    @pytest.mark.skipif(os.name == "nt", reason="POSIX file modes are not meaningful on Windows")
    def test_key_files_are_owner_only(self, tmp_path):
        path = AgentKeyPair.generate("a").save(tmp_path / "a.pem")
        assert path.stat().st_mode & 0o077 == 0

    def test_files_that_are_not_ed25519_keys_are_refused(self, tmp_path):
        rsa_pem = rsa.generate_private_key(65537, 2048).private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
        ec_pem = ec.generate_private_key(ec.SECP256R1()).private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
        for name, data in (("rsa", rsa_pem), ("ec", ec_pem), ("junk", b"not a key")):
            (tmp_path / f"{name}.pem").write_bytes(data)
            with pytest.raises(KeyManagementError):
                AgentKeyPair.load("a", tmp_path / f"{name}.pem")

    def test_private_key_material_never_appears_in_repr_str_or_logs(self, caplog):
        caplog.set_level(logging.DEBUG)
        k = AgentKeyPair.generate("a")
        raw = k._private.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption())
        text = repr(k) + str(k) + f"{k}" + format(k)
        for secret in (raw.hex(), base64.b64encode(raw).decode()):
            assert secret not in text and secret not in caplog.text
        assert "fingerprint=" in repr(k) and "PRIVATE" not in text

    def test_a_key_pair_cannot_be_pickled(self):
        with pytest.raises(TypeError, match="cannot be pickled"):
            pickle.dumps(AgentKeyPair.generate("a"))

    def test_public_key_parsing_is_strict(self):
        k = AgentKeyPair.generate("a")
        assert public_key_from_b64(k.public_key_b64)
        for bad in ("!!!", "", base64.b64encode(b"short").decode(), base64.b64encode(b"x" * 33).decode()):
            with pytest.raises(KeyManagementError):
                public_key_from_b64(bad)
        with pytest.raises(KeyManagementError):
            public_key_from_bytes(b"x" * 31)

    def test_equality_and_hash_use_identity_and_public_key(self):
        a = AgentKeyPair.generate("a")
        assert a == a and a != AgentKeyPair.generate("a") and len({a, a}) == 1


# ══ Canonical form and signing ═══════════════════════════════════════════════

class TestSigning:
    def test_a_signed_message_verifies(self, keys, registry):
        signed = sign_envelope(msg(), keys["coordinator"])
        result = verify_envelope(signed, registry)
        assert result.ok and result.code == "ok" and result.agent.agent_id == "coordinator" and bool(result)

    def test_signing_returns_a_copy_and_leaves_the_original_unsigned(self, keys):
        original = msg()
        signed = sign_envelope(original, keys["coordinator"])
        assert original.signature is None and signed is not original
        assert len(base64.b64decode(signed.signature, validate=True)) == 64

    def test_you_cannot_sign_on_someone_elses_behalf(self, keys):
        with pytest.raises(SigningError, match="cannot sign as"):
            sign_envelope(msg(sender="coordinator"), keys["mallory"])

    def test_canonical_bytes_ignore_the_signature_and_payload_key_order(self, keys):
        a = msg(payload={"x": 1, "y": {"b": 2, "a": 1}}, message_id="m1", timestamp=NOW)
        b = msg(payload={"y": {"a": 1, "b": 2}, "x": 1}, message_id="m1", timestamp=NOW)
        assert canonical_bytes(a) == canonical_bytes(b)
        assert canonical_bytes(sign_envelope(a, keys["coordinator"])) == canonical_bytes(a)

    def test_the_domain_prefix_is_part_of_what_is_signed(self, keys, registry):
        m = msg()
        raw_without_domain = keys["coordinator"].sign(canonical_bytes(m).split(b"\n", 1)[1])
        forged = m.model_copy(update={"signature": base64.b64encode(raw_without_domain).decode()})
        assert verify_envelope(forged, registry).code == "bad_signature"

    def test_a_signed_message_still_verifies_after_a_json_round_trip(self, keys, registry):
        signed = sign_envelope(msg(payload={"f": 0.1 + 0.2, "i": 10 ** 12, "s": "₹ naïve — ✓", "n": None, "l": [1, 2.5, "x"]}), keys["coordinator"])
        decoded = MessageCodec.decode(MessageCodec.encode(signed))
        assert verify_envelope(decoded, registry).ok

    def test_naive_timestamps_are_treated_as_utc(self, keys, registry):
        naive = msg(timestamp=datetime(2026, 11, 1, 12, 0, 0))
        assert verify_envelope(sign_envelope(naive, keys["coordinator"]), registry).ok

    @pytest.mark.parametrize("field,value", [
        ("message_id", "msg_other"), ("sender", "worker"), ("receiver", "agent_c"),
        ("message_type", MessageType.AWARD), ("task_id", "task_2"), ("schema_version", "9.9"),
        ("timestamp", NOW), ("payload", {"task_type": "evaluate", "product": {"width_cm": 150, "tags": ["a", "b"]}}),
    ])
    def test_changing_any_field_invalidates_the_signature(self, keys, registry, field, value):
        signed = sign_envelope(msg(timestamp=NOW + timedelta(days=1)), keys["coordinator"])
        changed = signed.model_copy(update={field: value})
        result = verify_envelope(changed, registry)
        assert not result.ok and result.code == "bad_signature", field

    def test_a_nested_payload_change_is_detected(self, keys, registry):
        signed = sign_envelope(msg(), keys["coordinator"])
        payload = copy.deepcopy(signed.payload)
        payload["product"]["tags"].append("extra")
        assert verify_envelope(signed.model_copy(update={"payload": payload}), registry).code == "bad_signature"

    def test_adding_or_removing_a_payload_key_is_detected(self, keys, registry):
        signed = sign_envelope(msg(), keys["coordinator"])
        assert verify_envelope(signed.model_copy(update={"payload": {**signed.payload, "x": 1}}), registry).code == "bad_signature"
        assert verify_envelope(signed.model_copy(update={"payload": {}}), registry).code == "bad_signature"

    def test_a_flipped_signature_bit_is_detected(self, keys, registry):
        signed = sign_envelope(msg(), keys["coordinator"])
        raw = bytearray(base64.b64decode(signed.signature))
        raw[10] ^= 0x01
        bad = signed.model_copy(update={"signature": base64.b64encode(bytes(raw)).decode()})
        assert verify_envelope(bad, registry).code == "bad_signature"

    def test_a_signature_from_another_trusted_agent_is_not_accepted_for_this_sender(self, keys, registry):
        m = msg(sender="coordinator")
        stolen = m.model_copy(update={"signature": base64.b64encode(keys["worker"].sign(canonical_bytes(m))).decode()})
        assert verify_envelope(stolen, registry).code == "bad_signature"

    @pytest.mark.parametrize("signature,code", [
        (None, "unsigned"), ("", "unsigned"), ("!!!not base64!!!", "malformed_signature"),
        (base64.b64encode(b"short").decode(), "malformed_signature"), (base64.b64encode(b"x" * 65).decode(), "malformed_signature"),
    ])
    def test_unsigned_and_malformed_signatures(self, registry, signature, code):
        assert verify_envelope(msg(signature=signature), registry).code == code

    def test_unknown_and_revoked_senders(self, keys, registry):
        assert verify_envelope(sign_envelope(msg(sender="mallory"), keys["mallory"]), registry).code == "unknown_sender"
        signed = sign_envelope(msg(sender="worker", mtype=MessageType.INFORM), keys["worker"])
        assert verify_envelope(signed, registry).ok
        registry.revoke("worker")
        assert verify_envelope(signed, registry).code == "revoked_sender"

    def test_the_sender_key_comes_only_from_the_registry(self, keys, registry):
        # An envelope has no key field; extra data cannot smuggle one in.
        m = msg(sender="coordinator", payload={"public_key": keys["mallory"].public_key_b64})
        forged = m.model_copy(update={"signature": base64.b64encode(keys["mallory"].sign(canonical_bytes(m))).decode()})
        assert verify_envelope(forged, registry).code == "bad_signature"

    def test_non_finite_numbers_cannot_be_signed(self, keys):
        with pytest.raises(SigningError, match="canonicalised"):
            sign_envelope(msg(payload={"x": float("nan")}), keys["coordinator"])
        with pytest.raises(SigningError):
            sign_envelope(msg(payload={"x": float("inf")}), keys["coordinator"])

    def test_after_key_rotation_old_signatures_fail(self, keys, registry):
        signed = sign_envelope(msg(), keys["coordinator"])
        new = AgentKeyPair.generate("coordinator")
        registry.add("coordinator", new, ["coordinator"], replace=True)
        assert verify_envelope(signed, registry).code == "bad_signature"
        assert verify_envelope(sign_envelope(msg(), new), registry).ok


# ══ Registry ═════════════════════════════════════════════════════════════════

class TestRegistry:
    def test_add_get_contains_len(self, keys):
        r = TrustedAgentRegistry()
        agent = r.add("a", keys["worker"], ["worker"])
        assert "a" in r and r.get("a") == agent and len(r) == 1 and agent.roles == frozenset({"worker"}) and not agent.revoked
        assert r.get("missing") is None and r.get(5) is None and "missing" not in r

    def test_public_keys_can_be_given_as_pair_bytes_or_base64(self, keys):
        r = TrustedAgentRegistry()
        a = keys["coordinator"]
        for name, key in (("x", a), ("y", keys["worker"].public_key_bytes), ("z", keys["observer"].public_key_b64)):
            r.add(name, key)
        assert r.get("y").public_key_b64 == keys["worker"].public_key_b64

    def test_invalid_keys_are_refused(self):
        r = TrustedAgentRegistry()
        for bad in (b"short", "!!!", "", 5, None):
            with pytest.raises((RegistryError, KeyManagementError)):
                r.add("a", bad)

    def test_re_registering_needs_an_explicit_rotation(self, keys):
        r = TrustedAgentRegistry()
        r.add("a", keys["worker"])
        with pytest.raises(RegistryError, match="already registered"):
            r.add("a", keys["observer"])
        r.add("a", keys["observer"], replace=True)
        assert r.get("a").public_key_b64 == keys["observer"].public_key_b64

    def test_one_key_cannot_be_trusted_under_two_ids(self, keys):
        r = TrustedAgentRegistry()
        r.add("a", keys["worker"])
        with pytest.raises(RegistryError, match="already registered for 'a'"):
            r.add("b", keys["worker"])

    def test_revoke_and_remove(self, keys):
        r = TrustedAgentRegistry()
        r.add("a", keys["worker"])
        assert r.revoke("a") and r.get("a").revoked and not r.revoke("nobody")
        assert r.remove("a") and r.get("a") is None and not r.remove("a")

    def test_json_round_trip_keeps_public_data_only(self, keys):
        r = TrustedAgentRegistry()
        r.add("a", keys["worker"], ["worker", "extra"])
        r.add("b", keys["observer"], ["observer"])
        r.revoke("b")
        text = r.to_json()
        for pair in keys.values():
            raw = pair._private.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption())
            assert raw.hex() not in text and base64.b64encode(raw).decode() not in text
        assert "PRIVATE" not in text
        again = TrustedAgentRegistry.from_json(text)
        assert again.get("a").roles == frozenset({"worker", "extra"}) and again.get("b").revoked and len(again) == 2

    @pytest.mark.parametrize("text", ["not json", "{}", '{"agents": [{"agent_id": "a"}]}', '{"agents": [{"agent_id": "a", "public_key": "!!"}]}'])
    def test_corrupt_registry_documents_are_refused(self, text):
        with pytest.raises(RegistryError):
            TrustedAgentRegistry.from_json(text)

    def test_concurrent_registration_is_safe(self):
        r = TrustedAgentRegistry()
        pairs = [AgentKeyPair.generate(f"a{i}") for i in range(40)]
        threads = [threading.Thread(target=lambda p=p: r.add(p.agent_id, p)) for p in pairs]
        [t.start() for t in threads]; [t.join() for t in threads]
        assert len(r) == 40


# ══ Authorization ════════════════════════════════════════════════════════════

class TestPolicy:
    policy = AuthorizationPolicy()

    def agent(self, registry, name):
        return registry.get(name)

    def test_the_coordinator_may_send_everything(self, registry):
        for t in MessageType:
            assert self.policy.check(self.agent(registry, "coordinator"), msg(mtype=t)).allowed, t

    @pytest.mark.parametrize("mtype", sorted(COMMAND_TYPES))
    def test_workers_and_observers_may_not_send_commands(self, registry, mtype):
        for name in ("worker", "observer"):
            decision = self.policy.check(self.agent(registry, name), msg(sender=name, mtype=MessageType(mtype)))
            assert not decision.allowed and "command" in decision.reason and name in decision.reason

    @pytest.mark.parametrize("mtype", [MessageType.INFORM, MessageType.ACK, MessageType.PROPOSAL, MessageType.INFORM_RESULT])
    def test_workers_may_report(self, registry, mtype):
        assert self.policy.check(self.agent(registry, "worker"), msg(sender="worker", mtype=mtype)).allowed

    def test_observers_may_only_ask_and_acknowledge(self, registry):
        o = self.agent(registry, "observer")
        assert all(self.policy.check(o, msg(mtype=t)).allowed for t in (MessageType.QUERY, MessageType.INFORM, MessageType.ACK))
        assert not self.policy.check(o, msg(mtype=MessageType.PROPOSAL)).allowed

    def test_agents_without_roles_or_with_unknown_roles_may_send_nothing(self, keys):
        r = TrustedAgentRegistry()
        r.add("norole", keys["worker"])
        r.add("weird", keys["observer"], ["wizard"])
        for name in ("norole", "weird"):
            assert not any(self.policy.check(r.get(name), msg(mtype=t)).allowed for t in MessageType)

    def test_roles_combine(self, keys):
        r = TrustedAgentRegistry()
        r.add("dual", keys["worker"], ["worker", "observer"])
        assert self.policy.check(r.get("dual"), msg(mtype=MessageType.QUERY)).allowed

    def test_task_types_can_be_restricted_per_role(self, keys):
        policy = AuthorizationPolicy(permissions={"planner": frozenset({MessageType.TASK_REQUEST.value})},
                                     allowed_tasks={"planner": frozenset({"evaluate_furniture_fit"})})
        r = TrustedAgentRegistry()
        r.add("p", keys["worker"], ["planner"])
        ok = policy.check(r.get("p"), msg(payload={"task_type": "evaluate_furniture_fit"}))
        bad = policy.check(r.get("p"), msg(payload={"task_type": "delete_everything"}))
        none = policy.check(r.get("p"), msg(payload={}))
        assert ok.allowed and not bad.allowed and "delete_everything" in bad.reason and not none.allowed

    def test_an_unrestricted_role_can_still_request_any_task(self, registry):
        assert self.policy.check(self.agent(registry, "coordinator"), msg(payload={"task_type": "anything"})).allowed


# ══ Replay protection ════════════════════════════════════════════════════════

class FakeClock:
    def __init__(self, now=NOW):
        self.now = now

    def __call__(self):
        return self.now


class TestReplayGuard:
    def guard(self, clock=None, **kw):
        return ReplayGuard(clock=clock or FakeClock(), **kw)

    def test_a_fresh_message_is_accepted_once(self):
        g = self.guard()
        m = msg(timestamp=NOW)
        assert g.check(m)[0] is True
        accepted, code, _ = g.check(m)
        assert not accepted and code == "replay"

    def test_a_message_older_than_the_window_is_stale(self):
        accepted, code, detail = self.guard(window_seconds=60).check(msg(timestamp=NOW - timedelta(seconds=61)))
        assert not accepted and code == "stale" and "61" in detail

    def test_a_message_from_the_far_future_is_refused_but_small_skew_is_tolerated(self):
        g = self.guard(max_skew_seconds=30)
        assert g.check(msg(timestamp=NOW + timedelta(seconds=29)))[0]
        accepted, code, _ = g.check(msg(timestamp=NOW + timedelta(seconds=31)))
        assert not accepted and code == "from_the_future"

    def test_the_window_edge_is_inclusive(self):
        assert self.guard(window_seconds=60).check(msg(timestamp=NOW - timedelta(seconds=60)))[0]

    def test_the_same_id_from_different_senders_is_not_a_replay(self):
        g = self.guard()
        assert g.check(msg(sender="a", message_id="m1", timestamp=NOW))[0]
        assert g.check(msg(sender="b", message_id="m1", timestamp=NOW))[0]
        assert not g.check(msg(sender="a", message_id="m1", timestamp=NOW))[0]

    def test_old_entries_are_forgotten_and_memory_is_bounded(self):
        clock = FakeClock()
        g = ReplayGuard(window_seconds=60, max_entries=5, clock=clock)
        for i in range(20):
            g.check(msg(message_id=f"m{i}", timestamp=NOW))
        assert len(g) == 5
        clock.now = NOW + timedelta(seconds=120)
        g.check(msg(message_id="fresh", timestamp=clock.now))
        assert len(g) == 1

    def test_a_forgotten_id_still_cannot_be_replayed_because_it_is_stale(self):
        clock = FakeClock()
        g = ReplayGuard(window_seconds=60, clock=clock)
        m = msg(timestamp=NOW)
        assert g.check(m)[0]
        clock.now = NOW + timedelta(seconds=120)
        assert g.check(m)[1] == "stale"

    def test_naive_timestamps_are_handled(self):
        assert self.guard().check(msg(timestamp=datetime(2026, 11, 1, 12, 0, 0)))[0]

    @pytest.mark.parametrize("kwargs", [{"window_seconds": 0}, {"window_seconds": -1}, {"max_skew_seconds": -1}, {"max_entries": 0}])
    def test_invalid_settings(self, kwargs):
        with pytest.raises(ValueError):
            ReplayGuard(**kwargs)

    def test_concurrent_duplicates_are_accepted_exactly_once(self):
        g = ReplayGuard()
        m = msg()
        results = []
        threads = [threading.Thread(target=lambda: results.append(g.check(m)[0])) for _ in range(30)]
        [t.start() for t in threads]; [t.join() for t in threads]
        assert results.count(True) == 1


# ══ The secure router ════════════════════════════════════════════════════════

class Harness:
    """A SecureMessageRouter around a counting inner router."""

    def __init__(self, keys, registry, **kw):
        self.calls = []
        self.inner = MessageRouter(agent_id="agent_b")
        for t in (MessageType.TASK_REQUEST, MessageType.STATE_SYNC, MessageType.INFORM, MessageType.AWARD):
            self.inner.register(t, self.handler)
        self.audit = AuditLog()
        self.router = SecureMessageRouter(keys["agent_b"], self.inner, registry, audit=self.audit, **kw)

    def handler(self, message):
        self.calls.append((message.sender, message.message_type))
        return MessageEnvelope(sender="someone_else", receiver="ignored", message_type=MessageType.INFORM_RESULT,
                               task_id=message.task_id, payload={"result": {"ok": True}})


@pytest.fixture()
def h(keys, registry):
    return Harness(keys, registry)


class TestSecureRouter:
    def test_an_authentic_authorized_message_reaches_the_handler_and_the_reply_is_signed(self, h, keys, registry):
        reply = h.router.route(sign_envelope(msg(), keys["coordinator"]))
        assert h.calls == [("coordinator", "task_request")] and reply.message_type == "inform_result"
        assert reply.sender == "agent_b" and reply.receiver == "coordinator"      # never the handler's choice of identity
        assert verify_envelope(reply, registry).ok and h.audit.summary() == {"accepted:ok": 1}

    def test_a_handler_that_returns_nothing_gives_no_reply(self, keys, registry):
        inner = MessageRouter(agent_id="agent_b")
        inner.register(MessageType.INFORM, lambda m: None)
        router = SecureMessageRouter(keys["agent_b"], inner, registry)
        assert router.route(sign_envelope(msg(sender="worker", mtype=MessageType.INFORM), keys["worker"])) is None

    def test_a_failing_handler_still_produces_a_signed_failure(self, keys, registry):
        inner = MessageRouter(agent_id="agent_b")
        inner.register(MessageType.TASK_REQUEST, lambda m: 1 / 0)
        reply = SecureMessageRouter(keys["agent_b"], inner, registry).route(sign_envelope(msg(), keys["coordinator"]))
        assert reply.message_type == "failure" and verify_envelope(reply, registry).ok

    def rejects(self, h, registry, message, code, public):
        before = len(h.calls)
        reply = h.router.route(message)
        assert len(h.calls) == before, "a handler ran for a rejected message"
        assert reply.message_type == "failure" and reply.payload == {"reason": public}
        assert verify_envelope(reply, registry).ok, "the rejection itself must be signed"
        assert h.audit.rejected()[-1].code == code
        return reply

    def test_unsigned(self, h, registry):
        self.rejects(h, registry, msg(), "unsigned", "authentication_failed")

    def test_modified_in_transit(self, h, keys, registry):
        signed = sign_envelope(msg(), keys["coordinator"])
        evil = signed.model_copy(update={"payload": {"task_type": "evaluate", "product": {"width_cm": 1}}})
        self.rejects(h, registry, evil, "bad_signature", "authentication_failed")

    def test_impersonation(self, h, keys, registry):
        m = msg(sender="coordinator")
        forged = m.model_copy(update={"signature": base64.b64encode(keys["mallory"].sign(canonical_bytes(m))).decode()})
        self.rejects(h, registry, forged, "bad_signature", "authentication_failed")

    def test_unknown_sender_even_with_a_valid_signature(self, h, keys, registry):
        self.rejects(h, registry, sign_envelope(msg(sender="mallory"), keys["mallory"]), "unknown_sender", "authentication_failed")

    def test_revoked_sender(self, h, keys, registry):
        registry.revoke("coordinator")
        self.rejects(h, registry, sign_envelope(msg(), keys["coordinator"]), "revoked_sender", "authentication_failed")

    @pytest.mark.parametrize("mtype", sorted(COMMAND_TYPES))
    def test_a_trusted_worker_cannot_issue_any_command(self, h, keys, registry, mtype):
        signed = sign_envelope(msg(sender="worker", mtype=MessageType(mtype)), keys["worker"])
        self.rejects(h, registry, signed, "not_authorized", "not_authorized")

    def test_a_trusted_worker_can_still_report(self, h, keys):
        reply = h.router.route(sign_envelope(msg(sender="worker", mtype=MessageType.INFORM), keys["worker"]))
        assert reply.message_type == "inform_result" and h.calls == [("worker", "inform")]

    def test_replay(self, h, keys, registry):
        signed = sign_envelope(msg(), keys["coordinator"])
        assert h.router.route(signed).message_type == "inform_result"
        self.rejects(h, registry, signed, "replay", "message_rejected")

    def test_stale_and_future_messages(self, h, keys, registry):
        old = sign_envelope(msg(timestamp=datetime.now(timezone.utc) - timedelta(hours=1)), keys["coordinator"])
        future = sign_envelope(msg(timestamp=datetime.now(timezone.utc) + timedelta(hours=1)), keys["coordinator"])
        self.rejects(h, registry, old, "stale", "message_rejected")
        self.rejects(h, registry, future, "from_the_future", "message_rejected")

    def test_a_message_addressed_to_someone_else(self, h, keys, registry):
        self.rejects(h, registry, sign_envelope(msg(receiver="agent_c"), keys["coordinator"]), "wrong_receiver", "message_rejected")

    def test_a_forged_message_cannot_poison_the_replay_cache(self, h, keys):
        legit = sign_envelope(msg(message_id="msg_fixed"), keys["coordinator"])
        forged = msg(message_id="msg_fixed").model_copy(update={"signature": base64.b64encode(b"x" * 64).decode()})
        h.router.route(forged)                                   # an attacker guesses the id first
        assert h.router.route(legit).message_type == "inform_result"

    def test_the_peer_learns_only_a_coarse_reason(self, h, keys):
        reply = h.router.route(sign_envelope(msg(sender="mallory"), keys["mallory"]))
        assert set(reply.payload) == {"reason"} and "mallory" not in str(reply.payload) and "trusted" not in str(reply.payload)

    def test_the_audit_log_explains_each_rejection_without_the_payload(self, h, keys):
        secret = {"task_type": "evaluate", "api_token": "TOP-SECRET-VALUE"}
        h.router.route(sign_envelope(msg(sender="mallory", payload=secret), keys["mallory"]))
        event = h.audit.rejected()[-1]
        assert event.code == "unknown_sender" and "mallory" in event.format()
        assert "TOP-SECRET-VALUE" not in event.format() and "TOP-SECRET-VALUE" not in repr(event)

    def test_decisions_are_written_to_the_security_log(self, h, keys, caplog):
        caplog.set_level(logging.INFO, logger="smartnest.security.audit")
        h.router.route(sign_envelope(msg(), keys["coordinator"]))
        h.router.route(msg())
        levels = {r.levelname: r.getMessage() for r in caplog.records if r.name == "smartnest.security.audit"}
        assert "ACCEPTED ok" in levels["INFO"] and "REJECTED unsigned" in levels["WARNING"]

    def test_the_audit_log_is_bounded(self):
        log = AuditLog(max_events=3)
        for i in range(10):
            log.record("rejected", "x", "", msg(message_id=f"m{i}"))
        assert len(log.events) == 3 and log.events[-1].message_id == "m9"

    def test_a_flood_of_attacks_never_reaches_a_handler(self, h, keys):
        for i in range(200):
            h.router.route(msg(message_id=f"a{i}"))
            h.router.route(sign_envelope(msg(sender="mallory", message_id=f"b{i}"), keys["mallory"]))
        assert h.calls == [] and h.audit.summary() == {"rejected:unsigned": 200, "rejected:unknown_sender": 200}

    def test_a_custom_policy_is_honoured(self, keys, registry):
        policy = AuthorizationPolicy(permissions={"coordinator": frozenset({MessageType.INFORM.value})})
        h2 = Harness(keys, registry, policy=policy)
        assert h2.router.route(sign_envelope(msg(), keys["coordinator"])).payload == {"reason": "not_authorized"}
        assert h2.router.route(sign_envelope(msg(mtype=MessageType.INFORM), keys["coordinator"])).message_type == "inform_result"


# ══ Real sockets ═════════════════════════════════════════════════════════════

@pytest.fixture()
def live(keys, registry):
    h = Harness(keys, registry)
    server = RemoteAgentServer(router=h.router, host="127.0.0.1", port=0, agent_id="agent_b")
    server.start()
    client_registry = TrustedAgentRegistry()
    client_registry.add("agent_b", keys["agent_b"], ["worker"])
    yield h, server, client_registry
    server.stop(timeout=3.0)


def exchange(server, message):
    with SocketTransport.connect(server.host, server.port, timeout=5.0) as t:
        t.send_message(message)
        return t.receive_message()


class TestOverTcp:
    def test_a_signed_command_is_executed_and_the_signed_reply_verifies(self, live, keys):
        h, server, creg = live
        with SecureChannel(keys["coordinator"], creg, server.host, server.port, expected_server="agent_b") as ch:
            reply = ch.request(msg())
        assert reply.trusted and reply.message.message_type == "inform_result" and h.calls == [("coordinator", "task_request")]

    def test_tampering_unsigned_impersonation_and_replay_are_all_blocked_over_the_wire(self, live, keys):
        h, server, _ = live
        signed = sign_envelope(msg(), keys["coordinator"])
        tampered = signed.model_copy(update={"payload": {"task_type": "evil"}})
        forged = msg().model_copy(update={"signature": base64.b64encode(keys["mallory"].sign(canonical_bytes(msg()))).decode()})
        for attack in (tampered, msg(), forged, sign_envelope(msg(sender="mallory"), keys["mallory"])):
            assert exchange(server, attack).payload == {"reason": "authentication_failed"}
        assert exchange(server, signed).message_type == "inform_result"
        assert exchange(server, signed).payload == {"reason": "message_rejected"}
        assert h.calls == [("coordinator", "task_request")]

    def test_concurrent_legitimate_clients_all_succeed_while_an_attacker_is_blocked(self, live, keys):
        h, server, creg = live
        results, errors = [], []

        def good(i):
            try:
                with SecureChannel(keys["coordinator"], creg, server.host, server.port, expected_server="agent_b") as ch:
                    results.append(ch.request(msg(message_id=f"good{i}")).trusted)
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        def bad():
            for _ in range(5):
                exchange(server, msg())

        threads = [threading.Thread(target=good, args=(i,)) for i in range(10)] + [threading.Thread(target=bad) for _ in range(3)]
        [t.start() for t in threads]; [t.join() for t in threads]
        assert not errors and results == [True] * 10 and len(h.calls) == 10

    def test_the_server_keeps_working_after_attacks(self, live, keys):
        h, server, creg = live
        exchange(server, msg())
        with SecureChannel(keys["coordinator"], creg, server.host, server.port, expected_server="agent_b") as ch:
            assert ch.request(msg()).trusted


class RogueServer:
    """A server that replies with whatever it is told, to test that clients verify replies."""

    def __init__(self, make_reply):
        self.make_reply = make_reply
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(1)
        self.port = self.sock.getsockname()[1]
        threading.Thread(target=self.serve, daemon=True).start()

    def serve(self):
        conn, _ = self.sock.accept()
        t = SocketTransport(conn, timeout=5.0)
        try:
            request = t.receive_message()
            t.send_message(self.make_reply(request))
        except Exception:
            pass
        finally:
            t.close()
            self.sock.close()


class TestReplyVerification:
    def run(self, keys, creg, make_reply, expected="agent_b"):
        rogue = RogueServer(make_reply)
        with SecureChannel(keys["coordinator"], creg, "127.0.0.1", rogue.port, expected_server=expected, timeout=5.0) as ch:
            return ch.request(msg())

    def creg(self, keys, *extra):
        r = TrustedAgentRegistry()
        r.add("agent_b", keys["agent_b"], ["worker"])
        for name in extra:
            r.add(name, keys[name], ["worker"])
        return r

    def reply(self, request, **over):
        data = dict(sender="agent_b", receiver="coordinator", message_type=MessageType.INFORM_RESULT, task_id=request.task_id, payload={"ok": 1})
        data.update(over)
        return MessageEnvelope(**data)

    def test_an_unsigned_reply_is_not_trusted(self, keys):
        r = self.run(keys, self.creg(keys), lambda req: self.reply(req))
        assert not r.trusted and r.verification.code == "unsigned"

    def test_a_reply_signed_by_the_wrong_key_is_not_trusted(self, keys):
        r = self.run(keys, self.creg(keys), lambda req: sign_envelope(self.reply(req, sender="mallory"), keys["mallory"]).model_copy(update={"sender": "agent_b"}))
        assert not r.trusted and r.verification.code == "bad_signature"

    def test_a_reply_modified_after_signing_is_not_trusted(self, keys):
        r = self.run(keys, self.creg(keys), lambda req: sign_envelope(self.reply(req), keys["agent_b"]).model_copy(update={"payload": {"ok": 0}}))
        assert not r.trusted and r.verification.code == "bad_signature"

    def test_a_genuine_reply_from_the_wrong_agent_is_not_accepted(self, keys):
        creg = self.creg(keys, "worker")
        r = self.run(keys, creg, lambda req: sign_envelope(self.reply(req, sender="worker"), keys["worker"]))
        assert not r.trusted and r.verification.code == "unexpected_sender"

    def test_a_reply_addressed_to_someone_else_is_not_accepted(self, keys):
        r = self.run(keys, self.creg(keys), lambda req: sign_envelope(self.reply(req, receiver="worker"), keys["agent_b"]))
        assert not r.trusted and r.verification.code == "wrong_receiver"

    def test_a_stale_reply_is_not_accepted(self, keys):
        old = datetime.now(timezone.utc) - timedelta(hours=1)
        r = self.run(keys, self.creg(keys), lambda req: sign_envelope(self.reply(req, timestamp=old), keys["agent_b"]))
        assert not r.trusted and r.verification.code == "stale"

    def test_a_correct_reply_is_trusted(self, keys):
        r = self.run(keys, self.creg(keys), lambda req: sign_envelope(self.reply(req), keys["agent_b"]))
        assert r.trusted and r.message.payload == {"ok": 1}


# ══ Isolation ════════════════════════════════════════════════════════════════

class TestIsolation:
    ROOT = pathlib.Path(__file__).resolve().parents[1] / "src" / "smartnest"

    @staticmethod
    def imports(path: pathlib.Path) -> set[str]:
        found = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module:
                found.add(node.module)
            elif isinstance(node, ast.Import):
                found.update(a.name for a in node.names)
        return found

    def test_the_security_layer_depends_only_on_core_and_the_router_transport_interfaces(self):
        allowed = ("smartnest.security", "smartnest.core", "smartnest.communication.message_router",
                   "smartnest.communication.socket_transport")
        for path in (self.ROOT / "security").glob("*.py"):
            for name in self.imports(path):
                if name.startswith("smartnest"):
                    assert name.startswith(allowed), (path.name, name)

    #: The only module outside ``security`` that may use it: the furnishing agents' signed message bus (composition).
    ALLOWED_USERS = {("furnishing", "activity.py")}

    def test_no_other_layer_depends_on_the_security_package(self):
        for path in self.ROOT.rglob("*.py"):
            parts = path.relative_to(self.ROOT).parts
            if "security" in parts or parts in self.ALLOWED_USERS:
                continue
            assert not any(n.startswith("smartnest.security") for n in self.imports(path)), path.name

    def test_the_one_allowed_user_is_really_the_only_one(self):
        users = {path.relative_to(self.ROOT).parts for path in self.ROOT.rglob("*.py")
                 if "security" not in path.relative_to(self.ROOT).parts
                 and any(n.startswith("smartnest.security") for n in self.imports(path))}
        assert users == self.ALLOWED_USERS

    def test_the_security_layer_never_uses_non_cryptographic_randomness_or_weak_hashes(self):
        for path in (self.ROOT / "security").glob("*.py"):
            assert "random" not in self.imports(path), path.name        # keys come from the OS secure source via `cryptography`
            source = path.read_text(encoding="utf-8").lower()
            assert "md5" not in source and "sha1" not in source, path.name


# ══ The demonstration ════════════════════════════════════════════════════════

def test_the_demonstration_runs_clean(capsys):
    import security_run
    assert security_run.main() == 0
    out = capsys.readouterr().out
    assert "All checks passed." in out and "[FAIL]" not in out
    for code in ("bad_signature", "unsigned", "unknown_sender", "not_authorized", "replay", "wrong_receiver", "revoked_sender"):
        assert f"REJECTED {code}" in out
