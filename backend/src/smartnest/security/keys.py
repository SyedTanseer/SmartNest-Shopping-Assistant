"""Ed25519 key generation and secure key handling.

Each agent owns one :class:`AgentKeyPair`.  The **private key** is used only to sign; it is never logged, never
included in ``repr``/``str``, never serialised into a message, and is written to disk only through
:meth:`AgentKeyPair.save` (PKCS#8 PEM, optionally encrypted with a passphrase, file mode 0600 where the operating
system supports it).  The **public key** (32 raw bytes, base64) and its **fingerprint** (first 16 hex characters of its
SHA-256) are safe to share and to log.

Keys belong in a git-ignored directory (``backend/keys/``); never commit them.

Honest limits: on Windows ``chmod`` cannot make a file private to one user, so an encrypted key (a passphrase) is the
real protection there.  A signature gives authenticity and integrity, not secrecy: it does not encrypt anything.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from smartnest.security.errors import KeyManagementError

_AGENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:\-]{0,127}$")
PUBLIC_KEY_LENGTH = 32
SIGNATURE_LENGTH = 64


def validate_agent_id(agent_id: object) -> str:
    """An agent id is a short identifier (letters, digits, ``_ . : -``); anything else is refused."""
    if not isinstance(agent_id, str) or not _AGENT_ID.match(agent_id):
        raise KeyManagementError("agent_id must be 1-128 characters: letters, digits, '_', '.', ':' or '-'")
    return agent_id


def public_key_from_bytes(raw: bytes) -> Ed25519PublicKey:
    if not isinstance(raw, (bytes, bytearray)) or len(raw) != PUBLIC_KEY_LENGTH:
        raise KeyManagementError(f"an Ed25519 public key is exactly {PUBLIC_KEY_LENGTH} bytes")
    try:
        return Ed25519PublicKey.from_public_bytes(bytes(raw))
    except Exception:
        raise KeyManagementError("not a valid Ed25519 public key") from None


def public_key_from_b64(text: str) -> Ed25519PublicKey:
    try:
        raw = base64.b64decode(text, validate=True)
    except Exception:
        raise KeyManagementError("the public key is not valid base64") from None
    return public_key_from_bytes(raw)


def public_key_bytes(key: Ed25519PublicKey) -> bytes:
    return key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def fingerprint_of(public_key: bytes) -> str:
    """Short, loggable identifier of a public key."""
    return hashlib.sha256(public_key).hexdigest()[:16]


def verify_raw(public_key: Ed25519PublicKey, signature: bytes, data: bytes) -> bool:
    """Whether ``signature`` is a valid Ed25519 signature of ``data`` (never raises on a bad signature)."""
    if len(signature) != SIGNATURE_LENGTH:
        return False
    try:
        public_key.verify(signature, data)
    except InvalidSignature:
        return False
    return True


class AgentKeyPair:
    """An agent's identity: its id and Ed25519 key pair.  The private key never leaves this object except via ``save``."""

    __slots__ = ("_agent_id", "_private", "_public")

    def __init__(self, agent_id: str, private_key: Ed25519PrivateKey) -> None:
        self._agent_id = validate_agent_id(agent_id)
        if not isinstance(private_key, Ed25519PrivateKey):
            raise KeyManagementError("private_key must be an Ed25519 private key")
        self._private = private_key
        self._public = public_key_bytes(private_key.public_key())

    # ── Creation ─────────────────────────────────────────────────────────

    @classmethod
    def generate(cls, agent_id: str) -> "AgentKeyPair":
        """A fresh random key pair from the operating system's secure random source."""
        return cls(agent_id, Ed25519PrivateKey.generate())

    @classmethod
    def load(cls, agent_id: str, path: str | Path, passphrase: bytes | str | None = None) -> "AgentKeyPair":
        """Load a key saved by :meth:`save`.  A wrong passphrase or a corrupt file raises ``KeyManagementError``."""
        data = Path(path).read_bytes()
        pw = passphrase.encode("utf-8") if isinstance(passphrase, str) else passphrase
        try:
            key = serialization.load_pem_private_key(data, password=pw or None)
        except Exception:
            raise KeyManagementError("could not load the private key (wrong passphrase or corrupt file)") from None
        if not isinstance(key, Ed25519PrivateKey):
            raise KeyManagementError("the file does not contain an Ed25519 private key")
        return cls(agent_id, key)

    # ── Public information (safe to share and log) ───────────────────────

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def public_key_bytes(self) -> bytes:
        return self._public

    @property
    def public_key_b64(self) -> str:
        return base64.b64encode(self._public).decode("ascii")

    @property
    def fingerprint(self) -> str:
        return fingerprint_of(self._public)

    # ── Signing ──────────────────────────────────────────────────────────

    def sign(self, data: bytes) -> bytes:
        """The 64-byte Ed25519 signature of ``data``."""
        return self._private.sign(bytes(data))

    # ── Storage ──────────────────────────────────────────────────────────

    def save(self, path: str | Path, passphrase: bytes | str | None = None, *, overwrite: bool = False) -> Path:
        """Write the private key as PKCS#8 PEM (encrypted if a passphrase is given), mode 0600 where supported.

        Refuses to replace an existing file unless ``overwrite`` is set, so a key is never destroyed by accident.
        """
        target = Path(path)
        if target.exists() and not overwrite:
            raise KeyManagementError(f"{target.name} already exists; pass overwrite=True to replace it")
        pw = passphrase.encode("utf-8") if isinstance(passphrase, str) else passphrase
        encryption = serialization.BestAvailableEncryption(pw) if pw else serialization.NoEncryption()
        pem = self._private.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, encryption)
        target.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0)  # O_BINARY: no newline translation on Windows
        fd = os.open(target, flags, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(pem)
        try:
            os.chmod(target, 0o600)
        except OSError:
            pass  # not supported on every platform; see the module docstring
        return target

    # ── Safe representations ─────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"AgentKeyPair(agent_id={self._agent_id!r}, fingerprint={self.fingerprint})"

    __str__ = __repr__

    def __reduce__(self):  # pragma: no cover - guard against accidentally pickling a private key
        raise TypeError("AgentKeyPair cannot be pickled: private keys must not be serialised")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, AgentKeyPair) and other._agent_id == self._agent_id and other._public == self._public

    def __hash__(self) -> int:
        return hash((self._agent_id, self._public))
