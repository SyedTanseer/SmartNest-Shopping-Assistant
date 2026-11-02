"""Errors raised by the security layer.

Messages never contain key material.
"""

from __future__ import annotations


class SecurityError(Exception):
    """Base class for every error raised by the security layer."""


class KeyManagementError(SecurityError):
    """A key could not be generated, saved, loaded or parsed."""


class SigningError(SecurityError):
    """A message could not be signed (for example, it names a different sender)."""


class RegistryError(SecurityError):
    """The trusted-agent registry rejected an operation."""
