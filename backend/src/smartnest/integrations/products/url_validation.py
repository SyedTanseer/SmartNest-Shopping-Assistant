"""URL validation and provenance for product links.

SmartNest never shows a product link it cannot trace to a retrieved
source.  A listing's URL is accepted only if

    1. it is a well-formed public ``https`` URL (no credentials, no IP
       literal, no localhost / private names), and
    2. it matches a URL that the search/grounding metadata of the *same
       response* actually reported (:class:`AllowedUrls`).

Google Search grounding reports each source as a
``vertexaisearch.cloud.google.com/grounding-api-redirect/...`` link.
:class:`RedirectResolver` turns such a link into the real page URL by
reading the redirect's ``Location`` header — one request to Google's own
redirect host, never to the retailer — and only ever for that host.

Network helpers are behind small protocols so tests inject fakes.
"""

from __future__ import annotations

import hashlib
import ipaddress
import logging
import re
import urllib.error
import urllib.request
from typing import Protocol
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit

logger = logging.getLogger(__name__)

MAX_URL_LENGTH = 2048
#: Hosts whose links are redirects that may be resolved (and only these).
REDIRECT_HOSTS = frozenset({"vertexaisearch.cloud.google.com"})

_TRACKING_PARAMS = re.compile(r"^(utm_.*|fbclid|gclid|msclkid|ref|ref_|tag|affid|srsltid)$", re.IGNORECASE)
_BAD_HOSTS = ("localhost",)
_BAD_SUFFIXES = (".local", ".internal", ".localhost", ".lan", ".home", ".corp")


def check_url(url: object) -> str | None:
    """Return why ``url`` is unusable as a public product/image link, or ``None`` if it is fine."""
    if not isinstance(url, str) or not url.strip():
        return "URL is missing"
    if url != url.strip() or any(c.isspace() for c in url):
        return "URL contains whitespace"
    if len(url) > MAX_URL_LENGTH:
        return "URL is too long"
    try:
        parts = urlsplit(url)
    except ValueError:
        return "URL is malformed"
    if parts.scheme != "https":
        return "URL must use https"
    if parts.username or parts.password:
        return "URL must not contain credentials"
    host = (parts.hostname or "").lower()
    if not host or "." not in host:
        return "URL has no valid host name"
    try:
        ipaddress.ip_address(host)
        return "URL must use a host name, not an IP address"
    except ValueError:
        pass
    if host in _BAD_HOSTS or host.endswith(_BAD_SUFFIXES):
        return "URL points to a non-public host"
    try:
        parts.port
    except ValueError:
        return "URL has an invalid port"
    return None


def canonical_key(url: str) -> str:
    """A comparison key: lower-case host without ``www.``, path without trailing slash, tracking parameters dropped."""
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").lower().removeprefix("www.")
    path = parts.path.rstrip("/") or "/"
    query = sorted((k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not _TRACKING_PARAMS.match(k))
    return f"{host}{path}" + (f"?{urlencode(query)}" if query else "")


def strip_tracking(url: str) -> str:
    """``url`` without tracking parameters (``utm_*``, ``gclid``, ...) and without a fragment."""
    parts = urlsplit(url)
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                       if not _TRACKING_PARAMS.match(k)])
    return parts._replace(query=query, fragment="").geturl()


def url_fingerprint(url: str) -> str:
    """Short stable hash of a URL's canonical key (used in product ids)."""
    return hashlib.sha1(canonical_key(url).encode("utf-8")).hexdigest()[:8].upper()


def domain_of(url: str) -> str:
    return (urlsplit(url).hostname or "").lower().removeprefix("www.")


class AllowedUrls:
    """The set of source URLs a response actually reported."""

    def __init__(self) -> None:
        self._by_key: dict[str, str] = {}

    def add(self, url: str) -> bool:
        """Add a URL if it passes :func:`check_url`.  Returns whether it was added."""
        if check_url(url) is not None:
            return False
        self._by_key.setdefault(canonical_key(url), strip_tracking(url))
        return True

    def lookup(self, url: str) -> str | None:
        """The reported URL matching ``url`` (ignoring tracking parameters, ``www.``, trailing slash)."""
        if check_url(url) is not None:
            return None
        return self._by_key.get(canonical_key(url))

    def urls(self) -> list[str]:
        return list(self._by_key.values())

    def __len__(self) -> int:
        return len(self._by_key)


# ── Network helpers (injectable) ─────────────────────────────────────────────

class UrlResolver(Protocol):
    def resolve(self, url: str) -> str:
        """The real destination of ``url`` (or ``url`` itself if it is not a redirect link)."""


class ImageVerifier(Protocol):
    def verify(self, url: str) -> bool:
        """Whether ``url`` really serves an image."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # surface the redirect instead of following it
        return None


class RedirectResolver:
    """Resolves Google grounding redirect links by reading the ``Location`` header.

    Only hosts in :data:`REDIRECT_HOSTS` are contacted; the destination
    itself is never requested.  If resolution fails the original link is
    returned unchanged, and the caller will not treat it as a product URL.
    """

    def __init__(self, timeout: float = 8.0, max_hops: int = 4) -> None:
        self._timeout = timeout
        self._max_hops = max_hops
        self._opener = urllib.request.build_opener(_NoRedirect)

    def resolve(self, url: str) -> str:
        current = url
        for _ in range(self._max_hops):
            host = (urlsplit(current).hostname or "").lower()
            if host not in REDIRECT_HOSTS:
                return current
            request = urllib.request.Request(current, headers={"User-Agent": "SmartNest/0.1 (grounding link resolver)"})
            try:
                with self._opener.open(request, timeout=self._timeout) as response:
                    logger.warning("Grounding link did not redirect (HTTP %s)", response.status)
                    return url
            except urllib.error.HTTPError as err:
                location = err.headers.get("Location") if err.headers else None
                if err.code in (301, 302, 303, 307, 308) and location:
                    current = urljoin(current, location)
                    continue
                logger.warning("Could not resolve a grounding link (HTTP %s)", err.code)
                return url
            except Exception as exc:
                logger.warning("Could not resolve a grounding link (%s)", type(exc).__name__)
                return url
        return url


class HttpImageVerifier:
    """Checks with a ``HEAD`` request that an image URL answers 200 with an ``image/*`` type."""

    def __init__(self, timeout: float = 6.0) -> None:
        self._timeout = timeout

    def verify(self, url: str) -> bool:
        if check_url(url) is not None:
            return False
        request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "SmartNest/0.1 (image check)"})
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                return response.status == 200 and response.headers.get_content_type().startswith("image/")
        except Exception:
            return False
