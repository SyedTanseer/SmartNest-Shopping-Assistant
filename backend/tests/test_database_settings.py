"""Tests for database configuration and credential redaction.

All test URIs use obviously fake credentials.
"""

from __future__ import annotations

import pytest

from smartnest.database import (
    ALL_COLLECTIONS,
    Collections,
    DatabaseConfigError,
    DatabaseSettings,
    load_settings,
    redact_text,
)

FAKE_URI = "mongodb+srv://demo_user:fakePass123@cluster.example.mongodb.net/"


def _load(env_text: str | None = None, environ: dict | None = None, tmp_path=None):
    env_file = None
    if env_text is not None:
        env_file = tmp_path / ".env"
        env_file.write_text(env_text, encoding="utf-8")
    elif tmp_path is not None:
        env_file = tmp_path / "missing.env"
    return load_settings(env_file=env_file, environ=environ if environ is not None else {})


# ── Loading ──────────────────────────────────────────────────────────────────

class TestLoading:
    def test_loads_from_env_file(self, tmp_path):
        s = _load(f"MONGODB_URI={FAKE_URI}\nMONGODB_DATABASE=mydb\n", tmp_path=tmp_path)
        assert s.uri == FAKE_URI
        assert s.database == "mydb"

    def test_database_defaults_to_smartnest(self, tmp_path):
        assert _load(f"MONGODB_URI={FAKE_URI}\n", tmp_path=tmp_path).database == "smartnest"

    def test_environment_overrides_file(self, tmp_path):
        s = _load("MONGODB_URI=mongodb://file-host/\nMONGODB_DATABASE=from_file\n",
                  environ={"MONGODB_URI": "mongodb://env-host/", "MONGODB_DATABASE": "from_env"},
                  tmp_path=tmp_path)
        assert s.host == "env-host"
        assert s.database == "from_env"

    def test_works_from_environment_alone(self, tmp_path):
        s = _load(None, environ={"MONGODB_URI": FAKE_URI}, tmp_path=tmp_path)
        assert s.host == "cluster.example.mongodb.net"

    def test_blank_environment_value_falls_back_to_file(self, tmp_path):
        s = _load(f"MONGODB_URI={FAKE_URI}\n", environ={"MONGODB_URI": "  "}, tmp_path=tmp_path)
        assert s.uri == FAKE_URI

    def test_quoted_value_and_bom_are_handled(self, tmp_path):
        env = tmp_path / ".env"
        env.write_bytes(b"\xef\xbb\xbf" + f'MONGODB_URI="{FAKE_URI}"\n'.encode())
        assert load_settings(env_file=env, environ={}).uri == FAKE_URI

    def test_load_does_not_modify_os_environ(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MONGODB_URI", raising=False)
        env = tmp_path / ".env"
        env.write_text(f"MONGODB_URI={FAKE_URI}\n", encoding="utf-8")
        load_settings(env_file=env)
        import os
        assert "MONGODB_URI" not in os.environ

    def test_optional_numeric_settings(self, tmp_path):
        s = _load(f"MONGODB_URI={FAKE_URI}\nMONGODB_MAX_POOL_SIZE=10\n"
                  "MONGODB_SERVER_SELECTION_TIMEOUT_MS=1500\n", tmp_path=tmp_path)
        assert (s.max_pool_size, s.server_selection_timeout_ms) == (10, 1500)

    @pytest.mark.parametrize("name,value", [
        ("MONGODB_MAX_POOL_SIZE", "many"),
        ("MONGODB_MAX_POOL_SIZE", "0"),
        ("MONGODB_SERVER_SELECTION_TIMEOUT_MS", "-5"),
    ])
    def test_invalid_numeric_settings(self, tmp_path, name, value):
        with pytest.raises(DatabaseConfigError, match="integer"):
            _load(f"MONGODB_URI={FAKE_URI}\n{name}={value}\n", tmp_path=tmp_path)


# ── Validation ───────────────────────────────────────────────────────────────

class TestValidation:
    def test_missing_uri(self, tmp_path):
        with pytest.raises(DatabaseConfigError, match="MONGODB_URI is not set"):
            _load("LOG_LEVEL=INFO\n", tmp_path=tmp_path)

    def test_no_env_file_and_no_environment(self, tmp_path):
        with pytest.raises(DatabaseConfigError, match="not set"):
            _load(None, tmp_path=tmp_path)

    @pytest.mark.parametrize("uri", [
        "mongodb+srv://<db_username>:<db_password>@cluster.example.mongodb.net/",
        "<paste your connection string>",
        "mongodb+srv://user:<password>@host/",
    ])
    def test_placeholder_uri_rejected(self, uri):
        with pytest.raises(DatabaseConfigError, match="placeholder"):
            DatabaseSettings(uri=uri)

    @pytest.mark.parametrize("uri", ["http://example.com", "postgres://u:p@h/db", "cluster.example.com"])
    def test_wrong_scheme_rejected(self, uri):
        with pytest.raises(DatabaseConfigError, match="must start with mongodb"):
            DatabaseSettings(uri=uri)

    def test_whitespace_in_uri_rejected(self):
        with pytest.raises(DatabaseConfigError, match="whitespace"):
            DatabaseSettings(uri="mongodb://host/ db")

    @pytest.mark.parametrize("name", ["", "  ", "a/b", "a.b", "a b", "a$b", "x" * 64, 'q"q'])
    def test_invalid_database_names(self, name):
        with pytest.raises(DatabaseConfigError, match="MONGODB_DATABASE"):
            DatabaseSettings(uri=FAKE_URI, database=name)

    def test_pool_sizes_validated(self):
        with pytest.raises(DatabaseConfigError):
            DatabaseSettings(uri=FAKE_URI, max_pool_size=5, min_pool_size=6)
        with pytest.raises(DatabaseConfigError):
            DatabaseSettings(uri=FAKE_URI, server_selection_timeout_ms=0)

    def test_settings_are_immutable(self):
        s = DatabaseSettings(uri=FAKE_URI)
        with pytest.raises(Exception):
            s.database = "other"

    def test_validation_errors_never_contain_the_uri(self):
        bad = "mongodb+srv://demo_user:fakePass123@cluster.example.mongodb.net/ with space"
        with pytest.raises(DatabaseConfigError) as info:
            DatabaseSettings(uri=bad)
        assert "fakePass123" not in str(info.value)
        assert "demo_user" not in str(info.value)


# ── Credential safety ────────────────────────────────────────────────────────

class TestCredentialSafety:
    def test_repr_and_str_hide_credentials(self):
        s = DatabaseSettings(uri=FAKE_URI)
        for text in (repr(s), str(s), f"{s}", f"{s!r}"):
            assert "fakePass123" not in text
            assert "demo_user" not in text

    def test_redacted_uri_and_host(self):
        s = DatabaseSettings(uri=FAKE_URI)
        assert s.host == "cluster.example.mongodb.net"
        assert s.redacted_uri == "mongodb+srv://***@cluster.example.mongodb.net/"

    def test_host_ignores_path_and_options(self):
        assert DatabaseSettings(uri="mongodb://u:p@h1:27017/db?retryWrites=true").host == "h1:27017"
        assert DatabaseSettings(uri="mongodb://localhost").host == "localhost"

    def test_secret_fragments_include_password_forms(self):
        s = DatabaseSettings(uri="mongodb+srv://bob:p%40ss%21word@h.example.net/")
        assert "p%40ss%21word" in s.secret_fragments
        assert "p@ss!word" in s.secret_fragments

    def test_redact_masks_password_wherever_it_appears(self):
        s = DatabaseSettings(uri=FAKE_URI)
        text = f"auth failed for {FAKE_URI} (password fakePass123 rejected)"
        cleaned = s.redact(text)
        assert "fakePass123" not in cleaned
        assert "demo_user" not in cleaned
        assert "cluster.example.mongodb.net" in cleaned  # host stays useful

    def test_redact_text_masks_any_uri(self):
        assert redact_text("see mongodb://admin:secret@h/") == "see mongodb://***@h/"
        assert redact_text("see MONGODB+SRV://a:b@h") == "see MONGODB+SRV://***@h"

    def test_redact_truncates_long_messages(self):
        assert len(redact_text("x" * 5000)) <= 310


# ── Collection names ─────────────────────────────────────────────────────────

class TestCollections:
    def test_established_names(self):
        assert ALL_COLLECTIONS == (
            "users", "home_projects", "products", "agent_states",
            "tasks", "messages", "recommendations", "negotiations",
        )

    def test_members_behave_as_strings(self):
        assert Collections.PRODUCTS == "products"
        assert str(Collections.AGENT_STATES) == "agent_states"
        assert f"{Collections.NEGOTIATIONS}" == "negotiations"

    def test_names_are_unique(self):
        assert len(set(ALL_COLLECTIONS)) == len(ALL_COLLECTIONS) == 8
