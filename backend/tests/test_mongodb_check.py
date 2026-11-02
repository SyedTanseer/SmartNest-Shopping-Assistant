"""Tests for src/mongodb_check.py using an in-memory fake client (no network)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import mongodb_check
from smartnest.database import DatabaseSettings, MongoConnection
from tests.fake_mongo import FakeCollection

PASSWORD = "fakePass123"
URI = f"mongodb+srv://demo_user:{PASSWORD}@cluster.example.mongodb.net/"


class FakeDatabase:
    def __init__(self):
        self.cols: dict[str, FakeCollection] = {}
        self.dropped: list[str] = []

    def list_collection_names(self):
        return list(self.cols)

    def __getitem__(self, name):
        return self.cols.setdefault(name, FakeCollection(name))

    def drop_collection(self, name):
        self.cols.pop(name, None)
        self.dropped.append(name)


class FakeClient:
    def __init__(self, ping_error=None):
        self.db = FakeDatabase()
        self.closed = 0
        self.admin = SimpleNamespace(command=self._command)
        self._ping_error = ping_error

    def _command(self, name):
        if self._ping_error:
            raise self._ping_error
        return {"ok": 1}

    def __getitem__(self, _name):
        return self.db

    def close(self):
        self.closed += 1


def _connection(client=None):
    client = client or FakeClient()
    return MongoConnection(DatabaseSettings(uri=URI), lambda uri, **kw: client), client


def _run(connection):
    lines: list[str] = []
    code = mongodb_check.run(connection, out=lines.append)
    return code, "\n".join(lines)


class TestSuccessfulRun:
    def test_all_steps_pass(self):
        conn, client = _connection()
        code, output = _run(conn)
        assert code == 0
        for step in range(1, 8):
            assert f"[{step}/7] OK" in output
        assert "RESULT: SUCCESS" in output

    def test_test_document_is_deleted_and_scratch_collection_dropped(self):
        conn, client = _connection()
        _run(conn)
        assert client.db.dropped == [mongodb_check.SCRATCH_COLLECTION]
        assert mongodb_check.SCRATCH_COLLECTION not in client.db.cols

    def test_existing_scratch_collection_is_not_dropped(self):
        client = FakeClient()
        client.db[mongodb_check.SCRATCH_COLLECTION]  # pre-existing
        conn, _ = _connection(client)
        code, _ = _run(conn)
        assert code == 0 and client.db.dropped == []

    def test_connection_is_closed(self):
        conn, client = _connection()
        _run(conn)
        assert client.closed == 1 and conn.is_closed

    def test_output_never_contains_credentials(self):
        conn, _ = _connection()
        _, output = _run(conn)
        assert PASSWORD not in output and "demo_user" not in output
        assert "mongodb+srv://" not in output
        assert "cluster.example.mongodb.net" in output  # host is fine to show


class TestFailures:
    def test_unreachable_server_fails_cleanly(self):
        from pymongo.errors import ServerSelectionTimeoutError
        client = FakeClient(ping_error=ServerSelectionTimeoutError(f"cannot reach {URI} ({PASSWORD})"))
        conn, _ = _connection(client)
        code, output = _run(conn)
        assert code == 1
        assert "FAIL" in output and "RESULT: FAILED" in output
        assert PASSWORD not in output and "demo_user" not in output
        assert client.closed == 1

    def test_missing_configuration(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MONGODB_URI", raising=False)
        monkeypatch.setattr("smartnest.database.settings.DEFAULT_ENV_FILE", tmp_path / "none.env")
        lines: list[str] = []
        assert mongodb_check.run(None, out=lines.append) == 1
        assert "MONGODB_URI is not set" in "\n".join(lines)

    def test_delete_not_confirmed_is_a_failure_and_still_cleans_up(self, monkeypatch):
        conn, client = _connection()
        real = FakeCollection.delete_one
        calls = {"n": 0}

        def flaky(self, flt):
            calls["n"] += 1
            if calls["n"] == 1:  # first delete "does nothing"
                return SimpleNamespace(deleted_count=0)
            return real(self, flt)

        monkeypatch.setattr(FakeCollection, "delete_one", flaky)
        code, output = _run(conn)
        assert code == 1 and "delete reported that nothing was deleted" in output
        assert "cleaned up leftover test document" in output
        assert client.closed == 1
        assert client.db.cols.get(mongodb_check.SCRATCH_COLLECTION) is None  # dropped: empty

    def test_unexpected_exception_is_reported_without_traceback_secrets(self, monkeypatch):
        conn, client = _connection()

        def boom(self, doc):
            raise RuntimeError(f"unexpected with {URI}")

        monkeypatch.setattr(FakeCollection, "insert_one", boom)
        code, output = _run(conn)
        assert code == 1 and PASSWORD not in output and "demo_user" not in output
        assert client.closed == 1
