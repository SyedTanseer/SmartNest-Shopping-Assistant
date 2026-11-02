"""Tests for MongoConnection and the shared connection (mocked driver).

No network access: the MongoClient is replaced by a fake factory.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest
from pymongo.errors import (
    ConfigurationError,
    InvalidURI,
    OperationFailure,
    ServerSelectionTimeoutError,
)

from smartnest.database import (
    Collections,
    DatabaseConfigError,
    DatabaseConnectionError,
    DatabaseSettings,
    MongoConnection,
    close_connection,
    get_connection,
)

PASSWORD = "fakePass123"
URI = f"mongodb+srv://demo_user:{PASSWORD}@cluster.example.mongodb.net/"


def _settings(**kw) -> DatabaseSettings:
    return DatabaseSettings(uri=URI, **kw)


def _fake_client() -> MagicMock:
    client = MagicMock(name="MongoClient")
    client.admin.command.return_value = {"ok": 1}
    return client


def _factory(client=None):
    return MagicMock(name="factory", return_value=client or _fake_client())


@pytest.fixture(autouse=True)
def _reset_shared():
    close_connection()
    yield
    close_connection()


# ── Client creation ──────────────────────────────────────────────────────────

class TestClientCreation:
    def test_creating_manager_does_not_create_client(self):
        factory = _factory()
        conn = MongoConnection(_settings(), factory)
        factory.assert_not_called()
        assert not conn.is_connected

    def test_client_created_lazily_and_reused(self):
        factory = _factory()
        conn = MongoConnection(_settings(), factory)
        first = conn.client
        assert conn.client is first and conn.database is not None
        assert factory.call_count == 1
        assert conn.is_connected

    def test_pool_and_timeout_options_are_passed(self):
        factory = _factory()
        MongoConnection(_settings(max_pool_size=12, min_pool_size=2,
                                  server_selection_timeout_ms=1234), factory).client
        args, kwargs = factory.call_args
        assert args == (URI,)
        assert kwargs["maxPoolSize"] == 12 and kwargs["minPoolSize"] == 2
        assert kwargs["serverSelectionTimeoutMS"] == 1234
        assert kwargs["tz_aware"] is True
        assert kwargs["appName"] == "smartnest"

    def test_database_and_collection_names(self):
        client = _fake_client()
        conn = MongoConnection(_settings(database="mydb"), _factory(client))
        conn.collection(Collections.PRODUCTS)
        client.__getitem__.assert_called_with("mydb")
        client.__getitem__.return_value.__getitem__.assert_called_with("products")

    @pytest.mark.parametrize("name", ["", "   "])
    def test_empty_collection_name_rejected(self, name):
        with pytest.raises(DatabaseConfigError):
            MongoConnection(_settings(), _factory()).collection(name)

    def test_invalid_uri_becomes_config_error_without_credentials(self):
        factory = MagicMock(side_effect=InvalidURI(f"bad uri {URI} password {PASSWORD}"))
        with pytest.raises(DatabaseConfigError) as info:
            MongoConnection(_settings(), factory).client
        assert PASSWORD not in str(info.value) and "demo_user" not in str(info.value)

    def test_configuration_error_becomes_config_error(self):
        factory = MagicMock(side_effect=ConfigurationError("The DNS query name does not exist"))
        with pytest.raises(DatabaseConfigError, match="DNS"):
            MongoConnection(_settings(), factory).client

    def test_traceback_chain_does_not_expose_driver_message(self):
        factory = MagicMock(side_effect=InvalidURI(f"leaked {PASSWORD}"))
        with pytest.raises(DatabaseConfigError) as info:
            MongoConnection(_settings(), factory).client
        assert info.value.__cause__ is None and info.value.__suppress_context__


# ── Ping and health ──────────────────────────────────────────────────────────

class TestPingAndHealth:
    def test_ping_returns_latency(self):
        client = _fake_client()
        conn = MongoConnection(_settings(), _factory(client))
        assert conn.ping() >= 0.0
        client.admin.command.assert_called_once_with("ping")

    def test_ping_failure_is_connection_error_without_credentials(self):
        client = _fake_client()
        client.admin.command.side_effect = ServerSelectionTimeoutError(
            f"cannot reach {URI} (password {PASSWORD})")
        with pytest.raises(DatabaseConnectionError) as info:
            MongoConnection(_settings(), _factory(client)).ping()
        assert PASSWORD not in str(info.value) and "demo_user" not in str(info.value)
        assert info.value.__cause__ is None

    def test_authentication_failure_is_connection_error(self):
        client = _fake_client()
        client.admin.command.side_effect = OperationFailure("bad auth : authentication failed", 18)
        with pytest.raises(DatabaseConnectionError, match="authentication failed"):
            MongoConnection(_settings(), _factory(client)).ping()

    def test_health_check_ok(self):
        status = MongoConnection(_settings(database="mydb"), _factory()).check_health()
        assert status.ok and status.database == "mydb"
        assert status.latency_ms is not None and status.error is None

    def test_health_check_never_raises(self):
        client = _fake_client()
        client.admin.command.side_effect = ServerSelectionTimeoutError(PASSWORD)
        conn = MongoConnection(_settings(), _factory(client))
        status = conn.check_health()
        assert not status.ok and status.latency_ms is None
        assert PASSWORD not in status.error
        assert conn.is_healthy() is False

    def test_health_check_on_closed_connection_reports_unhealthy(self):
        conn = MongoConnection(_settings(), _factory())
        conn.close()
        assert conn.check_health().ok is False


# ── connect / close ──────────────────────────────────────────────────────────

class TestConnectAndClose:
    def test_connect_verifies_with_ping(self):
        client = _fake_client()
        conn = MongoConnection(_settings(), _factory(client))
        assert conn.connect() is client
        client.admin.command.assert_called_once_with("ping")

    def test_connect_without_verify_does_not_ping(self):
        client = _fake_client()
        MongoConnection(_settings(), _factory(client)).connect(verify=False)
        client.admin.command.assert_not_called()

    def test_failed_verify_discards_client_and_allows_retry(self):
        bad, good = _fake_client(), _fake_client()
        bad.admin.command.side_effect = ServerSelectionTimeoutError("down")
        factory = MagicMock(side_effect=[bad, good])
        conn = MongoConnection(_settings(), factory)
        with pytest.raises(DatabaseConnectionError):
            conn.connect()
        bad.close.assert_called_once()
        assert not conn.is_connected and not conn.is_closed
        assert conn.connect() is good

    def test_close_closes_client_once(self):
        client = _fake_client()
        conn = MongoConnection(_settings(), _factory(client))
        conn.connect(verify=False)
        conn.close()
        conn.close()
        client.close.assert_called_once()
        assert conn.is_closed and not conn.is_connected

    def test_close_without_connecting_is_safe(self):
        factory = _factory()
        conn = MongoConnection(_settings(), factory)
        conn.close()
        factory.assert_not_called()
        assert conn.is_closed

    def test_use_after_close_raises(self):
        conn = MongoConnection(_settings(), _factory())
        conn.close()
        for action in (lambda: conn.client, lambda: conn.database, conn.ping,
                       lambda: conn.collection("x")):
            with pytest.raises(DatabaseConnectionError, match="closed"):
                action()

    def test_connect_reopens_after_close(self):
        factory = MagicMock(side_effect=[_fake_client(), _fake_client()])
        conn = MongoConnection(_settings(), factory)
        first = conn.connect(verify=False)
        conn.close()
        second = conn.connect(verify=False)
        assert second is not first and not conn.is_closed
        assert factory.call_count == 2

    def test_error_while_closing_does_not_propagate(self):
        client = _fake_client()
        client.close.side_effect = RuntimeError("boom")
        conn = MongoConnection(_settings(), _factory(client))
        conn.connect(verify=False)
        conn.close()  # must not raise
        assert conn.is_closed

    def test_context_manager_connects_and_closes(self):
        client = _fake_client()
        with MongoConnection(_settings(), _factory(client)) as conn:
            client.admin.command.assert_called_once_with("ping")
            assert conn.is_connected
        client.close.assert_called_once()

    def test_context_manager_closes_on_error(self):
        client = _fake_client()
        with pytest.raises(ValueError):
            with MongoConnection(_settings(), _factory(client)):
                raise ValueError("inside")
        client.close.assert_called_once()

    def test_repr_hides_credentials(self):
        text = repr(MongoConnection(_settings(), _factory()))
        assert PASSWORD not in text and "demo_user" not in text
        assert "cluster.example.mongodb.net" in text


# ── Shared connection ────────────────────────────────────────────────────────

class TestSharedConnection:
    def test_returns_same_instance(self):
        factory = _factory()
        a = get_connection(_settings(), factory)
        assert get_connection() is a

    def test_threads_share_one_connection_and_client(self):
        factory = _factory()
        results, clients = [], []

        def worker():
            conn = get_connection(_settings(), factory)
            results.append(conn)
            clients.append(conn.client)

        threads = [threading.Thread(target=worker) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len({id(c) for c in results}) == 1
        assert len({id(c) for c in clients}) == 1
        assert factory.call_count == 1

    def test_close_connection_closes_and_resets(self):
        client = _fake_client()
        first = get_connection(_settings(), _factory(client))
        first.connect(verify=False)
        close_connection()
        client.close.assert_called_once()
        second = get_connection(_settings(), _factory())
        assert second is not first

    def test_close_connection_when_none_is_safe(self):
        close_connection()
        close_connection()

    def test_closed_shared_connection_is_replaced(self):
        first = get_connection(_settings(), _factory())
        first.close()
        assert get_connection(_settings(), _factory()) is not first
