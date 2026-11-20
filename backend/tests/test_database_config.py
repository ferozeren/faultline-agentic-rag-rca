"""Database URL construction tests that never connect to or print credentials."""

from app.database.config import database_url


def test_database_url_is_preferred_and_uses_psycopg(monkeypatch):
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql://test-user:test-password@localhost:5432/test-db",
    )
    monkeypatch.setenv("POSTGRES_HOST", "ignored")
    monkeypatch.setenv("POSTGRES_DB", "ignored")

    url = database_url()

    assert url.drivername == "postgresql+psycopg"
    assert url.username == "test-user"
    assert url.database == "test-db"
    assert "test-password" not in url.render_as_string(hide_password=True)


def test_postgres_variables_build_url_when_database_url_is_empty(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setenv("POSTGRES_HOST", "127.0.0.1")
    monkeypatch.setenv("POSTGRES_PORT", "5433")
    monkeypatch.setenv("POSTGRES_DB", "investigations")
    monkeypatch.setenv("POSTGRES_USER", "investigator")
    monkeypatch.setenv("POSTGRES_PASSWORD", "private-test-value")

    url = database_url()

    assert url.drivername == "postgresql+psycopg"
    assert url.host == "127.0.0.1"
    assert url.port == 5433
    assert url.database == "investigations"
    assert url.username == "investigator"
    assert "private-test-value" not in url.render_as_string(hide_password=True)


def test_invalid_database_url_errors_do_not_echo_secret(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///not-postgres")

    try:
        database_url()
    except ValueError as error:
        assert "not-postgres" not in str(error)
    else:
        raise AssertionError("non-PostgreSQL URLs must be rejected")
