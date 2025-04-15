import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from connection import DatabaseConnection

@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("DB_SERVER", "test_server")
    monkeypatch.setenv("DB_NAME", "test_database")


@patch("connection.create_engine")
def test_init_creates_engine(mock_create_engine, mock_env):
    db = DatabaseConnection()

    mock_create_engine.assert_called_once()


def test_get_engine_returns_engine(mock_env):
    db = DatabaseConnection()
    db.engine = "fake_engine"
    engine = db.get_engine()
    assert engine == "fake_engine"


@patch("connection.pd.read_sql")
def test_select_executes_sql_query(mock_read_sql, mock_env):
    mock_read_sql.return_value = pd.DataFrame({"col": [1, 2, 3]})

    db = DatabaseConnection()
    db.engine = MagicMock()

    query = "SELECT * FROM dbo.test"
    result = db.select(query)

    mock_read_sql.assert_called_once_with(query, db.engine)
    pd.testing.assert_frame_equal(result, pd.DataFrame({"col": [1, 2, 3]}))