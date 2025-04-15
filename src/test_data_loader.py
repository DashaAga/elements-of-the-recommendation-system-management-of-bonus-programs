import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from connection import DatabaseConnection
from data_loader import DataLoader

def test_dataloader_initializes_db_connection():
    loader = DataLoader()
    assert isinstance(loader.db, DatabaseConnection)

@patch.object(DatabaseConnection, 'select')
def test_load_data_returns_dataframe(mock_select):
    mock_df = pd.DataFrame({"col": [1, 2, 3]})
    mock_select.return_value = mock_df

    loader = DataLoader()
    query = "SELECT * FROM table"
    result = loader.load_data(query)

    mock_select.assert_called_once_with(query)
    pd.testing.assert_frame_equal(result, mock_df)