import pytest
import pandas as pd
from unittest.mock import MagicMock
from feature import Feature


@pytest.fixture
def sample_customers():
    return pd.DataFrame({
        "customer_mindbox_id": [1, 2, 3, 4]
    })


@pytest.fixture
def mock_select_func():
    def select_func(query):
        return pd.DataFrame({
            "customer_mindbox_id": [int(id_) for id_ in query.split("IN (")[1].split(")")[0].split(", ")],
            "feature_value": [100] * len(query.split(", "))
        })
    return select_func


def test_calculate_batch_formats_query_correctly(sample_customers, mock_select_func):
    feature = Feature("test_feature", "SELECT * FROM features WHERE customer_mindbox_id IN ({values})", batch_size=2)
    batch = sample_customers.iloc[:2]

    result = feature.calculate_batch(batch, mock_select_func)

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"customer_mindbox_id", "feature_value"}
    assert list(result["customer_mindbox_id"]) == [1, 2]


def test_calculate_batches_and_concatenates(sample_customers, mock_select_func):
    feature = Feature("test_feature", "SELECT * FROM features WHERE customer_mindbox_id IN ({values})", batch_size=2)

    result_df = feature.calculate(sample_customers, mock_select_func)

    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) == 4
    assert "feature_value" in result_df.columns


def test_read_stores_result_in_df(sample_customers, mock_select_func):
    feature = Feature("test_feature", "SELECT * FROM features WHERE customer_mindbox_id IN ({values})", batch_size=2)

    feature.read(sample_customers, mock_select_func)

    assert hasattr(feature, "df")
    assert isinstance(feature.df, pd.DataFrame)
    assert len(feature.df) == 4


def test_purge_deletes_df(sample_customers, mock_select_func):
    feature = Feature("test_feature", "SELECT * FROM features WHERE customer_mindbox_id IN ({values})", batch_size=2)
    feature.read(sample_customers, mock_select_func)

    assert hasattr(feature, "df")

    feature.purge()

    assert not hasattr(feature, "df")