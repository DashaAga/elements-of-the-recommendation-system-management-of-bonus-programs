import pytest
import pandas as pd
from unittest.mock import MagicMock, patch, mock_open
from feature_manager import FeatureManager

@pytest.fixture
def dummy_customers():
    return pd.DataFrame({"customer_mindbox_id": [1, 2]})

@patch("feature_manager.Feature")
@patch("builtins.open", new_callable=mock_open, read_data="""
features:
  - name: test_feature
    source: some_source
    column: test_column
""")
@patch("feature_manager.yaml.safe_load", return_value={"features": [{"name": "test_feature", "source": "some_source", "column": "test_column"}]})
def test_feature_manager_initialization(mock_yaml, mock_file, mock_feature_class):
    mock_data_loader = MagicMock()
    fm = FeatureManager("fake_path.yaml", mock_data_loader)

    assert len(fm.features) == 1
    mock_feature_class.assert_called_once_with(name="test_feature", source="some_source", column="test_column")

@patch("feature_manager.Feature")
@patch("builtins.open", new_callable=mock_open, read_data="""
features:
  - name: test_feature
    source: some_source
    column: test_column
""")
@patch("feature_manager.yaml.safe_load", return_value={"features": [{"name": "test_feature", "source": "some_source", "column": "test_column"}]})
def test_generate_features_merges_data(mock_yaml, mock_file, mock_feature_class, dummy_customers):
    mock_data_loader = MagicMock()

    mock_feature = MagicMock()
    mock_feature.df = pd.DataFrame({
        "customer_mindbox_id": [1, 2],
        "test_column": [10, 20]
    })
    mock_feature_class.return_value = mock_feature

    fm = FeatureManager("fake_path.yaml", mock_data_loader)
    result_df = fm.generate_features(dummy_customers)

    assert "test_column" in result_df.columns
    pd.testing.assert_series_equal(
    result_df["test_column"],
    pd.Series([10, 20], name="test_column")
    )
    mock_feature.read.assert_called_once()
    mock_feature.purge.assert_called_once()