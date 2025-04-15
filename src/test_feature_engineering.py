import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from feature_engineering import FeatureEngineering

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "days_since_last_purchase": [10, 5, np.nan],
        "days_since_last_redemption": [15, 3, 8],
        "purchase_sum_restore": [100, 200, 150],
        "bonuses_spisanie": [10, 0, np.nan],
        "purchase_count": [5, 10, 2],
        "redemption_count": [1, 2, 0],
        "customer_mindbox_id": [101, 102, 103]
    })

def test_clean_data(sample_df):
    fe = FeatureEngineering(sample_df)
    fe.generate_features()
    cleaned_df = fe.clean_data(columns_to_drop=["non_existent_column", "bonuses_spisanie"])

    assert not cleaned_df.isin([np.nan, np.inf, -np.inf]).any().any()
    assert "bonuses_spisanie" not in cleaned_df.columns

@patch("matplotlib.pyplot.savefig")
@patch("matplotlib.pyplot.show")
def test_plot_correlation_matrix(mock_show, mock_savefig, sample_df, tmp_path):
    fe = FeatureEngineering(sample_df)
    fe.generate_features()
    save_path = tmp_path / "corr_matrix.png"

    fe.plot_correlation_matrix(method="pearson", save_path=str(save_path))

    mock_savefig.assert_called_once()
    mock_show.assert_called_once()