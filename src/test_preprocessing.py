import pandas as pd
import numpy as np
import pytest
from preprocessing import DataPreprocessor


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "customer_mindbox_id": [1, 2, 3],
        "days_since_last_purchase": [10, np.nan, 5],
        "days_since_last_redemption": [7, 15, np.nan],
        "purchase_sum_restore": [100, 200, 300],
        "bonuses_spisanie": [10, np.nan, 30]
    })


def test_handle_missing_values_mean(sample_df):
    dp = DataPreprocessor(fill_strategy="mean")
    df = dp.handle_missing_values(sample_df.copy())
    assert not df.isnull().values.any()


def test_handle_missing_values_median(sample_df):
    dp = DataPreprocessor(fill_strategy="median")
    df = dp.handle_missing_values(sample_df.copy())
    assert not df.isnull().values.any()


def test_handle_missing_values_zero(sample_df):
    dp = DataPreprocessor(fill_strategy="zero")
    df = dp.handle_missing_values(sample_df.copy())
    assert not df.isnull().values.any()
    assert (df == 0).sum().sum() > 0

def test_scale_features(sample_df):
    dp = DataPreprocessor()
    filled_df = dp.handle_missing_values(sample_df.copy())
    df_scaled = dp.scale_features(filled_df.copy())

    numeric_cols = df_scaled.select_dtypes(include=["float64", "int64"]).columns
    numeric_cols = [col for col in numeric_cols if col != "customer_mindbox_id"]
    means = df_scaled[numeric_cols].mean().round()
    stds = df_scaled[numeric_cols].std().round()

    assert all(abs(means) < 1)
    assert all((stds == 1) | (stds == 0))


def test_full_preprocess(sample_df):
    dp = DataPreprocessor()
    df_processed = dp.preprocess(sample_df.copy())
    assert not df_processed.isnull().values.any()