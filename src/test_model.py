import pytest
import pandas as pd
import numpy as np
import os
from model import Model
from sklearn.ensemble import GradientBoostingClassifier


@pytest.fixture
def sample_data():
    X = pd.DataFrame({
        "f1": [0.1, 0.4, 0.3, 0.9],
        "f2": [1, 0, 1, 0]
    }, index=[101, 102, 103, 104])
    y = pd.Series([0, 1, 0, 1], index=X.index)
    return X, y


def test_model_initialization():
    params = {"n_estimators": 10, "max_depth": 3}
    model = Model(params)

    assert isinstance(model.model, GradientBoostingClassifier)
    assert model.model.n_estimators == 10
    assert model.model.max_depth == 3


def test_model_training_and_prediction(sample_data):
    X, y = sample_data
    model = Model({"n_estimators": 10})
    model.train(X, y)

    predictions = model.predict(X)

    assert isinstance(predictions, pd.DataFrame)
    assert "Predicted Probability" in predictions.columns
    assert "Predicted Class" in predictions.columns
    assert predictions.index.name == "customer_mindbox_id"


def test_model_evaluation_metrics(sample_data):
    X, y = sample_data
    model = Model({"n_estimators": 10})
    model.train(X, y)

    metrics = model.evaluate(X, y)

    assert isinstance(metrics, dict)
    for key in ["ROC AUC", "F1-score", "Precision", "Recall"]:
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0


def test_model_save_and_load(tmp_path):
    model = Model({"n_estimators": 5})
    model_path = tmp_path / "test_model.pkl"
    model.save_model(str(model_path))

    assert os.path.exists(model_path)

    loaded_model = Model.load_model(str(model_path))
    assert isinstance(loaded_model, GradientBoostingClassifier)