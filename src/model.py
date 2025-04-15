import pickle
import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.base import ClassifierMixin

class Model:
    def __init__(self, best_params: Dict[str, Any]) -> None:
        """
        Инициализация модели с переданными гиперпараметрами.
        :param best_params: Словарь с гиперпараметрами модели.
        """
        self.model: GradientBoostingClassifier = GradientBoostingClassifier(**best_params)
        self.best_params: Dict[str, Any] = best_params

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Обучает модель на тренировочных данных.
        :param X_train: Матрица признаков (обучающая выборка).
        :param y_train: Целевая переменная.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Выполняет предсказание вероятностей и классов, добавляя customer_mindbox_id.
        
        :param X_test: Матрица признаков (тестовая выборка).
        :return: DataFrame с предсказанными вероятностями, классами и ID клиента.
        """
        y_pred_proba: np.ndarray = self.model.predict_proba(X_test)[:, 1]
        y_pred: np.ndarray = (y_pred_proba > 0.5).astype(int)

        predictions = pd.DataFrame({
            "Predicted Probability": y_pred_proba,
            "Predicted Class": y_pred
        }, index=X_test.index)

        predictions.index.name = "customer_mindbox_id"

        return predictions

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Оценивает модель по метрикам ROC AUC, F1-score, Precision и Recall.
        :param X_test: Матрица признаков (тестовая выборка).
        :param y_test: Целевая переменная (тестовая выборка).
        :return: Словарь с метриками качества модели.
        """
        y_pred_proba: np.ndarray = self.model.predict_proba(X_test)[:, 1]
        y_pred: np.ndarray = (y_pred_proba > 0.5).astype(int)

        metrics: Dict[str, float] = {
            "ROC AUC": roc_auc_score(y_test, y_pred_proba),
            "F1-score": f1_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred)
        }
        return metrics

    def save_model(self, path: str = "model.pkl") -> None:
        """
        Сохраняет обученную модель в файл.
        :param path: Путь для сохранения файла.
        """
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    @staticmethod
    def load_model(path: str = "model.pkl") -> ClassifierMixin:
        """
        Загружает модель из файла.
        :param path: Путь к файлу с моделью.
        :return: Загруженная модель.
        """
        with open(path, "rb") as f:
            model: ClassifierMixin = pickle.load(f)
        return model