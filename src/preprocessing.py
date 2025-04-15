import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Literal

class DataPreprocessor:
    def __init__(self, fill_strategy: Literal["mean", "median", "zero"] = "mean", scale_method: Literal["standard"] = "standard") -> None:
        """
        Класс для предобработки данных, включая обработку пропущенных значений, генерацию новых признаков и нормализацию.

        :param fill_strategy: Стратегия заполнения пропущенных значений ("mean", "median", "zero").
        :param scale_method: Метод нормализации данных ("standard").
        """
        self.fill_strategy = fill_strategy
        self.scale_method = scale_method
        self.scaler = StandardScaler() if scale_method == "standard" else None

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Обрабатывает пропущенные значения в DataFrame.

        :param df: Исходный DataFrame.
        :return: DataFrame без пропущенных значений.
        """
        if self.fill_strategy == "mean":
            df = df.fillna(df.mean())
        elif self.fill_strategy == "median":
            df = df.fillna(df.median())
        elif self.fill_strategy == "zero":
            df = df.fillna(0)
        return df

    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Нормализует числовые признаки, исключая 'customer_mindbox_id'.

        :param df: DataFrame с числовыми признаками.
        :return: DataFrame с нормализованными признаками.
        """
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        numeric_cols = [col for col in numeric_cols if col != 'customer_mindbox_id']

        if df[numeric_cols].shape[0] > 0:
            if self.scaler:
                df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Полный процесс предобработки данных.

        :param df: Исходный DataFrame.
        :return: Очищенный и нормализованный DataFrame.
        """
        df = self.handle_missing_values(df)
        df = self.scale_features(df)
        return df