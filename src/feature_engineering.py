import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from typing import Optional


class FeatureEngineering:
    def __init__(self, df: pd.DataFrame) -> None:
        """
        Класс для генерации новых признаков и построения корреляционной матрицы.

        :param df: Исходный DataFrame.
        """
        self.df = df.copy()

    def generate_features(self) -> pd.DataFrame:
        """
        Создает новые признаки на основе имеющихся данных.

        :return: DataFrame с добавленными признаками.
        """
        if "days_since_last_purchase" in self.df.columns and "days_since_last_redemption" in self.df.columns:
            self.df["days_since_last_activity"] = self.df[
                ["days_since_last_purchase", "days_since_last_redemption"]
            ].min(axis=1)

        if "purchase_sum_restore" in self.df.columns and "bonus_write_offs" in self.df.columns:
            self.df["purchase_value_per_bonus"] = self.df["purchase_sum_restore"] / (self.df["bonus_write_offs"] + 1)

        if "bonus_write_offs" in self.df.columns:
            self.df["target"] = (self.df["bonus_write_offs"].fillna(0) > 0).astype(int)

        if "purchase_count" in self.df.columns and "redemption_count" in self.df.columns:
            self.df["purchase_redemption_ratio"] = self.df["purchase_count"] / (self.df["redemption_count"] + 1)

        return self.df

    def clean_data(self, columns_to_drop: list[str]) -> pd.DataFrame:
        """
        Удаляет указанные столбцы, убирает NaN и Inf значения,
        а также делает customer_mindbox_id индексом.

        :param columns_to_drop: Список столбцов, которые нужно удалить.
        :return: Очищенный DataFrame.
        """
        self.df = self.df.drop(columns=columns_to_drop, errors="ignore")
        self.df = self.df.set_index("customer_mindbox_id", drop=False)
        self.df = self.df.replace([np.inf, -np.inf], np.nan).fillna(0)
        return self.df

    def plot_correlation_matrix(self, method: str = "pearson", figsize: tuple = (12, 8), save_path: Optional[str] = None) -> None:
        """
        Строит и отображает корреляционную матрицу.

        :param method: Метод корреляции ("pearson", "kendall", "spearman").
        :param figsize: Размер графика.
        :param save_path: Путь для сохранения изображения (если передан).
        """
        plt.figure(figsize=figsize)
        corr_matrix = self.df.corr(method=method)
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
        plt.title(f"Correlation Matrix ({method})")
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_feature_importance(self, target_col: str = "target", figsize: tuple = (10, 6)) -> None:
        """
        Обучает LightGBM и отображает важности признаков.

        :param target_col: Название целевой переменной.
        :param figsize: Размер графика.
        """
        if target_col not in self.df.columns:
            print(f"Целевая переменная '{target_col}' не найдена в данных.")
            return

        features = self.df.drop(columns=[target_col])
        target = self.df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        model = lgb.LGBMClassifier(random_state=42, verbosity = -1)
        model.fit(X_train, y_train)

        importances = pd.Series(model.feature_importances_, index=features.columns)
        importances = importances.sort_values(ascending=False)

        plt.figure(figsize=figsize)
        sns.barplot(x=importances.values, y=importances.index, palette="viridis")
        plt.title("Feature Importance (LightGBM)")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.show()

    def plot_target_scatter(self, target_col: str = "target") -> None:
        """
        Строит scatter-графики по каждому признаку в сравнении с целевой переменной.
        :param target_col: Название целевой переменной.
        """
        if target_col not in self.df.columns:
            print(f"Целевая переменная '{target_col}' не найдена в данных.")
            return

        features = self.df.drop(columns=[target_col])
        target = self.df[target_col]

        for col in features.columns:
            plt.figure(figsize=(6, 4))
            sns.scatterplot(x=self.df[col], y=target)
            plt.title(f"{col} vs {target_col}")
            plt.xlabel(col)
            plt.ylabel(target_col)
            plt.grid(True)
            plt.tight_layout()
            plt.show()