import pandas as pd
import yaml
from typing import List
from data_loader import DataLoader
from feature import Feature

class FeatureManager:
    def __init__(self, feature_file: str, data_loader: DataLoader) -> None:
        """
        Инициализирует FeatureManager с файлом конфигурации фичей и экземпляром DataLoader.

        :param feature_file: Путь к YAML-файлу с конфигурациями фичей.
        :param data_loader: Экземпляр класса DataLoader для загрузки данных.
        """
        self.data_loader = data_loader
        with open(feature_file, "r", encoding="utf-8") as f:
            self.feature_configs = yaml.safe_load(f)["features"]
        self.features: List[Feature] = [Feature(**config) for config in self.feature_configs]

    def generate_features(self, customers: pd.DataFrame) -> pd.DataFrame:
        """
        Генерирует фичи для списка клиентов, объединяя их с данными клиентов.

        :param customers: Данные клиентов в виде DataFrame.
        :return: Обновленные данные клиентов с добавленными фичами.
        """
        for feature in self.features:
            feature.read(customers, self.data_loader.db.select)
            customers = customers.merge(feature.df, on="customer_mindbox_id", how="left")
            feature.purge()
        return customers