import pandas as pd
from connection import DatabaseConnection
from typing import Optional

class DataLoader:
    def __init__(self) -> None:
        """
        Инициализирует объект DataLoader и создает экземпляр подключения к базе данных.
        """
        self.db: DatabaseConnection = DatabaseConnection()

    def load_data(self, query: str) -> pd.DataFrame:
        """
        Загружает данные из базы данных, выполняя SQL-запрос.

        :param query: SQL-запрос в виде строки
        :return: Результат запроса в виде DataFrame
        """
        return self.db.select(query)