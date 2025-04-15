import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

class DatabaseConnection:
    def __init__(self) -> None:
        """
        Инициализирует соединение с базой данных.
        Загружает параметры из .env файла и устанавливает соединение.
        """
        server: str = os.getenv("DB_SERVER")
        database: str = os.getenv("DB_NAME")
        driver: str = "SQL Server"

        connection_string: str = f"DRIVER={{{driver}}};SERVER={server};DATABASE={database};Trusted_Connection=yes"
        connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string})

        self.engine = create_engine(connection_url, use_setinputsizes=False)

    def get_engine(self) -> Optional[object]:
        """
        Возвращает объект engine для взаимодействия с базой данных.

        :return: объект engine или None, если соединение не установлено
        """
        return self.engine

    def select(self, sql: str) -> pd.DataFrame:
        """
        Выполняет SQL-запрос и возвращает результат в виде DataFrame.

        :param sql: SQL-запрос
        :return: Результат запроса в виде DataFrame
        """
        return pd.read_sql(sql, self.engine)