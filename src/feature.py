import pandas as pd
import gc
from typing import Callable

class Feature:
    def __init__(self, name: str, calculate_query: str, batch_size: int = 1000) -> None:
        """
        Инициализирует объект Feature с параметрами для вычисления фичи.

        :param name: Имя фичи
        :param calculate_query: SQL-запрос для вычисления фичи
        :param batch_size: Размер батча для обработки (по умолчанию 1000)
        """
        self.name: str = name
        self.calculate_query: str = calculate_query
        self.batch_size: int = batch_size
        self.df: pd.DataFrame = pd.DataFrame()

    def calculate_batch(self, batch: pd.DataFrame, select_func: Callable[[str], pd.DataFrame]) -> pd.DataFrame:
        """
        Выполняет SQL-запрос для одного батча клиентов.

        :param batch: DataFrame с клиентами для обработки
        :param select_func: Функция для выполнения SQL-запроса
        :return: DataFrame с результатами запроса
        """
        values = ', '.join(map(str, batch['customer_mindbox_id']))
        query = self.calculate_query.format(values=values)
        return select_func(query)

    def calculate(self, customers: pd.DataFrame, select_func: Callable[[str], pd.DataFrame]) -> pd.DataFrame:
        """
        Разбивает клиентов на батчи и выполняет запросы для каждой группы клиентов.

        :param customers: DataFrame с клиентами
        :param select_func: Функция для выполнения SQL-запроса
        :return: DataFrame с объединенными результатами
        """
        customer_batches = [customers[i:i + self.batch_size] for i in range(0, len(customers), self.batch_size)]
        all_results = [self.calculate_batch(batch, select_func) for batch in customer_batches]
        return pd.concat(all_results, ignore_index=True)

    def read(self, customers: pd.DataFrame, select_func: Callable[[str], pd.DataFrame]) -> None:
        """
        Читает данные и сохраняет их в self.df.

        :param customers: DataFrame с клиентами
        :param select_func: Функция для выполнения SQL-запроса
        """
        self.df = self.calculate(customers, select_func)

    def purge(self) -> None:
        """Очищает данные из памяти"""
        del self.df
        gc.collect()