from abc import ABC, abstractmethod

from openai import OpenAI

from data_analysis_agent.enum import JoinMethod
from data_analysis_agent.struct import Dataset, Datasets


class BaseEnvironment(ABC):
    @abstractmethod
    def list_datasets(self) -> Datasets:
        raise NotImplementedError()

    @abstractmethod
    def add_dataset(self, client: OpenAI, dataset: Dataset, instruction: str) -> None:
        raise NotImplementedError()

    @abstractmethod
    def add_column(
        self,
        client: OpenAI,
        src_dataset_id: int,
        dst_dataset_id: int,
        expression: str,
        instruction: str,
    ) -> None:
        raise NotImplementedError()

    @abstractmethod
    def join(
        self,
        client: OpenAI,
        left_dataset_id: int,
        right_dataset_id: int,
        left_on: list[str],
        right_on: list[str],
        dst_dataset_id: int,
        how: JoinMethod,
    ) -> None:
        raise NotImplementedError()
