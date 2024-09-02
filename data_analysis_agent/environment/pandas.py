import pandas as pd
from openai import OpenAI

from data_analysis_agent.enum import JoinMethod
from data_analysis_agent.environment import BaseEnvironment
from data_analysis_agent.struct import DataframeWithMetadata, Dataset, Datasets
from data_analysis_agent.util import transform_df_by_llm


class PandasEnvironment(BaseEnvironment):
    def __init__(self, dataframes: dict[str, pd.DataFrame] | None = None) -> None:
        self._dataframes = (
            {
                key: DataframeWithMetadata.from_pandas(df, i + 1, key)
                for i, (key, df) in enumerate(dataframes.items())
            }
            if dataframes
            else {}
        )

    def __getitem__(self, key: str | int) -> DataframeWithMetadata:
        if isinstance(key, str):
            return self._dataframes[key]
        elif isinstance(key, int):
            return next(
                df for df in self._dataframes.values() if df.metadata.dataset_id == key
            )
        else:
            raise ValueError(f"Invalid key type: {type(key)}")

    def list_datasets(self) -> Datasets:
        return Datasets(datasets=[df.metadata for df in self._dataframes.values()])

    def _register(
        self, df: pd.DataFrame, dataset_id: int, base_dataset: DataframeWithMetadata
    ) -> None:
        dataset_name_base = base_dataset.metadata.dataset_name.split(":")[0]
        dst_name = f"{dataset_name_base}:{dataset_id}"
        self._dataframes[dst_name] = DataframeWithMetadata.from_pandas(
            df, dataset_id, dst_name
        )

    def add_dataset(self, client: OpenAI, dataset: Dataset, instruction: str) -> None:
        df = pd.DataFrame(
            {
                col.name: dataset.columns[i].sample_values
                for i, col in enumerate(dataset.columns)
            }
        )
        self._dataframes[dataset.dataset_name] = DataframeWithMetadata(
            dataframe=df, metadata=dataset
        )

    def add_column(
        self,
        client: OpenAI,
        src_dataset_id: int,
        dst_dataset_id: int,
        expression: str,
        instruction: str,
    ) -> None:
        prompt = f"""
- Instruction: {instruction}
- Pseudo-code: {expression}
        """
        dst = transform_df_by_llm(client, self[src_dataset_id], prompt)

        self._register(dst, dst_dataset_id, self[src_dataset_id])

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
        left = self[left_dataset_id].dataframe
        right = self[right_dataset_id].dataframe

        merged = pd.merge(
            left, right, left_on=left_on, right_on=right_on, how=how.value.lower()
        )

        self._register(merged, dst_dataset_id, self[left_dataset_id])
