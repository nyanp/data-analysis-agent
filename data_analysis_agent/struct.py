import pandas as pd
from pydantic import BaseModel, Field

from data_analysis_agent.enum import ColumnType, FilterOp


class Column(BaseModel):
    name: str
    column_type: ColumnType
    sample_values: list[str] = Field(..., description="5 values of this column")


class Dataset(BaseModel):
    dataset_id: int
    dataset_name: str
    columns: list[Column]

    @classmethod
    def from_pandas(
        cls, df: pd.DataFrame, dataset_id: int, dataset_name: str
    ) -> "Dataset":
        columns = [
            Column(
                name=col,
                column_type=ColumnType.infer_from_series(df[col]),
                sample_values=df[col].iloc[:5].values.astype(str).tolist(),
            )
            for col in df.columns
        ]
        return cls(dataset_id=dataset_id, dataset_name=dataset_name, columns=columns)


class Datasets(BaseModel):
    datasets: list[Dataset]


class FilterCondition(BaseModel):
    column: str
    op: FilterOp
    values: list[str]


class DataframeWithMetadata:
    def __init__(self, dataframe: pd.DataFrame, metadata: Dataset) -> None:
        self._dataframe = dataframe
        self._metadata = metadata

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._dataframe

    @property
    def metadata(self) -> Dataset:
        return self._metadata

    @classmethod
    def from_pandas(
        cls, df: pd.DataFrame, dataset_id: int, dataset_name: str
    ) -> "DataframeWithMetadata":
        return cls(
            dataframe=df, metadata=Dataset.from_pandas(df, dataset_id, dataset_name)
        )

    def __repr__(self) -> str:
        meta = f"DataframeWithMetadata(id={self.metadata.dataset_id}, name={self.metadata.dataset_name}, shape={self._dataframe.shape})"
        schema = "\n".join(
            [f"{col.name}: {col.column_type}" for col in self.metadata.columns]
        )
        data = self._dataframe.head().to_string()
        return f"{meta}\n{schema}\n{data}"
