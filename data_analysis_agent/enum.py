from enum import Enum

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_string_dtype,
)


class Language(str, Enum):
    JAPANESE = "Japanese"
    ENGLISH = "English"


class ColumnType(str, Enum):
    DATETIME = "datetime"
    DATE = "date"
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"

    @classmethod
    def infer_from_series(cls, series: pd.Series) -> "ColumnType":
        if is_datetime64_any_dtype(series):
            return cls.DATETIME
        if is_bool_dtype(series):
            return cls.BOOLEAN
        if is_integer_dtype(series):
            return cls.INTEGER
        if is_numeric_dtype(series):
            return cls.FLOAT
        if is_string_dtype(series):
            return cls.STRING
        return cls.STRING


class AggregateFunction(str, Enum):
    SUM = "sum"
    COUNT = "count"
    AVG = "avg"
    MAX = "max"
    MIN = "min"


class FilterOp(str, Enum):
    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    BETWEEN = "between"
    IN = "in"
    NOT_IN = "not_in"


class ChartType(str, Enum):
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    SCATTER = "scatter"


class JoinMethod(str, Enum):
    INNER = "inner"
    LEFT = "left"
    FULL = "full"
