from logging import getLogger
from typing import Any, Callable

import pandas as pd
from openai import OpenAI

from data_analysis_agent.struct import DataframeWithMetadata

TFunc = Callable[..., Any]


_logger = getLogger(__name__)


def extract_code_block(s: str) -> str:
    if "```" in s:
        block = s.split("```")[1]
        if block.startswith("python"):
            return block[6:]
        else:
            return block
    else:
        return s


def unsafe_function(f: TFunc) -> Callable[[TFunc], TFunc]:
    def _wrapper(*args, **kwargs):
        _logger.warning(f"Calling unsafe function {f.__name__}.")
        v = f(*args, **kwargs)
        return v

    return _wrapper


def transform_df_by_llm(
    client: OpenAI, source: DataframeWithMetadata, request: str
) -> pd.DataFrame:
    system_prompt = f"""
    以下のデータフレームに対して、ユーザーが要求する操作を行う自己完結したPythonコードを記述してください。

    ## データフレーム
    {source.dataframe.head().to_string()}

    ## ルール
    - データフレームはsource.csvから読み込み、結果をresult.csvに保存してください。
    - Pythonコード以外の出力はしないでください。
    """

    source.dataframe.to_csv("source.csv", index=False)

    completion = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request},
        ],
    )
    code = extract_code_block(completion.choices[0].message.content)
    exec(code)
    return pd.read_csv("result.csv")
