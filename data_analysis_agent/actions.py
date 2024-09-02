from typing import Sequence

import openai
from openai import OpenAI
from pydantic import BaseModel

import data_analysis_agent.enum as enum
from data_analysis_agent.environment import BaseEnvironment
from data_analysis_agent.planner import Plan, Task, TaskType
from data_analysis_agent.struct import Column, Dataset, Datasets, FilterCondition


class Action(BaseModel):
    def apply(self, env: BaseEnvironment, client: OpenAI, task_context: Task) -> None:
        raise NotImplementedError()


class DataProcessingAction(Action):
    pass


class DataVisualizationAction(Action):
    pass


class Measure(BaseModel):
    column: str
    aggregation: enum.AggregateFunction


class BarOrLineChart(BaseModel):
    axis_columns: list[str]
    color_column: str | None
    measure: Measure
    stacked: bool
    horizontal: bool


class ScatterChart(BaseModel):
    x_column: str
    y_column: str
    color_column: str | None


class PieChart(BaseModel):
    columns: list[str]
    measure: Measure


class DrawChartAction(BaseModel):
    chart_type: enum.ChartType
    chart_config: BarOrLineChart | ScatterChart | PieChart
    filter_conditions: list[FilterCondition] | None


class AddDatasetAction(BaseModel):
    dataset: Dataset

    def apply(self, env: BaseEnvironment, client: OpenAI, task_context: Task) -> None:
        env.add_dataset(client, self.dataset, task_context.instruction)


class AddColumn(DataProcessingAction):
    source_dataset_id: int
    dst_dataset_id: int
    column: Column
    expression: str

    def apply(self, env: BaseEnvironment, client: OpenAI, task_context: Task) -> None:
        env.add_column(
            client,
            self.source_dataset_id,
            self.dst_dataset_id,
            self.expression,
            task_context.instruction,
        )


class RemoveColumn(DataProcessingAction):
    source_dataset_id: int
    dst_dataset_id: int
    keep: list[str] | None
    remove: list[str] | None


class ChangeColumnType(DataProcessingAction):
    source_dataset_id: int
    dst_dataset_id: int
    column_name: str
    new_type: enum.ColumnType


class Join(DataProcessingAction):
    left_dataset_id: int
    right_dataset_id: int
    dst_dataset_id: int
    left_on: list[str]
    right_on: list[str]
    how: enum.JoinMethod
    right_columns_to_keep: list[str] | None

    def apply(self, env: BaseEnvironment, client: OpenAI, task_context: Task) -> None:
        env.join(
            client,
            self.left_dataset_id,
            self.right_dataset_id,
            self.left_on,
            self.right_on,
            self.dst_dataset_id,
            self.how,
        )


class Union(DataProcessingAction):
    source_dataset_ids: list[int]
    dst_dataset_id: int


class Pivot(DataProcessingAction):
    source_dataset_id: int
    dst_dataset_id: int
    index: str
    columns: str
    values: str


class Aggregate(DataProcessingAction):
    source_dataset_id: int
    dst_dataset_id: int
    group_by: list[str]
    agg: enum.AggregateFunction
    column: list[str]


class Filter(DataProcessingAction):
    source_dataset_id: int
    dst_dataset_id: int
    conditions: list[FilterCondition]


tools_data_processing = [
    openai.pydantic_function_tool(cls)
    for cls in [
        AddColumn,
        RemoveColumn,
        ChangeColumnType,
        Aggregate,
        Filter,
        Join,
        Union,
        Pivot,
    ]
]

tools_add_dataset = [openai.pydantic_function_tool(cls) for cls in [AddDatasetAction]]

tools_data_visualization = [
    openai.pydantic_function_tool(cls) for cls in [DrawChartAction]
]

_PROMPT_TASK_DETAIL = """
The user is attempting to perform data analysis based on the datasets and is executing tasks according to the plan.

## Datasets
{datasets}

## User requests
{user_request}

## Overall plan
{plan}

## Next task
You should generate specific actions to accomplish the following tasks that you are about to perform.
Generate considering that the user's native language is {language}.

{next_task}
"""


def build_action_prompt(
    datasets: Datasets, user_prompt: str, plan: Plan, language: enum.Language
) -> str:
    next_task = plan.next_task
    if next_task is None:
        raise ValueError("next_task is None")
    return _PROMPT_TASK_DETAIL.format(
        datasets=datasets.model_dump_json(indent=2),
        user_request=user_prompt,
        plan=plan.model_dump_json(indent=2),
        next_task=next_task.model_dump_json(indent=2),
        language=language.value,
    )


def generate_action(
    client: OpenAI,
    datasets: Datasets,
    plan: Plan,
    user_prompt: str,
    language: enum.Language = enum.Language.JAPANESE,
) -> Sequence[Action]:
    next_task = plan.next_task
    if next_task is None:
        raise ValueError("next_task is None")

    if next_task.task_type == TaskType.ADD_DATA:
        tools = tools_add_dataset
    elif next_task.task_type == TaskType.DATA_PREPROCESSING:
        tools = tools_data_processing
    elif next_task.task_type == TaskType.DATA_VISUALIZATION:
        tools = tools_data_visualization
    else:
        raise ValueError(f"Unknown task type: {next_task.task_type}")

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "system",
                "content": build_action_prompt(datasets, user_prompt, plan, language),
            },
            {"role": "user", "content": user_prompt},
        ],
        tool_choice="required",
        tools=tools,
    )

    message = completion.choices[0].message

    return [tool_call.function.parsed_arguments for tool_call in message.tool_calls]  # type: ignore
