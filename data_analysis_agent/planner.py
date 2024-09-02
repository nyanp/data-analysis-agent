from enum import Enum

import openai
from openai import OpenAI
from pydantic import BaseModel, Field

from data_analysis_agent.enum import Language
from data_analysis_agent.struct import Datasets

_PLANNING_PROMPT = """
You are responsible for performing data analysis on one or more given datasets to achieve the goals requested by the user. 
First, generate an execution plan that should be followed to achieve the goal.

The execution plan should consist of one or more sequential task calls. 
Each task should be broken down into as fine a granularity as possible and contain only a single, concrete operation.

In the data visualization tasks of this system, you can perform aggregated visualizations by specifying dimensions and
facts from multiple datasets, similar to BI tools. Therefore, there is no need to join datasets or pre-aggregate
metrics solely for visualization purposes. However, you must complete tasks such as data type conversion, item
extraction, and value transformation of columns in advance.

Additionally, aggregation is limited to simple calculations like sums and averages, and other advanced aggregations
require preprocessing of the data.

The instructions to be generated should be in {language}.

## Tasks
- add_data: If the current dataset is insufficient to meet the user's instructions, register a new dataset.
- data_preprocessing: Perform data preprocessing.
- data_visualization: Perform data visualization.

## Datasets
{datasets}
"""


class TaskType(str, Enum):
    DATA_PREPROCESSING = "data_preprocessing"
    DATA_VISUALIZATION = "data_visualization"
    ADD_DATA = "add_data"


class Task(BaseModel):
    task_id: int = Field(..., description="Task ID (sequential number)")
    task_type: TaskType = Field(..., description="Task Overview")
    instruction: str = Field(
        ...,
        description="A summary of the process to be performed in this task. It should be a short phrase or sentence",
    )

    def pretty(self) -> str:
        return f"[type={self.task_type.value}] {self.instruction}"


class Plan(BaseModel):
    tasks: list[Task]
    next_task_id: int = Field(..., description="ID of the next task to be executed")

    @property
    def next_task(self) -> Task | None:
        for task in self.tasks:
            if task.task_id == self.next_task_id:
                return task
        return None

    @property
    def finished(self) -> bool:
        max_task_id = max([task.task_id for task in self.tasks])
        return self.next_task_id > max_task_id

    def pretty(self) -> str:
        tasks = [f"Step {i + 1}: {task.pretty()}" for i, task in enumerate(self.tasks)]
        return "\n".join(tasks)


class ClarificationQuestion(BaseModel):
    question: str


def generate_plan(
    client: OpenAI,
    datasets: Datasets,
    user_prompt: str,
    language: Language = Language.JAPANESE,
) -> Plan:
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": _build_planning_prompt(datasets, language)},
            {"role": "user", "content": user_prompt},
        ],
        response_format=Plan,
    )

    plan = completion.choices[0].message.parsed
    assert isinstance(plan, Plan)
    if not plan.next_task:
        plan.next_task_id = plan.tasks[0].task_id
    return plan


def generate_plan_with_clarifications(
    client: OpenAI,
    datasets: Datasets,
    user_prompt: str,
    language: Language = Language.JAPANESE,
) -> Plan | ClarificationQuestion:
    prompt = _build_planning_prompt(datasets, language)

    prompt += """
If it is possible to plan based on the user's request, generate `Tasks`.
If the user's request is ambiguous, generate a `ClarificationQuestion` and ask the user for clarification
(for example, what metrics to aggregate, etc.).
"""

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_prompt},
        ],
        tools=[
            openai.pydantic_function_tool(cls) for cls in [Plan, ClarificationQuestion]
        ],
        tool_choice="required",
        parallel_tool_calls=False,
    )

    if not completion.choices[0].message.tool_calls:
        raise ValueError("tool_calls not invoked")

    parsed = completion.choices[0].message.tool_calls[0].function.parsed_arguments

    assert isinstance(parsed, Plan) or isinstance(parsed, ClarificationQuestion)

    return parsed


def _build_planning_prompt(datasets: Datasets, language: Language) -> str:
    return _PLANNING_PROMPT.format(
        datasets=datasets.model_dump_json(indent=2), language=language.value
    )
