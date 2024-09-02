from logging import getLogger

from openai import OpenAI

from data_analysis_agent.actions import generate_action
from data_analysis_agent.environment import BaseEnvironment
from data_analysis_agent.planner import Plan, generate_plan

_logger = getLogger(__name__)


def process_next_action(
    client: OpenAI, env: BaseEnvironment, user_prompt: str, plan: Plan
):
    next_actions = generate_action(client, env.list_datasets(), plan, user_prompt)

    for action in next_actions:
        _logger.info(f"next action: {action.dict()}")
        action.apply(env, client, plan.next_task)

    _logger.info("done")
    plan.next_task_id = plan.next_task.task_id + 1


def execute(client: OpenAI, env: BaseEnvironment, user_prompt: str):
    plan = generate_plan(client, env.list_datasets(), user_prompt)

    _logger.info(plan.pretty())

    while not plan.finished:
        process_next_action(client, env, user_prompt, plan)
