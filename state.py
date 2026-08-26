from typing import TypedDict
from langgraph.graph import MessagesState


class AgentState(MessagesState):
    job_requirements: dict
    candidates: list
    research_output: str
    analysis_output: str
    validation: str
    next_agent: str
    task_completed: bool