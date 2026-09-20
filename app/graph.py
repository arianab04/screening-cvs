"""Grafo LangGraph: supervisor + agentes especializados, con checkpointer persistente en Redis."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.analyst import analyst_node
from app.agents.approval import human_approval_node
from app.agents.research import research_node
from app.agents.supervisor import route_from_supervisor, supervisor_node
from app.agents.validation import validation_node
from app.config import get_settings
from app.state import AgentState


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("research", research_node)
    builder.add_node("analyst", analyst_node)
    builder.add_node("validation", validation_node)
    builder.add_node("approval", human_approval_node)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {name: name for name in ("research", "analyst", "validation", "approval")} | {"end": END},
    )
    for specialist in ("research", "analyst", "validation", "approval"):
        builder.add_edge(specialist, "supervisor")

    return builder.compile(checkpointer=checkpointer)


@asynccontextmanager
async def open_graph(redis_url: str | None = None) -> AsyncIterator[CompiledStateGraph]:
    """Compila el grafo con AsyncRedisSaver: las conversaciones sobreviven a reinicios del proceso."""
    url = redis_url or get_settings().redis_url
    async with AsyncRedisSaver.from_conn_string(url) as saver:
        await saver.asetup()
        yield build_graph(saver)
