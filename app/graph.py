import os
from typing import Literal, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.redis import RedisSaver

from state import AgentState
from agents.research_agent import research_node
from agents.analyst_agent import analyst_node
from app.hitl import human_approval_node


# --------------------------------------------------
# TIPOS DE ROUTING DEL SUPERVISOR
# --------------------------------------------------

NextAgent = Literal[
    "research",
    "analyst",
    "validation",
    "approval",
    "end",
]


class SupervisorDecision(TypedDict):
    next_agent: NextAgent


# --------------------------------------------------
# VALIDATION NODE
# --------------------------------------------------

def validation_node(state: AgentState):
    """
    Verifica si el análisis generado por el Analyst
    contiene la información mínima necesaria.
    """

    analysis = state["analysis_output"]

    print("\n✅ VALIDATION → verificando resultado...")

    if not analysis:
        validation = "El análisis está vacío."
        task_completed = False

    elif "Candidato recomendado" not in analysis:
        validation = "Falta identificar un candidato recomendado."
        task_completed = False

    elif "Score" not in analysis:
        validation = "Falta presentar los scores."
        task_completed = False

    elif not state["candidates"]:
        validation = "No se encontraron candidatos."
        task_completed = False

    else:
        validation = "El análisis contiene la información necesaria."
        task_completed = True

    print(f"✅ VALIDATION → {validation}")

    return {
        "validation": validation,
        "task_completed": task_completed,
    }


# --------------------------------------------------
# SUPERVISOR NODE
# --------------------------------------------------

def supervisor_node(state: AgentState) -> SupervisorDecision:
    """
    Decide qué nodo debe ejecutarse a continuación.
    """

    if not state["candidates"]:
        next_agent: NextAgent = "research"

    elif not state["analysis_output"]:
        next_agent = "analyst"

    elif not state["validation"]:
        next_agent = "validation"

    elif not state["task_completed"]:
        next_agent = "analyst"

    elif not state["human_approved"]:
        next_agent = "approval"

    else:
        next_agent = "end"

    print(f"\n🧠 SUPERVISOR → {next_agent}")

    return {
        "next_agent": next_agent
    }


# --------------------------------------------------
# ROUTING
# --------------------------------------------------

def route_from_supervisor(state: AgentState) -> NextAgent:
    return state["next_agent"]


# --------------------------------------------------
# CONSTRUCCIÓN DEL GRAFO
# --------------------------------------------------

builder = StateGraph(AgentState)


# --------------------------------------------------
# NODOS
# --------------------------------------------------

builder.add_node("supervisor", supervisor_node)
builder.add_node("research", research_node)
builder.add_node("analyst", analyst_node)
builder.add_node("validation", validation_node)
builder.add_node("approval", human_approval_node)


# --------------------------------------------------
# START
# --------------------------------------------------

builder.add_edge(
    START,
    "supervisor"
)


# --------------------------------------------------
# CONDITIONAL EDGES DEL SUPERVISOR
# --------------------------------------------------

builder.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {
        "research": "research",
        "analyst": "analyst",
        "validation": "validation",
        "approval": "approval",
        "end": END,
    }
)


# --------------------------------------------------
# RETORNO DE LOS ESPECIALISTAS
# --------------------------------------------------

builder.add_edge(
    "research",
    "supervisor"
)

builder.add_edge(
    "analyst",
    "supervisor"
)

builder.add_edge(
    "validation",
    "supervisor"
)

builder.add_edge(
    "approval",
    "supervisor"
)


# --------------------------------------------------
# REDIS CHECKPOINTER
# --------------------------------------------------

REDIS_URL = os.getenv(
    "REDIS_URL",
    "redis://localhost:6379"
)

_checkpointer_context = RedisSaver.from_conn_string(
    REDIS_URL
)

checkpointer = _checkpointer_context.__enter__()

checkpointer.setup()


# --------------------------------------------------
# COMPILACIÓN
# --------------------------------------------------

graph = builder.compile(
    checkpointer=checkpointer
)