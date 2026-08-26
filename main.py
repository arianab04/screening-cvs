from typing import Literal, TypedDict

from langgraph.graph import StateGraph, START, END

from state import AgentState
from agents.research_agent import research_node
from agents.analyst_agent import analyst_node


# --------------------------------------------------
# TIPOS DE ROUTING DEL SUPERVISOR
# --------------------------------------------------

NextAgent = Literal[
    "research",
    "analyst",
    "validation",
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

    El retorno utiliza Literal para limitar las decisiones
    posibles a los nombres válidos de los nodos del grafo.
    """

    # Todavía no tenemos candidatos.
    if not state["candidates"]:
        next_agent: NextAgent = "research"

    # Tenemos candidatos pero todavía no tenemos análisis.
    elif not state["analysis_output"]:
        next_agent = "analyst"

    # Tenemos análisis pero todavía no fue validado.
    elif not state["validation"]:
        next_agent = "validation"

    # La validación detectó un problema.
    elif not state["task_completed"]:
        next_agent = "analyst"

    # Todo está completo.
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
    """
    Lee la decisión del Supervisor y determina
    hacia qué nodo debe dirigirse el grafo.
    """

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


# --------------------------------------------------
# COMPILACIÓN
# --------------------------------------------------

graph = builder.compile()


# --------------------------------------------------
# EJECUCIÓN DE LA DEMO
# --------------------------------------------------

if __name__ == "__main__":

    initial_state: AgentState = {
        "messages": [],

        "job_requirements": {
            "role": "Data Analyst",
            "experience": "2+ años",
            "python": "intermedio/avanzado",
            "sql": "intermedio/avanzado",
            "bi": "Power BI o equivalente",
            "statistics": "conocimientos aplicados",
            "financial_sector": "deseable",
        },

        "candidates": [],

        "research_output": "",

        "analysis_output": "",

        "validation": "",

        "next_agent": "",

        "task_completed": False,
    }

    print("\n")
    print("=" * 60)
    print("ORQUESTADOR MULTI-AGENTE DE RRHH")
    print("=" * 60)

    print("\nSolicitud:")
    print(
        "Seleccionar el mejor candidato para una posición "
        "de Data Analyst."
    )

    result = graph.invoke(initial_state)

    print("\n")
    print("=" * 60)
    print("RESULTADO FINAL")
    print("=" * 60)

    print(result["analysis_output"])

    print("\n")
    print("=" * 60)
    print("VALIDACIÓN")
    print("=" * 60)

    print(result["validation"])