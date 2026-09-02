import os

from dotenv import load_dotenv
from langsmith import traceable


load_dotenv()


@traceable(
    name="screening-cvs-job",
    project_name=os.getenv(
        "LANGSMITH_PROJECT",
        "screening-cvs"
    ),
)
def trace_graph_execution(graph, initial_state, config):
    """
    Ejecuta el grafo y registra la ejecución en LangSmith.
    """

    return graph.invoke(
        initial_state,
        config=config,
    )