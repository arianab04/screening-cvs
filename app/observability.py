"""Trazabilidad con LangSmith.

Con LANGSMITH_TRACING=true y LANGSMITH_API_KEY, LangChain/LangGraph envían automáticamente el árbol de
spans (supervisor, agentes, tools, llamadas al LLM, tokens y costo). Acá solo se estandariza el
`RunnableConfig` de cada ejecución para poder ubicar un job en el dashboard por su job_id.
"""

import logging

from langchain_core.runnables import RunnableConfig

from app.config import get_settings

log = logging.getLogger(__name__)


def configure_tracing() -> None:
    settings = get_settings()
    if settings.langsmith_tracing and not settings.langsmith_api_key:
        log.warning("LANGSMITH_TRACING=true pero falta LANGSMITH_API_KEY: no se enviarán trazas.")
    elif settings.langsmith_tracing:
        log.info("Trazas activadas en LangSmith (proyecto: %s).", settings.langsmith_project)
    else:
        log.info("Trazas desactivadas (LANGSMITH_TRACING != true).")


def run_config(job_id: str, phase: str) -> RunnableConfig:
    """`thread_id` enlaza la ejecución con su checkpoint; run_name/tags/metadata la hacen buscable en LangSmith."""
    return {
        "configurable": {"thread_id": job_id},
        "run_name": f"screening-cvs-{phase}",
        "tags": ["screening-cvs", phase],
        "metadata": {"job_id": job_id, "phase": phase},
    }
