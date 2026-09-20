"""Research Agent: recupera candidatos mediante RAG usando la tool `search_candidates`."""

import logging

from pydantic import ValidationError

from app.agents.tools import search_candidates
from app.config import get_settings
from app.llm import get_llm
from app.schemas import Candidate, TaskRequest
from app.state import AgentState

log = logging.getLogger(__name__)


def _format(candidates: list[Candidate]) -> str:
    return "\n".join(
        f"- {c.name}: {c.role}, {c.experience_years} años, Python {c.python}, SQL {c.sql}, BI {c.bi}, "
        f"estadística {c.statistics}, sector financiero {'sí' if c.financial_sector else 'no'}"
        for c in candidates
    )


async def research_node(state: AgentState) -> dict:
    settings = get_settings()
    requirements = TaskRequest.model_validate(state["job_requirements"])
    attempts = state.get("research_attempts", 0)
    default_args = {"query": requirements.to_query(), "top_k": settings.retrieval_top_k}

    if attempts == 0:
        prompt = (
            "Sos un agente de investigación de selección de personal. Buscá candidatos para esta vacante "
            f"usando la herramienta search_candidates.\n\nVacante: {requirements.to_query()}\n\n"
            "No hagas scoring ni elijas al mejor candidato: solo recuperá los perfiles relevantes."
        )
        response = await get_llm().bind_tools([search_candidates], tool_choice="required").ainvoke(prompt)
        args = response.tool_calls[0]["args"] if response.tool_calls else default_args
    else:
        # Reintento: búsqueda ampliada, sin filtros de metadatos.
        args = {**default_args, "top_k": min(settings.retrieval_top_k * 2, 20)}

    try:
        raw = await search_candidates.ainvoke(args)
    except ValidationError:
        log.warning("Argumentos inválidos del LLM para search_candidates (%s); uso los de la vacante.", args)
        raw = await search_candidates.ainvoke(default_args)

    candidates = [Candidate.model_validate(c) for c in raw]
    return {
        "candidates": [c.model_dump() for c in candidates],
        "research_output": _format(candidates),
        "research_attempts": attempts + 1,
    }
