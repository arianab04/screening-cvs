"""Supervisor: decide a qué agente delegar.

Diseño "el LLM propone, las reglas validan":
- `legal_next` calcula, a partir del estado, las transiciones permitidas (máquina de estados con topes de reintentos).
- En modo `llm` el modelo elige entre las opciones legales con salida estructurada (Pydantic). Si elige algo
  ilegal, o falla, se aplica la primera opción legal. En modo `rules` se usa siempre esa opción (determinístico).
"""

import logging

from app.config import Settings, get_settings
from app.llm import get_llm
from app.schemas import NextAgent, SupervisorDecision
from app.state import AgentState

log = logging.getLogger(__name__)

_DESCRIPTIONS = {
    "research": "recuperar candidatos con RAG (Research Agent)",
    "analyst": "calcular scores y redactar el informe (Analyst Agent)",
    "validation": "validar el informe contra los scores (Validation)",
    "approval": "pedir aprobación humana (Human-in-the-loop)",
    "end": "terminar la ejecución",
}


def legal_next(state: AgentState, settings: Settings) -> list[NextAgent]:
    """Transiciones permitidas desde el estado actual, en orden de preferencia."""
    if not state["candidates"]:
        attempts = state["research_attempts"]
        if attempts == 0:
            return ["research"]
        return ["research", "end"] if attempts < settings.max_research_attempts else ["end"]
    if not state["analysis_output"]:
        return ["analyst"]
    if not state["validation"]:
        return ["validation"]
    if not state["task_completed"]:
        can_retry = state["analysis_attempts"] <= settings.max_analysis_retries
        return ["analyst", "end"] if can_retry else ["end"]
    if not state["human_decision"]:
        return ["approval"]
    return ["end"]


def _summary(state: AgentState) -> str:
    return (
        f"- candidatos recuperados: {len(state['candidates'])} (intentos de búsqueda: {state['research_attempts']})\n"
        f"- informe generado: {'sí' if state['analysis_output'] else 'no'} (intentos: {state['analysis_attempts']})\n"
        f"- validación: {state['validation'] or 'pendiente'}\n"
        f"- decisión humana: {state['human_decision'] or 'pendiente'}"
    )


async def _ask_llm(state: AgentState, options: list[NextAgent]) -> SupervisorDecision:
    menu = "\n".join(f"- {o}: {_DESCRIPTIONS[o]}" for o in options)
    prompt = (
        "Sos el supervisor de un sistema multi-agente de screening de CVs. Elegí el próximo paso.\n\n"
        f"Estado actual:\n{_summary(state)}\n\nOpciones permitidas:\n{menu}\n\n"
        "Elegí una sola opción y justificala en una frase."
    )
    return await get_llm().with_structured_output(SupervisorDecision).ainvoke(prompt)


async def supervisor_node(state: AgentState) -> dict:
    settings = get_settings()
    options = legal_next(state, settings)
    choice, reason = options[0], "regla determinística"

    if settings.supervisor_mode == "llm":
        try:
            decision = await _ask_llm(state, options)
            if decision.next_agent in options:
                choice, reason = decision.next_agent, decision.reason
            else:
                reason = f"guardrail: el LLM propuso '{decision.next_agent}', no permitido; se usa '{choice}'"
                log.warning(reason)
        except Exception:  # noqa: BLE001 - el supervisor nunca debe romper el flujo
            log.exception("Fallo el supervisor LLM; se usa la regla determinística.")
            reason = "fallback: error del LLM"

    log.info("SUPERVISOR -> %s (%s)", choice, reason)
    return {"next_agent": choice, "supervisor_reason": reason}


def route_from_supervisor(state: AgentState) -> str:
    return state["next_agent"]
