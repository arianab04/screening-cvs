"""Analyst Agent: scores determinísticos (tool) + informe generado por el LLM."""

import asyncio

from app.agents.tools import calculate_candidate_score
from app.llm import get_llm
from app.schemas import Candidate, CandidateScore
from app.state import AgentState

_PROMPT = """Sos un analista especializado en selección de personal.

Vacante:
{requirements}

Los scores fueron calculados por una herramienta determinística de Python.

REGLAS:
- NO recalcules ni modifiques los scores: son la fuente de verdad.
- NO inventes candidatos ni características. Usá TODOS los candidatos recibidos.
- El candidato con mayor score es el recomendado.
- Para fortalezas y brechas usá únicamente los datos originales. Indicá explícitamente si tiene o no
  experiencia en el sector financiero, sin confundir un requisito deseable con uno obligatorio.

DATOS ORIGINALES DE LOS CANDIDATOS:
{candidates}

SCORES (ordenados de mayor a menor):
{scores}
{feedback}
Generá un informe en Markdown con exactamente estas secciones:

### 1. Ranking completo de candidatos
### 2. Score de cada candidato
### 3. Candidato recomendado
### 4. Justificación de la recomendación
### 5. Fortalezas del candidato recomendado
### 6. Principales brechas del candidato recomendado
### 7. Breve comparación con los demás candidatos
"""


async def analyst_node(state: AgentState) -> dict:
    candidates = [Candidate.model_validate(c) for c in state["candidates"]]

    results = await asyncio.gather(*(calculate_candidate_score.ainvoke({"candidate": c}) for c in candidates))
    scores = sorted((CandidateScore.model_validate(r) for r in results), key=lambda s: s.score, reverse=True)

    feedback = ""
    if state.get("validation"):
        feedback = f"\nATENCIÓN: el informe anterior fue rechazado por esta razón: {state['validation']} Corregilo.\n"

    prompt = _PROMPT.format(
        requirements=state["job_requirements"],
        candidates=[c.model_dump() for c in candidates],
        scores=[s.model_dump() for s in scores],
        feedback=feedback,
    )
    response = await get_llm().ainvoke(prompt)

    return {
        "scores": [s.model_dump() for s in scores],
        "analysis_output": str(response.content),
        "analysis_attempts": state.get("analysis_attempts", 0) + 1,
        "validation": "",
        "task_completed": False,
    }
