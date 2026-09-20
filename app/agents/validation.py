"""Validation: chequeos determinísticos del informe contra los datos y los scores calculados."""

from app.schemas import CandidateScore
from app.state import AgentState


def check_analysis(analysis: str, scores: list[CandidateScore]) -> str | None:
    """Devuelve el motivo del rechazo, o None si el informe es consistente."""
    if not scores:
        return "No se encontraron candidatos."
    if not analysis.strip():
        return "El análisis está vacío."
    if "Candidato recomendado" not in analysis:
        return "Falta la sección 'Candidato recomendado'."

    missing = [s.name for s in scores if s.name not in analysis]
    if missing:
        return f"El informe no menciona a: {', '.join(missing)}."

    top = scores[0]
    recommended_section = analysis.split("Candidato recomendado", 1)[1]
    next_heading = recommended_section.find("###")
    if next_heading != -1:
        recommended_section = recommended_section[:next_heading]
    if top.name not in recommended_section:
        return f"El candidato recomendado debe ser {top.name} (mayor score: {top.score})."
    return None


async def validation_node(state: AgentState) -> dict:
    scores = [CandidateScore.model_validate(s) for s in state["scores"]]
    problem = check_analysis(state["analysis_output"], scores)
    return {
        "validation": problem or "El análisis contiene la información necesaria.",
        "task_completed": problem is None,
    }
