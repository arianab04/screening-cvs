"""Herramientas de los agentes. Entradas y salidas validadas con Pydantic."""

from langchain_core.tools import tool

from app.rag.service import get_retriever
from app.schemas import Candidate, CandidateScore, ScoreBreakdown, ScoreInput, SearchCandidatesInput

_LEVEL_POINTS = {"básico": 0, "intermedio": 10, "avanzado": 20}
_BI_POINTS = {"Power BI": 15, "Tableau": 11.25}
_BI_DEFAULT = 7.5
_STATISTICS_POINTS = {"avanzada": 10, "intermedia": 5, "básica/intermedia": 2.5}
_EDUCATION_POINTS = {"Licenciatura en Economía": 10, "Ingeniería Industrial": 7.5}
_EDUCATION_DEFAULT = 5


def score_candidate(candidate: Candidate) -> CandidateScore:
    """Score determinístico sobre 100: experiencia 20, Python 20, SQL 20, BI 15, estadística 10, formación 10, sector 5."""
    breakdown = ScoreBreakdown(
        experience=round(min(candidate.experience_years / 4, 1) * 20, 2),
        python=_LEVEL_POINTS[candidate.python],
        sql=_LEVEL_POINTS[candidate.sql],
        bi=_BI_POINTS.get(candidate.bi, _BI_DEFAULT),
        statistics=_STATISTICS_POINTS.get(candidate.statistics, 0),
        education=_EDUCATION_POINTS.get(candidate.education, _EDUCATION_DEFAULT),
        financial_sector=5 if candidate.financial_sector else 0,
    )
    return CandidateScore(name=candidate.name, score=round(sum(breakdown.model_dump().values()), 2), breakdown=breakdown)


@tool(args_schema=SearchCandidatesInput)
async def search_candidates(query: str, top_k: int = 5, min_experience_years: int | None = None) -> list[dict]:
    """Busca en la base de CVs (RAG híbrido sobre Pinecone) los candidatos más relevantes para la vacante.
    Devuelve el perfil estructurado de cada candidato. Permite filtrar por años mínimos de experiencia."""
    candidates = await get_retriever().search(query, top_k=top_k, min_experience_years=min_experience_years)
    return [c.model_dump() for c in candidates]


@tool(args_schema=ScoreInput)
async def calculate_candidate_score(candidate: Candidate) -> dict:
    """Calcula de forma determinística el score (0-100) de un candidato con su desglose por criterio."""
    return score_candidate(Candidate.model_validate(candidate)).model_dump()
