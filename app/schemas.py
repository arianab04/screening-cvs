"""Contratos de datos (Pydantic) de cada frontera: API, cola, herramientas de agentes y supervisor."""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------
# Dominio
# --------------------------------------------------------------------------

Level = Literal["básico", "intermedio", "avanzado"]


class Candidate(BaseModel):
    """Perfil estructurado de un candidato (front matter del CV indexado en Pinecone)."""

    model_config = ConfigDict(extra="ignore")

    name: str
    education: str
    experience_years: Annotated[int, Field(ge=0, le=60)]
    role: str
    python: Level
    sql: Level
    bi: str
    statistics: str
    financial_sector: bool
    machine_learning: str = "ninguno"


class ScoreBreakdown(BaseModel):
    experience: float
    python: float
    sql: float
    bi: float
    statistics: float
    education: float
    financial_sector: float


class CandidateScore(BaseModel):
    name: str
    score: float
    breakdown: ScoreBreakdown


# --------------------------------------------------------------------------
# Herramientas de los agentes
# --------------------------------------------------------------------------


class SearchCandidatesInput(BaseModel):
    """Entrada de la tool `search_candidates` (RAG sobre los CVs)."""

    query: str = Field(min_length=3, description="Descripción de la vacante y del perfil buscado.")
    top_k: int = Field(default=5, ge=1, le=20, description="Cantidad máxima de candidatos a devolver.")
    min_experience_years: int | None = Field(
        default=None, ge=0, le=60, description="Filtro por metadatos: años mínimos de experiencia."
    )


class ScoreInput(BaseModel):
    """Entrada de la tool `calculate_candidate_score`."""

    candidate: Candidate


# --------------------------------------------------------------------------
# Supervisor
# --------------------------------------------------------------------------

NextAgent = Literal["research", "analyst", "validation", "approval", "end"]


class SupervisorDecision(BaseModel):
    """Salida estructurada del supervisor."""

    next_agent: NextAgent
    reason: str = Field(default="", description="Justificación breve de la decisión.")


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------


class JobStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    DONE = "DONE"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


class TaskRequest(BaseModel):
    """Vacante a evaluar."""

    model_config = ConfigDict(extra="forbid")

    role: str = Field(min_length=2, max_length=120, examples=["Data Analyst"])
    experience: str = Field(default="", max_length=200, examples=["2+ años"])
    python: str = Field(default="", max_length=200, examples=["intermedio/avanzado"])
    sql: str = Field(default="", max_length=200, examples=["intermedio/avanzado"])
    bi: str = Field(default="", max_length=200, examples=["Power BI o equivalente"])
    statistics: str = Field(default="", max_length=200, examples=["conocimientos aplicados"])
    financial_sector: str = Field(default="", max_length=200, examples=["deseable"])

    def to_query(self) -> str:
        parts = [f"Rol: {self.role}"]
        for label, value in (
            ("Experiencia", self.experience),
            ("Python", self.python),
            ("SQL", self.sql),
            ("BI", self.bi),
            ("Estadística", self.statistics),
            ("Sector financiero", self.financial_sector),
        ):
            if value:
                parts.append(f"{label}: {value}")
        return ". ".join(parts)


class TaskCreated(BaseModel):
    job_id: str
    status: JobStatus
    message: str


class TaskDetail(BaseModel):
    job_id: str
    status: JobStatus
    result: str | None = None
    validation: str | None = None
    scores: list[CandidateScore] | None = None
    decision: str | None = None
    comment: str | None = None
    error: str | None = None


class ApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    comment: str = Field(default="", max_length=500)


class ApprovalAccepted(BaseModel):
    job_id: str
    status: JobStatus
    message: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    redis: bool


# --------------------------------------------------------------------------
# Cola Redis (API -> worker)
# --------------------------------------------------------------------------


class RunMessage(BaseModel):
    type: Literal["run"] = "run"
    job_id: str
    job_data: TaskRequest


class ResumeMessage(BaseModel):
    type: Literal["resume"] = "resume"
    job_id: str
    approved: bool
    comment: str = ""


QueueMessage = Annotated[RunMessage | ResumeMessage, Field(discriminator="type")]
