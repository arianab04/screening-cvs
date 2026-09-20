"""API HTTP (FastAPI). Solo encola trabajo y consulta estado: el razonamiento ocurre en el worker."""

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request, status

from app.config import get_settings
from app.schemas import (
    ApprovalAccepted,
    ApprovalRequest,
    HealthResponse,
    JobStatus,
    ResumeMessage,
    TaskCreated,
    TaskDetail,
    TaskRequest,
)
from app.store import JobStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.store = JobStore.from_url(settings.redis_url, settings.queue_name)
    yield
    await app.state.store.aclose()


app = FastAPI(
    title="Screening CVs API",
    description=(
        "Sistema multi-agente de screening de CVs: RAG (Pinecone) + supervisor LangGraph + "
        "aprobación humana (human-in-the-loop). Flujo: `POST /tasks` -> `GET /tasks/{id}` -> "
        "`POST /tasks/{id}/approve` o `/reject`."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


def get_store(request: Request) -> JobStore:
    return request.app.state.store


Store = Annotated[JobStore, Depends(get_store)]


@app.get("/health", response_model=HealthResponse, tags=["sistema"])
async def health(store: Store) -> HealthResponse:
    redis_ok = await store.ping()
    return HealthResponse(status="ok" if redis_ok else "degraded", redis=redis_ok)


@app.post("/tasks", response_model=TaskCreated, status_code=status.HTTP_202_ACCEPTED, tags=["tareas"])
async def create_task(task: TaskRequest, store: Store) -> TaskCreated:
    """Encola una vacante para ser evaluada por el grafo de agentes."""
    job_id = str(uuid4())
    await store.create_job(job_id, task)
    return TaskCreated(job_id=job_id, status=JobStatus.PENDING, message="Tarea encolada.")


@app.get("/tasks/{job_id}", response_model=TaskDetail, response_model_exclude_none=True, tags=["tareas"])
async def get_task(job_id: str, store: Store) -> TaskDetail:
    """Estado del job. Con `WAITING_APPROVAL` incluye el informe listo para revisar."""
    job = await store.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job no encontrado.")
    return TaskDetail(
        job_id=job_id,
        status=job["status"],
        result=job.get("result"),
        validation=job.get("validation"),
        scores=json.loads(job["scores"]) if job.get("scores") else None,
        decision=job.get("decision"),
        comment=job.get("comment") or None,
        error=job.get("error"),
    )


async def _resume(job_id: str, approved: bool, comment: str, store: JobStore) -> ApprovalAccepted:
    job = await store.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job no encontrado.")
    if not await store.transition(job_id, JobStatus.WAITING_APPROVAL, JobStatus.RUNNING):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"La tarea no está esperando aprobación. Estado actual: {(await store.get(job_id) or job)['status']}",
        )
    await store.enqueue(ResumeMessage(job_id=job_id, approved=approved, comment=comment))
    return ApprovalAccepted(
        job_id=job_id,
        status=JobStatus.RUNNING,
        message="Decisión recibida; el worker reanuda el grafo desde su checkpoint.",
    )


@app.post("/tasks/{job_id}/approve", response_model=ApprovalAccepted, status_code=202, tags=["tareas"])
async def approve_task(job_id: str, store: Store, body: ApprovalRequest | None = None) -> ApprovalAccepted:
    """Aprueba el informe (human-in-the-loop) y reanuda el grafo."""
    return await _resume(job_id, True, body.comment if body else "", store)


@app.post("/tasks/{job_id}/reject", response_model=ApprovalAccepted, status_code=202, tags=["tareas"])
async def reject_task(job_id: str, store: Store, body: ApprovalRequest | None = None) -> ApprovalAccepted:
    """Rechaza el informe: el job termina en estado REJECTED."""
    return await _resume(job_id, False, body.comment if body else "", store)
