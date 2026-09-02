import json
import os
from uuid import uuid4

import redis
from fastapi import FastAPI, HTTPException
from langgraph.types import Command
from pydantic import BaseModel

from app.graph import graph


app = FastAPI(
    title="Screening CVS API",
    description="API para ejecutar el sistema multi-agente de screening.",
    version="1.0.0",
)


# --------------------------------------------------
# REDIS
# --------------------------------------------------

REDIS_URL = os.getenv(
    "REDIS_URL",
    "redis://localhost:6379"
)

redis_client = redis.from_url(
    REDIS_URL,
    decode_responses=True
)

QUEUE_NAME = "screening_tasks"


# --------------------------------------------------
# REQUEST MODEL
# --------------------------------------------------

class TaskRequest(BaseModel):
    role: str
    experience: str = ""
    python: str = ""
    sql: str = ""
    bi: str = ""
    statistics: str = ""
    financial_sector: str = ""


# --------------------------------------------------
# CREATE TASK
# --------------------------------------------------

@app.post("/tasks")
async def create_task(task: TaskRequest):

    job_id = str(uuid4())
    job_key = f"task:{job_id}"

    job_data = task.model_dump()

    redis_client.hset(
        job_key,
        mapping={
            "status": "PENDING",
            "task": json.dumps(job_data),
        },
    )

    redis_client.rpush(
        QUEUE_NAME,
        json.dumps({
            "job_id": job_id,
            "job_data": job_data,
        }),
    )

    print(
        f"API -> tarea enviada a cola: {job_id}"
    )

    return {
        "job_id": job_id,
        "status": "PENDING",
        "message": "Tarea recibida correctamente.",
    }


# --------------------------------------------------
# GET TASK STATUS
# --------------------------------------------------

@app.get("/tasks/{job_id}")
async def get_task(job_id: str):

    job_key = f"task:{job_id}"

    job = redis_client.hgetall(job_key)

    if not job:
        raise HTTPException(
            status_code=404,
            detail="Job no encontrado."
        )

    response = {
        "job_id": job_id,
        "status": job.get("status"),
    }

    if job.get("status") in [
        "WAITING_APPROVAL",
        "DONE",
    ]:
        response["result"] = job.get(
            "result",
            ""
        )

        response["validation"] = job.get(
            "validation",
            ""
        )

    elif job.get("status") == "FAILED":

        response["error"] = job.get(
            "error",
            "Error desconocido."
        )

    return response


# --------------------------------------------------
# HUMAN APPROVAL
# --------------------------------------------------

@app.post("/tasks/{job_id}/approve")
async def approve_task(job_id: str):

    job_key = f"task:{job_id}"

    job = redis_client.hgetall(job_key)

    if not job:
        raise HTTPException(
            status_code=404,
            detail="Job no encontrado."
        )

    status = job.get("status")

    if status != "WAITING_APPROVAL":
        raise HTTPException(
            status_code=409,
            detail=(
                f"La tarea no está esperando aprobación. "
                f"Estado actual: {status}"
            ),
        )

    try:

        print(
            f"API -> aprobando job {job_id}"
        )

        result = graph.invoke(
            Command(resume=True),
            config={
                "configurable": {
                    "thread_id": job_id
                }
            }
        )

        redis_client.hset(
            job_key,
            mapping={
                "status": "DONE",
                "result": result.get(
                    "analysis_output",
                    ""
                ),
                "validation": result.get(
                    "validation",
                    ""
                ),
            }
        )

        print(
            f"API -> job {job_id} aprobado y completado"
        )

        return {
            "job_id": job_id,
            "status": "DONE",
            "message": "Tarea aprobada correctamente.",
        }

    except Exception as e:

        redis_client.hset(
            job_key,
            mapping={
                "status": "FAILED",
                "error": str(e),
            }
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )