import json
import os

import redis

from app.graph import graph
from app.observability import trace_graph_execution


REDIS_URL = os.getenv(
    "REDIS_URL",
    "redis://localhost:6379"
)

redis_client = redis.from_url(
    REDIS_URL,
    decode_responses=True
)

QUEUE_NAME = "screening_tasks"


def run_task(job_id: str, job_data: dict):

    job_key = f"task:{job_id}"

    try:

        redis_client.hset(
            job_key,
            mapping={
                "status": "RUNNING"
            }
        )

        print(f"WORKER -> ejecutando job {job_id}")

        initial_state = {
            "messages": [],
            "job_requirements": job_data,
            "candidates": [],
            "research_output": "",
            "analysis_output": "",
            "validation": "",
            "next_agent": "",
            "task_completed": False,
            "human_approved": False,
        }

        result = trace_graph_execution(
            graph,
            initial_state,
            {
                "configurable": {
                    "thread_id": job_id
                }
            }
        )

        # --------------------------------------------------
        # HUMAN-IN-THE-LOOP
        # --------------------------------------------------

        if result.get("__interrupt__"):

            redis_client.hset(
                job_key,
                mapping={
                    "status": "WAITING_APPROVAL",
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
                f"WORKER -> job {job_id} "
                "esperando aprobación humana"
            )

            return

        # --------------------------------------------------
        # TAREA COMPLETADA
        # --------------------------------------------------

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
            f"WORKER -> job {job_id} completado"
        )

    except Exception as e:

        redis_client.hset(
            job_key,
            mapping={
                "status": "FAILED",
                "error": str(e),
            }
        )

        print(
            f"WORKER -> error en job {job_id}: {e}"
        )


def worker_loop():

    print("WORKER -> esperando tareas...")

    while True:

        try:

            _, payload = redis_client.blpop(
                QUEUE_NAME
            )

            task = json.loads(payload)

            job_id = task["job_id"]
            job_data = task["job_data"]

            run_task(
                job_id,
                job_data
            )

        except Exception as e:

            print(
                f"WORKER -> error general: {e}"
            )


if __name__ == "__main__":
    worker_loop()