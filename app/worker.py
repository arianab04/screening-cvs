"""Worker async: consume la cola de Redis y ejecuta/reanuda el grafo, con N jobs en simultáneo."""

import asyncio
import logging
import signal
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from app.config import get_settings
from app.graph import open_graph
from app.observability import configure_tracing, run_config
from app.rag.service import close_retriever
from app.schemas import JobStatus, QueueMessage, ResumeMessage, RunMessage
from app.state import initial_state
from app.store import JobStore

log = logging.getLogger("worker")


async def _record_outcome(store: JobStore, job_id: str, result: dict[str, Any]) -> None:
    """Traduce el resultado del grafo (pausado o terminado) al estado del job."""
    payload = {
        "result": result.get("analysis_output", ""),
        "validation": result.get("validation", ""),
        "scores": result.get("scores", []),
    }
    if result.get("__interrupt__"):
        await store.update(job_id, status=JobStatus.WAITING_APPROVAL, **payload)
        log.info("job %s esperando aprobación humana", job_id)
    elif result.get("human_decision") == "rejected":
        await store.update(job_id, status=JobStatus.REJECTED, decision="rejected",
                           comment=result.get("human_comment", ""), **payload)
    elif result.get("task_completed") and result.get("human_decision") == "approved":
        await store.update(job_id, status=JobStatus.DONE, decision="approved",
                           comment=result.get("human_comment", ""), **payload)
    else:
        reason = result.get("validation") or "El grafo terminó sin un resultado válido (sin candidatos o validación fallida)."
        await store.update(job_id, status=JobStatus.FAILED, error=reason, **payload)
    log.info("job %s -> %s", job_id, (await store.get(job_id) or {}).get("status"))


async def handle_message(graph: CompiledStateGraph, store: JobStore, message: QueueMessage) -> None:
    job_id = message.job_id
    try:
        if isinstance(message, RunMessage):
            await store.update(job_id, status=JobStatus.RUNNING)
            result = await graph.ainvoke(
                initial_state(message.job_data.model_dump()), config=run_config(job_id, "run")
            )
        else:
            assert isinstance(message, ResumeMessage)
            result = await graph.ainvoke(
                Command(resume={"approved": message.approved, "comment": message.comment}),
                config=run_config(job_id, "resume"),
            )
        await _record_outcome(store, job_id, result)
    except Exception as exc:  # noqa: BLE001 - un job fallido no debe tumbar al worker
        log.exception("job %s falló", job_id)
        await store.update(job_id, status=JobStatus.FAILED, error=f"{type(exc).__name__}: {exc}")


async def run_worker(stop: asyncio.Event | None = None) -> None:
    settings = get_settings()
    stop = stop or asyncio.Event()
    store = JobStore.from_url(settings.redis_url, settings.queue_name)
    slots = asyncio.Semaphore(settings.worker_concurrency)
    running: set[asyncio.Task] = set()

    async with open_graph(settings.redis_url) as graph:
        log.info("worker listo (concurrencia=%d, supervisor=%s)", settings.worker_concurrency, settings.supervisor_mode)
        while not stop.is_set():
            await slots.acquire()  # no se saca trabajo de la cola si no hay capacidad
            message = await store.pop(timeout=2)
            if message is None:
                slots.release()
                continue
            task = asyncio.create_task(handle_message(graph, store, message))
            running.add(task)
            task.add_done_callback(lambda t: (running.discard(t), slots.release()))

        log.info("apagando: esperando %d job(s) en curso", len(running))
        await asyncio.gather(*running, return_exceptions=True)

    await store.aclose()
    await close_retriever()


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    configure_tracing()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:  # Windows
            pass
    await run_worker(stop)


if __name__ == "__main__":
    asyncio.run(main())
