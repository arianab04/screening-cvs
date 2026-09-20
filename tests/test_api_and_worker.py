import json

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.graph import build_graph
from app.main import app
from app.schemas import JobStatus
from app.worker import handle_message
from tests.fakes import VACANTE, FakeJobStore


@pytest.fixture
def store():
    store = FakeJobStore()
    app.state.store = store
    return store


@pytest.fixture
async def client(store):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _drain(graph, store):
    while (message := await store.pop()) is not None:
        await handle_message(graph, store, message)


async def test_health(client):
    r = await client.get("/health")
    assert r.json() == {"status": "ok", "redis": True}


async def test_create_task_is_validated_and_enqueued(client, store):
    r = await client.post("/tasks", json=VACANTE.model_dump())
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    assert r.json()["status"] == "PENDING"
    assert store.queue[0].job_id == job_id


@pytest.mark.parametrize("payload", [{}, {"role": ""}, {"role": "Data Analyst", "otro": 1}, {"role": 5}])
async def test_invalid_payload_is_rejected_with_422(client, payload):
    assert (await client.post("/tasks", json=payload)).status_code == 422


async def test_unknown_job_is_404(client):
    assert (await client.get("/tasks/nope")).status_code == 404
    assert (await client.post("/tasks/nope/approve")).status_code == 404


async def test_approve_before_waiting_is_409(client):
    job_id = (await client.post("/tasks", json=VACANTE.model_dump())).json()["job_id"]
    assert (await client.post(f"/tasks/{job_id}/approve")).status_code == 409


async def test_end_to_end_run_pause_approve_done(client, store, fake_llm):
    graph = build_graph(InMemorySaver())
    job_id = (await client.post("/tasks", json=VACANTE.model_dump())).json()["job_id"]

    await _drain(graph, store)  # el worker ejecuta hasta el interrupt
    job = (await client.get(f"/tasks/{job_id}")).json()
    assert job["status"] == "WAITING_APPROVAL"
    assert "Candidato recomendado" in job["result"] and len(job["scores"]) == 5

    r = await client.post(f"/tasks/{job_id}/approve", json={"comment": "aprobado"})
    assert r.status_code == 202
    assert (await client.post(f"/tasks/{job_id}/approve")).status_code == 409  # doble aprobación

    await _drain(graph, store)  # el worker reanuda desde el checkpoint
    job = (await client.get(f"/tasks/{job_id}")).json()
    assert job["status"] == "DONE" and job["decision"] == "approved" and job["comment"] == "aprobado"


async def test_reject_flow(client, store, fake_llm):
    graph = build_graph(InMemorySaver())
    job_id = (await client.post("/tasks", json=VACANTE.model_dump())).json()["job_id"]
    await _drain(graph, store)
    await client.post(f"/tasks/{job_id}/reject", json={"comment": "no cumple"})
    await _drain(graph, store)
    job = (await client.get(f"/tasks/{job_id}")).json()
    assert job["status"] == "REJECTED" and job["comment"] == "no cumple"


async def test_failed_validation_marks_job_failed_with_reason(client, store, fake_llm):
    fake_llm.analysis_builder = lambda prompt: "sin secciones"
    graph = build_graph(InMemorySaver())
    job_id = (await client.post("/tasks", json=VACANTE.model_dump())).json()["job_id"]
    await _drain(graph, store)
    job = (await client.get(f"/tasks/{job_id}")).json()
    assert job["status"] == "FAILED" and "Candidato recomendado" in job["error"]


async def test_worker_survives_a_crashing_job(store, fake_llm):
    class BoomGraph:
        async def ainvoke(self, *a, **k):
            raise RuntimeError("boom")

    await store.create_job("j", VACANTE)
    await handle_message(BoomGraph(), store, await store.pop())
    job = await store.get("j")
    assert job["status"] == JobStatus.FAILED and "boom" in job["error"]
    json.dumps(job)  # el estado sigue siendo serializable
