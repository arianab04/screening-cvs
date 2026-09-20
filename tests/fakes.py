"""Dobles de prueba: sin red, sin API keys."""

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage

from app.rag.documents import chunk_cv, load_cvs, to_records
from app.rag.retriever import CandidateRetriever, RetrievedChunk, tokenize
from app.schemas import JobStatus, QueueMessage, TaskRequest

CV_DIR = Path(__file__).resolve().parent.parent / "data" / "cvs"
DIM = 64


def fake_embed(text: str) -> list[float]:
    """Bag-of-words hasheado y normalizado: suficiente para que la similitud coseno tenga sentido."""
    vec = [0.0] * DIM
    for token in tokenize(text):
        vec[sum(map(ord, token)) % DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


async def fake_embed_query(text: str) -> list[float]:
    return fake_embed(text)


class InMemoryVectorStore:
    """Vector store en memoria con filtro `$gte` sobre metadatos (como Pinecone)."""

    def __init__(self) -> None:
        pairs = [(d, c) for d in load_cvs(CV_DIR) for c in chunk_cv(d)]
        self.records = to_records(pairs, [fake_embed(c.text) for _, c in pairs])

    async def query(self, vector: list[float], top_k: int, filter: Mapping[str, Any] | None = None):
        rows = []
        for r in self.records:
            meta = r["metadata"]
            if filter and any(meta[k] < cond["$gte"] for k, cond in filter.items()):
                continue
            rows.append(RetrievedChunk(r["id"], sum(a * b for a, b in zip(vector, r["values"])), meta["text"], dict(meta)))
        return sorted(rows, key=lambda c: c.score, reverse=True)[:top_k]


def make_retriever() -> CandidateRetriever:
    return CandidateRetriever(fake_embed_query, InMemoryVectorStore())


class FakeLLM:
    """Imita al ChatOpenAI: llamada a tool en el research, informe en el analyst, decisión en el supervisor."""

    def __init__(self, analysis_builder=None, supervisor_choice: str | None = None, supervisor_error=False) -> None:
        self.analysis_builder = analysis_builder or self.good_analysis
        self.supervisor_choice = supervisor_choice
        self.supervisor_error = supervisor_error
        self.calls: list[str] = []

    @staticmethod
    def good_analysis(prompt: str) -> str:
        # El prompt contiene los scores ordenados: el primer 'name' tras SCORES es el mejor.
        scores_part = prompt.split("SCORES (ordenados de mayor a menor):", 1)[1]
        top = scores_part.split("'name': '", 1)[1].split("'", 1)[0]
        names = [chunk.split("'", 1)[0] for chunk in scores_part.split("'name': '")[1:]]
        return "### 3. Candidato recomendado\n" + top + "\n### 7. Comparación\n" + ", ".join(names)

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema):
        llm = self

        class _Structured:
            async def ainvoke(self, prompt):
                llm.calls.append("supervisor")
                if llm.supervisor_error:
                    raise RuntimeError("LLM caído")
                return schema(next_agent=llm.supervisor_choice, reason="fake")

        return _Structured()

    async def ainvoke(self, prompt: str):
        if "agente de investigación" in prompt:
            self.calls.append("research")
            return AIMessage(
                content="",
                tool_calls=[{"name": "search_candidates", "args": {"query": "Data Analyst Python SQL Power BI"}, "id": "1"}],
            )
        self.calls.append("analyst")
        return AIMessage(content=self.analysis_builder(prompt))


VACANTE = TaskRequest(
    role="Data Analyst", experience="2+ años", python="intermedio/avanzado", sql="intermedio/avanzado",
    bi="Power BI o equivalente", statistics="conocimientos aplicados", financial_sector="deseable",
)


class FakeJobStore:
    """Misma interfaz que JobStore, en memoria."""

    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, str]] = {}
        self.queue: list[QueueMessage] = []

    async def ping(self) -> bool:
        return True

    async def create_job(self, job_id: str, task: TaskRequest) -> None:
        from app.schemas import RunMessage

        self.jobs[job_id] = {"status": JobStatus.PENDING.value, "task": task.model_dump_json()}
        self.queue.append(RunMessage(job_id=job_id, job_data=task))

    async def get(self, job_id: str):
        return dict(self.jobs[job_id]) if job_id in self.jobs else None

    async def update(self, job_id: str, **fields: Any) -> None:
        import json

        for k, v in fields.items():
            if v is not None:
                self.jobs[job_id][k] = v if isinstance(v, str) else json.dumps(v)

    async def transition(self, job_id: str, expected: JobStatus, new: JobStatus) -> bool:
        if self.jobs[job_id]["status"] == expected.value:
            self.jobs[job_id]["status"] = new.value
            return True
        return False

    async def enqueue(self, message: QueueMessage) -> None:
        self.queue.append(message)

    async def pop(self, timeout: int = 5):
        return self.queue.pop(0) if self.queue else None
