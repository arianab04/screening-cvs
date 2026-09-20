"""5 pruebas end-to-end contra el sistema levantado (`./run.sh`). Cada una deja un trace en LangSmith.

Uso: python -m scripts.e2e_tests [--base-url http://localhost:8000] [--no-restart]

Genera docs/evidence/e2e_report.md con los job_id de cada prueba para ubicarlos en el dashboard
(filtrar por metadata `job_id` o por el tag `screening-cvs`).
"""

import argparse
import asyncio
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

REPORT = Path(__file__).resolve().parent.parent / "docs" / "evidence" / "e2e_report.md"

VACANTE = {
    "role": "Data Analyst",
    "experience": "2+ años",
    "python": "intermedio/avanzado",
    "sql": "intermedio/avanzado",
    "bi": "Power BI o equivalente",
    "statistics": "conocimientos aplicados",
    "financial_sector": "deseable",
}

results: list[tuple[str, bool, str]] = []


async def wait_for(client: httpx.AsyncClient, job_id: str, wanted: set[str], timeout: float = 180) -> dict:
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        job = (await client.get(f"/tasks/{job_id}")).json()
        if job["status"] in wanted or job["status"] == "FAILED":
            return job
        if asyncio.get_running_loop().time() > deadline:
            raise TimeoutError(f"job {job_id} no llegó a {wanted}: {job['status']}")
        await asyncio.sleep(1)


async def create(client: httpx.AsyncClient, payload: dict = VACANTE) -> str:
    r = await client.post("/tasks", json=payload)
    assert r.status_code == 202, r.text
    return r.json()["job_id"]


async def test_happy_path(client):
    job_id = await create(client)
    job = await wait_for(client, job_id, {"WAITING_APPROVAL"})
    assert job["status"] == "WAITING_APPROVAL", job
    top = max(job["scores"], key=lambda s: s["score"])["name"]
    assert top in job["result"]
    assert (await client.post(f"/tasks/{job_id}/approve")).status_code == 202
    job = await wait_for(client, job_id, {"DONE"})
    assert job["status"] == "DONE" and job["decision"] == "approved", job
    return job_id, f"DONE; recomendado: {top}"


async def test_rejection(client):
    job_id = await create(client)
    await wait_for(client, job_id, {"WAITING_APPROVAL"})
    r = await client.post(f"/tasks/{job_id}/reject", json={"comment": "Necesito más candidatos"})
    assert r.status_code == 202, r.text
    job = await wait_for(client, job_id, {"REJECTED"})
    assert job["status"] == "REJECTED" and job["comment"] == "Necesito más candidatos", job
    return job_id, "REJECTED con comentario"


async def test_concurrency(client):
    ids = await asyncio.gather(*(create(client) for _ in range(5)))
    jobs = await asyncio.gather(*(wait_for(client, i, {"WAITING_APPROVAL"}) for i in ids))
    assert all(j["status"] == "WAITING_APPROVAL" for j in jobs), [j["status"] for j in jobs]
    await asyncio.gather(*(client.post(f"/tasks/{i}/approve") for i in ids))
    jobs = await asyncio.gather(*(wait_for(client, i, {"DONE"}) for i in ids))
    assert all(j["status"] == "DONE" for j in jobs)
    return ", ".join(ids), "5 jobs simultáneos -> DONE"


async def test_validation_errors(client):
    bad = await client.post("/tasks", json={"role": "", "extra_field": 1})
    assert bad.status_code == 422, bad.text
    assert (await client.get("/tasks/no-existe")).status_code == 404
    job_id = await create(client)
    conflict = await client.post(f"/tasks/{job_id}/approve")  # todavía no está en WAITING_APPROVAL
    assert conflict.status_code == 409, conflict.text
    await wait_for(client, job_id, {"WAITING_APPROVAL"})
    return job_id, "422 (payload inválido), 404 (job inexistente), 409 (aprobar antes de tiempo)"


async def test_persistence_after_restart(client, restart: bool):
    job_id = await create(client)
    await wait_for(client, job_id, {"WAITING_APPROVAL"})
    if restart:
        subprocess.run(["docker", "compose", "restart", "worker"], check=True)
        await asyncio.sleep(5)
    assert (await client.post(f"/tasks/{job_id}/approve")).status_code == 202
    job = await wait_for(client, job_id, {"DONE"})
    assert job["status"] == "DONE", job
    if restart:
        return job_id, "worker reiniciado con el grafo pausado; el checkpoint en Redis permitió reanudar -> DONE"
    return job_id, "DONE (sin reinicio del worker: --no-restart)"


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--no-restart", action="store_true", help="omite el reinicio del worker (prueba 5)")
    args = parser.parse_args()

    async with httpx.AsyncClient(base_url=args.base_url, timeout=30) as client:
        health = (await client.get("/health")).json()
        assert health["status"] == "ok", f"API degradada: {health}"
        cases = [
            ("1. Camino feliz (RAG + agentes + aprobación)", test_happy_path(client)),
            ("2. Rechazo humano", test_rejection(client)),
            ("3. Concurrencia (5 jobs)", test_concurrency(client)),
            ("4. Validación Pydantic y errores HTTP", test_validation_errors(client)),
            ("5. Persistencia: reinicio del worker (Checkpointer)", test_persistence_after_restart(client, not args.no_restart)),
        ]
        for name, coro in cases:
            try:
                job_ids, detail = await coro
                results.append((name, True, f"{detail} — job_id: `{job_ids}`"))
            except Exception as exc:  # noqa: BLE001
                results.append((name, False, f"{type(exc).__name__}: {exc}"))
            print(("OK   " if results[-1][1] else "FAIL ") + name + " — " + results[-1][2])

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Reporte de pruebas E2E\n\nEjecutado: {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC\n"]
    lines += [f"- {'✅' if ok else '❌'} **{name}** — {detail}" for name, ok, detail in results]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReporte: {REPORT}")
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
