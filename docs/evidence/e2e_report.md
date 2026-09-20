# Reporte de pruebas E2E

Ejecutado: 2026-09-20 14:35 UTC

- ✅ **1. Camino feliz (RAG + agentes + aprobación)** — DONE; recomendado: Laura Gómez — job_id: `b1adef35-2ced-475d-b477-f63ac91659cf`
- ✅ **2. Rechazo humano** — REJECTED con comentario — job_id: `257d690e-c351-4256-897b-c83595003dde`
- ✅ **3. Concurrencia (5 jobs)** — 5 jobs simultáneos -> DONE — job_id: `589e2c8d-eec6-4e5d-8a10-cda0934b4ad2, 97ab5eed-3cab-4ef0-b050-4e3977a2dd9c, d150e250-c807-4129-a3c7-a47f34e9d88b, fe96c4f8-b86d-43c2-b0dc-f04095f5d79a, 7cf57a6d-dbb1-4bcc-8e83-d9aefe311333`
- ✅ **4. Validación Pydantic y errores HTTP** — 422 (payload inválido), 404 (job inexistente), 409 (aprobar antes de tiempo) — job_id: `7bf9962f-5983-4558-b32d-d7b63f70fc87`
- ✅ **5. Persistencia: reinicio del worker (Checkpointer)** — worker reiniciado con el grafo pausado; el checkpoint en Redis permitió reanudar -> DONE — job_id: `8d0f58c0-b0ac-42ca-98ad-78c5b7897c04`
