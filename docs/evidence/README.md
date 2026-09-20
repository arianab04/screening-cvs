# Evidencia de observabilidad

Trazas de LangSmith (proyecto `screening-cvs`) de las 5 pruebas end-to-end (`python -m scripts.e2e_tests`).
Los `job_id` de cada prueba están en [`e2e_report.md`](e2e_report.md); en LangSmith cada ejecución lleva la metadata
`job_id` y las fases `run` / `resume` (dos traces por job: la ejecución hasta la pausa y la reanudación tras la decisión humana).

| Prueba | Captura | Qué muestra |
|---|---|---|
| 1. Camino feliz | [`trace-1-happy-path.png`](trace-1-happy-path.png) | Árbol de spans: `supervisor` (LLM con salida `SupervisorDecision`) → `research` → tool `search_candidates` (RAG) → `analyst` → `calculate_candidate_score` |
| | [`trace-1c-analyst-scoring.png`](trace-1c-analyst-scoring.png) | Scores por candidato y llamada del LLM del analyst con tokens y costo |
| | [`trace-1d-validation-approval.png`](trace-1d-validation-approval.png) | `validation` → `supervisor` → `approval` (pausa human-in-the-loop) |
| | [`trace-1b-approval-resume.png`](trace-1b-approval-resume.png) | Fase `resume`: `human_decision: approved` → fin |
| 2. Rechazo humano | [`trace-2-rejection.png`](trace-2-rejection.png) | Fase `resume` con `human_decision: rejected` y el comentario del revisor |
| 3. Concurrencia | [`trace-3-concurrency.png`](trace-3-concurrency.png) | Listado de traces con varios `run` de ~12 s superpuestos (5 jobs simultáneos) |
| 4. Validación y errores HTTP | [`trace-4-validation.png`](trace-4-validation.png) | Reporte de la corrida: 422 (payload inválido), 404 (job inexistente), 409 (aprobar antes de tiempo). Estos rechazos ocurren en la API, antes del grafo, por eso no generan trace |
| 5. Persistencia | [`trace-5-persistence.png`](trace-5-persistence.png) | Mismo hilo con `run` y `resume`: el worker se reinició con el grafo pausado y el checkpoint de Redis permitió reanudar |

Cada span de LLM muestra sus tokens y el resumen del trace el costo total (≈ USD 0,0008 por job).

## Material previo

`pre-entrega-m7/` contiene capturas de la pre-entrega del Módulo 7 (versión anterior del sistema, con candidatos fijos en
el código y ejecución síncrona). Se conservan solo como historial; **no** corresponden a la arquitectura actual.
