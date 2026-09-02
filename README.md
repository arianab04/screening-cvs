# Orquestador Multi-Agente de RRHH

API de screening automatizado de CVs mediante un orquestador multi-agente construido con LangGraph.

El sistema recibe una tarea de screening, la coloca en una cola Redis y la procesa mediante varios agentes:

- Research Agent: obtiene los candidatos disponibles.
- Analyst Agent: calcula los puntajes de los candidatos.
- Validation: verifica que el análisis contenga la información necesaria.
- Human-in-the-Loop: solicita aprobación humana antes de finalizar.
- LangSmith: registra trazas de la ejecución.
- RedisSaver: persiste el estado del grafo.

## Arquitectura

```mermaid
flowchart TD
    CLIENT[Cliente / Swagger]
    API[FastAPI]
    QUEUE[Redis Queue]
    WORKER[Worker]
    GRAPH[LangGraph]
    SUP[Supervisor]
    RESEARCH[Research Agent]
    ANALYST[Analyst Agent]
    VALIDATION[Validation]
    APPROVAL[Human Approval]
    REDIS[RedisSaver]
    LANGSMITH[LangSmith]

    CLIENT --> API
    API --> QUEUE
    QUEUE --> WORKER
    WORKER --> GRAPH

    GRAPH --> SUP
    SUP --> RESEARCH
    RESEARCH --> SUP
    SUP --> ANALYST
    ANALYST --> SUP
    SUP --> VALIDATION
    VALIDATION --> SUP
    SUP --> APPROVAL
    APPROVAL --> SUP
    SUP --> END

    GRAPH <--> REDIS
    GRAPH --> LANGSMITH