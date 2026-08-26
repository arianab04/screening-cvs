# Orquestador Multi-Agente de RRHH

Prototipo funcional de un sistema multi-agente para asistir en el
screening de candidatos para una posición de Data Analyst.

El sistema utiliza LangGraph para coordinar un Supervisor y dos
agentes especialistas:

- Research Agent: búsqueda de candidatos.
- Analyst Agent: cálculo y análisis de candidatos.
- Validation: verificación del resultado antes de finalizar.

---

## Objetivo

Construir un orquestador multi-agente capaz de recibir una solicitud
de selección de personal, delegar la tarea entre distintos agentes
especializados y generar una recomendación final validada.

El caso de prueba utilizado es la selección del candidato más adecuado
para una posición de Data Analyst.

---

## Arquitectura

La arquitectura utiliza una topología jerárquica.

El Supervisor funciona como router central y decide qué nodo debe
ejecutarse en cada etapa del proceso.

```mermaid
flowchart TD

    START --> Supervisor

    Supervisor -->|research| Research
    Supervisor -->|analyst| Analyst
    Supervisor -->|validation| Validation
    Supervisor -->|end| END

    Research --> Supervisor
    Analyst --> Supervisor
    Validation --> Supervisor