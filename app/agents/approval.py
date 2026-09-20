"""Human-in-the-loop: pausa el grafo (checkpoint en Redis) hasta recibir la decisión humana."""

from langgraph.types import interrupt

from app.state import AgentState


async def human_approval_node(state: AgentState) -> dict:
    decision = interrupt(
        {
            "message": "El análisis fue validado. ¿Aprobás el resultado?",
            "analysis": state["analysis_output"],
            "validation": state["validation"],
        }
    )
    approved = bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)
    comment = decision.get("comment", "") if isinstance(decision, dict) else ""
    return {"human_decision": "approved" if approved else "rejected", "human_comment": comment}
