from langgraph.types import interrupt


def human_approval_node(state):
    """
    Pausa el flujo y solicita aprobación humana.
    El grafo queda suspendido hasta recibir un resume.
    """

    print("\n⏸️ HUMAN APPROVAL → esperando aprobación...")

    approval = interrupt({
        "message": "El análisis fue validado. ¿Aprobás el resultado?",
        "analysis": state["analysis_output"],
        "validation": state["validation"],
    })

    print(f"\n👤 HUMAN APPROVAL → respuesta recibida: {approval}")

    return {
        "human_approved": bool(approval)
    }