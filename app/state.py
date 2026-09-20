from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """Estado compartido del grafo. Todo se serializa en el checkpointer (Redis)."""

    job_requirements: dict
    candidates: list
    research_output: str
    research_attempts: int
    scores: list
    analysis_output: str
    analysis_attempts: int
    validation: str
    task_completed: bool
    next_agent: str
    supervisor_reason: str
    human_decision: str  # "", "approved" o "rejected"
    human_comment: str


def initial_state(job_requirements: dict) -> AgentState:
    return {
        "messages": [],
        "job_requirements": job_requirements,
        "candidates": [],
        "research_output": "",
        "research_attempts": 0,
        "scores": [],
        "analysis_output": "",
        "analysis_attempts": 0,
        "validation": "",
        "task_completed": False,
        "next_agent": "",
        "supervisor_reason": "",
        "human_decision": "",
        "human_comment": "",
    }
