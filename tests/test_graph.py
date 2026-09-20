from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.agents.supervisor import legal_next, supervisor_node
from app.agents.validation import check_analysis
from app.config import get_settings
from app.graph import build_graph
from app.llm import set_llm
from app.observability import run_config
from app.schemas import CandidateScore, ScoreBreakdown
from app.state import initial_state
from tests.fakes import VACANTE, FakeLLM


def _cfg(job_id: str):
    return run_config(job_id, "test")


def _score(name: str, score: float) -> CandidateScore:
    return CandidateScore(name=name, score=score, breakdown=ScoreBreakdown(
        experience=0, python=0, sql=0, bi=0, statistics=0, education=0, financial_sector=0))


# ---------------------------------------------------------------- validation


def test_validation_accepts_consistent_report_and_rejects_wrong_recommendation():
    scores = [_score("Ana", 90), _score("Beto", 50)]
    good = "### 3. Candidato recomendado\nAna\n### 7. Otros\nBeto"
    assert check_analysis(good, scores) is None
    wrong = "### 3. Candidato recomendado\nBeto\n### 7. Otros\nAna"
    assert "debe ser Ana" in check_analysis(wrong, scores)
    assert "no menciona" in check_analysis("### Candidato recomendado\nAna", scores)
    assert check_analysis("", scores) == "El análisis está vacío."
    assert check_analysis("x", []) == "No se encontraron candidatos."


# ---------------------------------------------------------------- supervisor


def test_legal_transitions_follow_the_state_machine(settings_env):
    s = get_settings()
    st = initial_state(VACANTE.model_dump())
    assert legal_next(st, s) == ["research"]
    st["candidates"] = [{"x": 1}]
    assert legal_next(st, s) == ["analyst"]
    st["analysis_output"] = "a"
    assert legal_next(st, s) == ["validation"]
    st.update(validation="fallo", task_completed=False, analysis_attempts=1)
    assert legal_next(st, s) == ["analyst", "end"]
    st["analysis_attempts"] = s.max_analysis_retries + 1  # reintentos agotados
    assert legal_next(st, s) == ["end"]
    st.update(task_completed=True)
    assert legal_next(st, s) == ["approval"]
    st.update(human_decision="approved")
    assert legal_next(st, s) == ["end"]


async def test_llm_supervisor_guardrail_overrides_illegal_choice(settings_env):
    settings_env(supervisor_mode="llm")
    set_llm(FakeLLM(supervisor_choice="approval"))  # ilegal: todavía no hay candidatos
    try:
        out = await supervisor_node(initial_state(VACANTE.model_dump()))
    finally:
        set_llm(None)
    assert out["next_agent"] == "research" and "guardrail" in out["supervisor_reason"]


async def test_llm_supervisor_falls_back_to_rules_on_error(settings_env):
    settings_env(supervisor_mode="llm")
    set_llm(FakeLLM(supervisor_error=True))
    try:
        out = await supervisor_node(initial_state(VACANTE.model_dump()))
    finally:
        set_llm(None)
    assert out["next_agent"] == "research" and "fallback" in out["supervisor_reason"]


async def test_llm_supervisor_accepts_legal_choice(settings_env):
    settings_env(supervisor_mode="llm")
    set_llm(FakeLLM(supervisor_choice="research"))
    try:
        out = await supervisor_node(initial_state(VACANTE.model_dump()))
    finally:
        set_llm(None)
    assert out == {"next_agent": "research", "supervisor_reason": "fake"}


# ---------------------------------------------------------------- grafo completo


async def test_full_flow_pauses_for_approval_then_completes(fake_llm):
    graph = build_graph(InMemorySaver())
    paused = await graph.ainvoke(initial_state(VACANTE.model_dump()), _cfg("j1"))

    assert paused["__interrupt__"], "el grafo debe pausarse en human approval"
    assert paused["task_completed"] is True and len(paused["candidates"]) == 5
    ranking = [s["score"] for s in paused["scores"]]
    assert ranking == sorted(ranking, reverse=True)

    done = await graph.ainvoke(Command(resume={"approved": True, "comment": "ok"}), _cfg("j1"))
    assert done["human_decision"] == "approved" and done["human_comment"] == "ok"
    assert "__interrupt__" not in done


async def test_rejection_ends_the_flow(fake_llm):
    graph = build_graph(InMemorySaver())
    await graph.ainvoke(initial_state(VACANTE.model_dump()), _cfg("j2"))
    done = await graph.ainvoke(Command(resume={"approved": False, "comment": "no"}), _cfg("j2"))
    assert done["human_decision"] == "rejected"


async def test_failing_validation_retries_are_bounded_no_infinite_loop(fake_llm):
    fake_llm.analysis_builder = lambda prompt: "informe sin la sección obligatoria"
    graph = build_graph(InMemorySaver())
    result = await graph.ainvoke(initial_state(VACANTE.model_dump()), _cfg("j3"))

    assert "__interrupt__" not in result
    assert result["task_completed"] is False
    assert result["analysis_attempts"] == get_settings().max_analysis_retries + 1
    assert fake_llm.calls.count("analyst") == result["analysis_attempts"]


async def test_analyst_retry_receives_validation_feedback(fake_llm):
    prompts: list[str] = []
    first = {"done": False}

    def builder(prompt: str) -> str:
        prompts.append(prompt)
        if not first["done"]:
            first["done"] = True
            return "malo"
        return FakeLLM.good_analysis(prompt)

    fake_llm.analysis_builder = builder
    graph = build_graph(InMemorySaver())
    result = await graph.ainvoke(initial_state(VACANTE.model_dump()), _cfg("j4"))
    assert result["task_completed"] is True and result["analysis_attempts"] == 2
    assert "fue rechazado" in prompts[1] and "fue rechazado" not in prompts[0]
