import pytest
from pydantic import ValidationError

from app.agents.tools import calculate_candidate_score, score_candidate, search_candidates
from app.rag.documents import chunk_cv, load_cvs, parse_front_matter
from app.schemas import Candidate, SearchCandidatesInput
from tests.fakes import CV_DIR, make_retriever


def test_cvs_load_and_validate_against_candidate_schema():
    docs = load_cvs(CV_DIR)
    assert len(docs) == 6
    assert {d.candidate.name for d in docs} >= {"Laura Gómez", "Martín Rodríguez", "Sofía Fernández"}


def test_chunking_is_per_section_and_ids_unique():
    docs = load_cvs(CV_DIR)
    chunks = [c for d in docs for c in chunk_cv(d)]
    assert len(chunks) == 6 * 4
    assert len({c.id for c in chunks}) == len(chunks)
    assert all(c.text.startswith(next(d.candidate.name for d in docs if d.candidate_id == c.candidate_id)) for c in chunks)


def test_front_matter_requires_delimiters():
    with pytest.raises(ValueError):
        parse_front_matter("sin front matter")


def test_scores_match_the_original_rubric():
    by_name = {d.candidate.name: score_candidate(d.candidate).score for d in load_cvs(CV_DIR)}
    assert by_name["Laura Gómez"] == 100.0
    assert by_name["Martín Rodríguez"] == 68.75
    assert by_name["Sofía Fernández"] == 67.5


async def test_score_tool_validates_input_with_pydantic():
    laura = load_cvs(CV_DIR)[3].candidate  # laura-gomez
    out = await calculate_candidate_score.ainvoke({"candidate": laura.model_dump()})
    assert out["score"] == 100.0 and out["breakdown"]["python"] == 20
    with pytest.raises(ValidationError):
        await calculate_candidate_score.ainvoke({"candidate": {"name": "X"}})


async def test_retriever_returns_unique_ranked_candidates():
    found = await make_retriever().search("Data Analyst Python SQL Power BI sector financiero", top_k=3)
    names = [c.name for c in found]
    assert len(names) == len(set(names)) == 3
    assert all(isinstance(c, Candidate) for c in found)


async def test_retriever_metadata_filter_on_experience():
    found = await make_retriever().search("Data Analyst Python SQL", top_k=6, min_experience_years=4)
    assert found and all(c.experience_years >= 4 for c in found)


async def test_search_tool_uses_injected_retriever_and_validates_args(fake_llm):
    result = await search_candidates.ainvoke({"query": "Data Analyst Python SQL", "top_k": 2})
    assert len(result) == 2 and {"name", "python", "sql"} <= result[0].keys()
    with pytest.raises(ValidationError):
        await search_candidates.ainvoke({"query": "ab", "top_k": 99})


def test_search_input_schema_bounds():
    with pytest.raises(ValidationError):
        SearchCandidatesInput(query="ok query", top_k=0)
