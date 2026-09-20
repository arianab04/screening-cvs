import pytest

from app.config import get_settings
from app.llm import set_llm
from app.rag.service import set_retriever
from tests.fakes import FakeLLM, make_retriever


@pytest.fixture
def settings_env(monkeypatch):
    """Permite cambiar variables de entorno y recargar Settings."""

    def apply(**env: str):
        for key, value in env.items():
            monkeypatch.setenv(key.upper(), value)
        get_settings.cache_clear()
        return get_settings()

    monkeypatch.setenv("SUPERVISOR_MODE", "rules")
    get_settings.cache_clear()
    yield apply
    get_settings.cache_clear()


@pytest.fixture
def fake_llm(settings_env):
    llm = FakeLLM()
    set_llm(llm)
    set_retriever(make_retriever())
    yield llm
    set_llm(None)
    set_retriever(None)
