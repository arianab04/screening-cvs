from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.config import get_settings

_llm: BaseChatModel | None = None


def get_llm() -> BaseChatModel:
    """LLM de chat compartido (modelo configurable por CHAT_MODEL)."""
    global _llm
    if _llm is None:
        settings = get_settings()
        settings.require("openai_api_key")
        _llm = ChatOpenAI(model=settings.chat_model, temperature=0, api_key=settings.openai_api_key)
    return _llm


def set_llm(llm: BaseChatModel | None) -> None:
    """Permite inyectar un LLM alternativo (tests)."""
    global _llm
    _llm = llm
