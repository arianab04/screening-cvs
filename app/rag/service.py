"""Punto único de acceso al retriever (singleton lazy, inyectable en tests)."""

from langchain_openai import OpenAIEmbeddings

from app.config import get_settings
from app.rag.retriever import CandidateRetriever
from app.rag.vectorstore import PineconeVectorStore

_retriever: CandidateRetriever | None = None
_store: PineconeVectorStore | None = None


def get_embeddings() -> OpenAIEmbeddings:
    settings = get_settings()
    settings.require("openai_api_key")
    return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)


def get_retriever() -> CandidateRetriever:
    global _retriever, _store
    if _retriever is None:
        _store = PineconeVectorStore(get_settings())
        _retriever = CandidateRetriever(get_embeddings().aembed_query, _store)
    return _retriever


def set_retriever(retriever: CandidateRetriever | None) -> None:
    global _retriever
    _retriever = retriever


async def close_retriever() -> None:
    global _retriever, _store
    if _store is not None:
        await _store.aclose()
    _retriever = _store = None
