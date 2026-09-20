"""Configuración centralizada. Todo valor que dependa del entorno se lee de variables de entorno."""

from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Exporta el .env a os.environ (sin pisar variables ya definidas) para que
# los SDKs que leen el entorno directamente (LangSmith) lo vean.
load_dotenv()


class ConfigError(RuntimeError):
    """Falta una variable de entorno obligatoria."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    # LLM y embeddings
    openai_api_key: str = ""
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536

    # Pinecone
    pinecone_api_key: str = ""
    index_name: str = ""
    pinecone_namespace: str = "screening-cvs"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # Redis (cola de trabajos, estado de jobs y checkpointer de LangGraph)
    redis_url: str = "redis://localhost:6379"
    queue_name: str = "screening_tasks"
    worker_concurrency: int = Field(default=4, ge=1)

    # Comportamiento del grafo
    supervisor_mode: Literal["llm", "rules"] = "llm"
    max_analysis_retries: int = Field(default=2, ge=0)
    max_research_attempts: int = Field(default=2, ge=1)
    retrieval_top_k: int = Field(default=5, ge=1, le=20)

    # Observabilidad (LangSmith lee LANGSMITH_* del entorno; se declaran para validarlas)
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "screening-cvs"

    def require(self, *names: str) -> None:
        missing = [n.upper() for n in names if not getattr(self, n)]
        if missing:
            raise ConfigError(
                f"Faltan variables de entorno obligatorias: {', '.join(missing)}. "
                "Copiá .env.example a .env y completalas."
            )


@lru_cache
def get_settings() -> Settings:
    return Settings()
