"""Cliente async de Pinecone (SDK oficial `PineconeAsyncio`)."""

import asyncio
from collections.abc import Mapping
from typing import Any

from pinecone import PineconeAsyncio, ServerlessSpec

from app.config import Settings
from app.rag.retriever import RetrievedChunk


class PineconeVectorStore:
    def __init__(self, settings: Settings) -> None:
        settings.require("pinecone_api_key", "index_name")
        self._settings = settings
        self._pc: PineconeAsyncio | None = None
        self._index: Any = None
        self._lock = asyncio.Lock()

    async def _client(self) -> PineconeAsyncio:
        if self._pc is None:
            self._pc = PineconeAsyncio(api_key=self._settings.pinecone_api_key)
        return self._pc

    async def ensure_index(self) -> None:
        """Crea el índice serverless si no existe y espera a que esté listo (idempotente)."""
        s, pc = self._settings, await self._client()
        if not await pc.has_index(s.index_name):
            await pc.create_index(
                name=s.index_name,
                dimension=s.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=s.pinecone_cloud, region=s.pinecone_region),
            )
        for _ in range(60):
            if (await pc.describe_index(s.index_name)).status.ready:
                return
            await asyncio.sleep(2)
        raise TimeoutError(f"El índice {s.index_name} no estuvo listo a tiempo.")

    async def _idx(self) -> Any:
        async with self._lock:
            if self._index is None:
                self._index = await (await self._client()).index(name=self._settings.index_name)
            return self._index

    async def upsert(self, records: list[dict[str, Any]]) -> None:
        await (await self._idx()).upsert(vectors=records, namespace=self._settings.pinecone_namespace)

    async def query(
        self, vector: list[float], top_k: int, filter: Mapping[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        response = await (await self._idx()).query(
            vector=vector,
            top_k=top_k,
            namespace=self._settings.pinecone_namespace,
            filter=filter,
            include_metadata=True,
        )
        chunks = []
        for match in response.matches:
            metadata = dict(match.metadata or {})
            chunks.append(
                RetrievedChunk(id=match.id, score=match.score or 0.0, text=metadata.pop("text", ""), metadata=metadata)
            )
        return chunks

    async def aclose(self) -> None:
        if self._index is not None:
            await self._index.close()
            self._index = None
        if self._pc is not None:
            await self._pc.close()
            self._pc = None
