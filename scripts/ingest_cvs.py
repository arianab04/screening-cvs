"""Indexa los CVs de data/cvs en Pinecone (idempotente: los ids son determinísticos).

Uso: python -m scripts.ingest_cvs
"""

import asyncio
import logging
from pathlib import Path

from app.config import get_settings
from app.rag.documents import chunk_cv, load_cvs, to_records
from app.rag.service import get_embeddings
from app.rag.vectorstore import PineconeVectorStore

CV_DIR = Path(__file__).resolve().parent.parent / "data" / "cvs"
log = logging.getLogger("ingest")


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    settings.require("openai_api_key", "pinecone_api_key", "index_name")

    pairs = [(doc, chunk) for doc in load_cvs(CV_DIR) for chunk in chunk_cv(doc)]
    log.info("%d chunks de %d CVs", len(pairs), len({d.candidate_id for d, _ in pairs}))

    vectors = await get_embeddings().aembed_documents([chunk.text for _, chunk in pairs])
    records = to_records(pairs, vectors)

    store = PineconeVectorStore(settings)
    try:
        await store.ensure_index()
        await store.upsert(records)
    finally:
        await store.aclose()
    log.info("Indexados %d vectores en '%s' (namespace '%s')", len(records), settings.index_name, settings.pinecone_namespace)


if __name__ == "__main__":
    asyncio.run(main())
