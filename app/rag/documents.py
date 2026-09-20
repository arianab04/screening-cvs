"""Carga de CVs (Markdown con front matter) y chunking por sección."""

from dataclasses import dataclass
from pathlib import Path

from app.schemas import Candidate


@dataclass(frozen=True)
class CVDocument:
    candidate_id: str
    candidate: Candidate
    body: str


@dataclass(frozen=True)
class CVChunk:
    id: str
    candidate_id: str
    section: str
    text: str


def _coerce(value: str) -> str | int | bool:
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if value.isdigit():
        return int(value)
    return value


def parse_front_matter(text: str) -> tuple[dict, str]:
    """Separa el bloque `---` inicial (pares clave: valor) del cuerpo del documento."""
    lines = text.strip().splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("El CV debe comenzar con un bloque de front matter '---'.")
    try:
        end = lines[1:].index("---") + 1
    except ValueError as exc:
        raise ValueError("Front matter sin cierre '---'.") from exc
    meta: dict = {}
    for line in lines[1:end]:
        if not line.strip():
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = _coerce(value.strip())
    return meta, "\n".join(lines[end + 1 :]).strip()


def load_cvs(directory: Path) -> list[CVDocument]:
    docs = []
    for path in sorted(directory.glob("*.md")):
        meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
        docs.append(CVDocument(candidate_id=path.stem, candidate=Candidate.model_validate(meta), body=body))
    if not docs:
        raise FileNotFoundError(f"No hay CVs (*.md) en {directory}")
    return docs


def to_records(pairs: list[tuple[CVDocument, CVChunk]], vectors: list[list[float]]) -> list[dict]:
    """Registros de upsert: el perfil estructurado viaja como metadatos junto a cada chunk."""
    return [
        {
            "id": chunk.id,
            "values": vector,
            "metadata": {
                **doc.candidate.model_dump(),
                "candidate_id": chunk.candidate_id,
                "section": chunk.section,
                "text": chunk.text,
            },
        }
        for (doc, chunk), vector in zip(pairs, vectors, strict=True)
    ]


def chunk_cv(doc: CVDocument) -> list[CVChunk]:
    """Un chunk por sección `## ...`. Cada chunk lleva el nombre del candidato para no perder contexto."""
    chunks: list[CVChunk] = []
    section, buffer = "Resumen", []

    def flush() -> None:
        content = "\n".join(buffer).strip()
        if content:
            chunks.append(
                CVChunk(
                    id=f"{doc.candidate_id}#{len(chunks)}",
                    candidate_id=doc.candidate_id,
                    section=section,
                    text=f"{doc.candidate.name} — {section}\n{content}",
                )
            )

    for line in doc.body.splitlines():
        if line.startswith("## "):
            flush()
            section, buffer = line[3:].strip(), []
        else:
            buffer.append(line)
    flush()
    return chunks
