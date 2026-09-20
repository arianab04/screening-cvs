#!/usr/bin/env bash
# Levanta todo el sistema (Redis + ingesta de CVs + API + worker) con una sola instrucción.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Se creó .env a partir de .env.example."
  echo "Completá OPENAI_API_KEY, PINECONE_API_KEY (y LANGSMITH_API_KEY) y volvé a ejecutar ./run.sh"
  exit 1
fi

docker compose up --build "$@"
