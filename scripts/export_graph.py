"""Imprime el diagrama Mermaid del grafo real (así el README nunca se desincroniza del código).

Uso: python -m scripts.export_graph
"""

from app.graph import build_graph

if __name__ == "__main__":
    print(build_graph().get_graph().draw_mermaid())
