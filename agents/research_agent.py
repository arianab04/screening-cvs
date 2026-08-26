from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()


CANDIDATES = [
    {
        "name": "Laura Gómez",
        "education": "Licenciatura en Economía",
        "experience_years": 4,
        "role": "Data Analyst",
        "python": "avanzado",
        "sql": "avanzado",
        "bi": "Power BI",
        "statistics": "avanzada",
        "financial_sector": True,
        "machine_learning": "básico",
    },
    {
        "name": "Martín Rodríguez",
        "education": "Ingeniería Industrial",
        "experience_years": 3,
        "role": "Data Analyst",
        "python": "intermedio",
        "sql": "avanzado",
        "bi": "Tableau",
        "statistics": "intermedia",
        "financial_sector": False,
        "machine_learning": "intermedio",
    },
    {
        "name": "Sofía Fernández",
        "education": "Licenciatura en Economía",
        "experience_years": 2,
        "role": "Data Analyst",
        "python": "avanzado",
        "sql": "intermedio",
        "bi": "Power BI",
        "statistics": "básica/intermedia",
        "financial_sector": False,
        "machine_learning": "básico",
    },
]


@tool
def search_candidates(query: str) -> list:
    """
    Busca candidatos en la base de datos de RRHH
    según los criterios solicitados.
    """
    return CANDIDATES


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


research_agent = llm.bind_tools([search_candidates])

def research_node(state):
    prompt = f"""
    Sos un agente de investigación especializado en selección de personal.

    Tu tarea es investigar los candidatos disponibles para la siguiente vacante:

    {state["job_requirements"]}

    Utilizá la herramienta search_candidates para obtener la información
    de los candidatos.

    No hagas scoring ni elijas al mejor candidato.
    Solo recopilá y organizá la información relevante para que otro agente
    pueda analizarla.
    """

    response = research_agent.invoke(prompt)

    if response.tool_calls:
        tool_call = response.tool_calls[0]

        tool_result = search_candidates.invoke(
            tool_call["args"]
        )

        return {
            "candidates": tool_result,
            "research_output": str(tool_result)
        }

    return {
        "research_output": response.content
    }

if __name__ == "__main__":
    test_state = {
        "messages": [],
        "job_requirements": {
            "role": "Data Analyst",
            "experience": "2+ años",
            "python": "intermedio/avanzado",
            "sql": "intermedio/avanzado",
            "bi": "Power BI o equivalente",
            "statistics": "conocimientos aplicados",
            "financial_sector": "deseable"
        },
        "candidates": [],
        "research_output": "",
        "analysis_output": "",
        "validation": "",
        "next_agent": "",
        "task_completed": False
    }

    result = research_node(test_state)

    print("\n--- RESEARCH RESULT ---")
    print(result)