from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()


# --------------------------------------------------
# TOOL: CALCULAR SCORE
# --------------------------------------------------

@tool
def calculate_candidate_score(candidate: dict) -> dict:
    """
    Calcula de forma determinística el score de un candidato
    según los criterios definidos para la vacante.
    """

    # ----------------------------------------------
    # Experiencia: 20 puntos máximos
    # 4 años o más = 20 puntos
    # 3 años = 15 puntos
    # 2 años = 10 puntos
    # ----------------------------------------------

    experience_score = min(
        candidate["experience_years"] / 4,
        1
    ) * 20

    # ----------------------------------------------
    # Python: 20 puntos
    # ----------------------------------------------

    python_scores = {
        "básico": 0,
        "intermedio": 10,
        "avanzado": 20,
    }

    python_score = python_scores.get(
        candidate["python"],
        0
    )

    # ----------------------------------------------
    # SQL: 20 puntos
    # ----------------------------------------------

    sql_scores = {
        "básico": 0,
        "intermedio": 10,
        "avanzado": 20,
    }

    sql_score = sql_scores.get(
        candidate["sql"],
        0
    )

    # ----------------------------------------------
    # BI: 15 puntos
    # ----------------------------------------------

    bi_scores = {
        "Power BI": 15,
        "Tableau": 11.25,
    }

    bi_score = bi_scores.get(
        candidate["bi"],
        7.5
    )

    # ----------------------------------------------
    # Estadística: 10 puntos
    # ----------------------------------------------

    statistics = candidate["statistics"]

    if statistics == "avanzada":
        statistics_score = 10

    elif statistics == "intermedia":
        statistics_score = 5

    elif statistics == "básica/intermedia":
        statistics_score = 2.5

    else:
        statistics_score = 0

    # ----------------------------------------------
    # Formación: 10 puntos
    # ----------------------------------------------

    education_scores = {
        "Licenciatura en Economía": 10,
        "Ingeniería Industrial": 7.5,
    }

    education_score = education_scores.get(
        candidate["education"],
        5
    )

    # ----------------------------------------------
    # Sector financiero: 5 puntos
    # ----------------------------------------------

    financial_score = (
        5 if candidate["financial_sector"] else 0
    )

    # ----------------------------------------------
    # SCORE FINAL
    # ----------------------------------------------

    total_score = (
        experience_score
        + python_score
        + sql_score
        + bi_score
        + statistics_score
        + education_score
        + financial_score
    )

    return {
        "name": candidate["name"],
        "score": round(total_score, 2),
        "breakdown": {
            "experience": round(experience_score, 2),
            "python": python_score,
            "sql": sql_score,
            "bi": bi_score,
            "statistics": statistics_score,
            "education": education_score,
            "financial_sector": financial_score,
        }
    }


# --------------------------------------------------
# LLM
# --------------------------------------------------

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# --------------------------------------------------
# ANALYST NODE
# --------------------------------------------------

def analyst_node(state):
    """
    Analiza los candidatos y genera una recomendación.

    El score se calcula mediante Python de forma
    determinística. El LLM se utiliza únicamente
    para interpretar los resultados y generar
    la explicación.
    """

    candidates = state["candidates"]
    requirements = state["job_requirements"]

    print("📊 ANALYST AGENT → calculando scores...")

    # ----------------------------------------------
    # CÁLCULO DETERMINÍSTICO
    # ----------------------------------------------

    scores = []

    for candidate in candidates:

        score = calculate_candidate_score.invoke(
            {
                "candidate": candidate
            }
        )

        scores.append(score)

    # Ordenar de mayor a menor score

    scores.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    print("📊 ANALYST AGENT → scores calculados:")

    for result in scores:

        print(
            f"   {result['name']} → {result['score']}"
        )

    # ----------------------------------------------
    # LLM: INTERPRETACIÓN DE LOS RESULTADOS
    # ----------------------------------------------

    prompt = f"""
Sos un analista especializado en selección de personal.

Tenés que analizar candidatos para esta vacante:

{requirements}

Los scores fueron calculados previamente mediante una
herramienta determinística de Python.

IMPORTANTE:

- NO recalcules los scores.
- NO modifiques los scores.
- NO inventes candidatos.
- NO inventes características de los candidatos.
- Utilizá TODOS los candidatos proporcionados.
- El candidato con mayor score debe ser el recomendado.
- Los scores recibidos son la fuente de verdad.
- Para explicar fortalezas y brechas, utilizá los datos
  originales de los candidatos.
- No infieras información que no esté presente.
- Si un candidato tiene experiencia en el sector financiero,
  indicá que la tiene.
- Si un candidato NO tiene experiencia en el sector financiero,
  indicá que no la tiene.
- No confundas un requisito deseable con un requisito obligatorio.

DATOS ORIGINALES DE LOS CANDIDATOS:

{candidates}

RESULTADOS CALCULADOS POR PYTHON:

{scores}

Generá un informe con esta estructura:

### 1. Ranking completo de candidatos

Mostrá los tres candidatos ordenados de mayor a menor
según el score calculado por Python.

### 2. Score de cada candidato

Mostrá exactamente los scores recibidos.

### 3. Candidato recomendado

Identificá al candidato con mayor score.

### 4. Justificación de la recomendación

Explicá por qué el candidato con mayor score es el
recomendado utilizando exclusivamente los datos disponibles.

### 5. Fortalezas del candidato recomendado

Mencioná las principales fortalezas basándote en sus datos.

### 6. Principales brechas del candidato recomendado

Indicá únicamente las brechas que realmente existan
según los datos.

### 7. Breve comparación con los demás candidatos

Explicá por qué los otros candidatos tienen un score menor.

No agregues información que no esté presente en los datos.
No modifiques ningún score.
"""


    response = llm.invoke(prompt)

    return {
        "analysis_output": response.content
    }


# --------------------------------------------------
# PRUEBA DEL ANALYST AGENT
# --------------------------------------------------

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
            "financial_sector": "deseable",
        },

        "candidates": [

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
        ],

        "research_output": "",

        "analysis_output": "",

        "validation": "",

        "next_agent": "",

        "task_completed": False,
    }


    result = analyst_node(test_state)


    print("\n")
    print("=" * 60)
    print("ANALYSIS RESULT")
    print("=" * 60)

    print(result["analysis_output"])