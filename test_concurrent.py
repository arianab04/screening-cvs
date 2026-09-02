from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen
import json
import time


URL = "http://127.0.0.1:8000/tasks"

payload = {
    "role": "Analista de Recursos Humanos",
    "title": "Analista de Recursos Humanos",
    "description": "Screening de candidatos para un puesto de Recursos Humanos.",
    "requirements": [
        "Experiencia en Recursos Humanos",
        "Experiencia en recruiting",
        "Capacidad analítica",
        "Buenas habilidades de comunicación",
        "Organización"
    ]
}


def create_task(_):
    data = json.dumps(payload).encode("utf-8")

    request = Request(
        URL,
        data=data,
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    with urlopen(request) as response:
        body = json.loads(response.read().decode("utf-8"))
        return response.status, body


print("Enviando 5 requests concurrentes...\n")

with ThreadPoolExecutor(max_workers=5) as executor:
    results = list(executor.map(create_task, range(5)))


for status, result in results:
    print(f"HTTP {status} -> {result}")


print("\nPrueba finalizada.")