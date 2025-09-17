# imports

import requests
import json
from bs4 import BeautifulSoup
from IPython.display import Markdown, display

# Constantes
OLLAMA_API = "http://localhost:11434/api/chat"
HEADERS = {"Content-Type": "application/json"}
MODEL = "llama3.2"
messages = [
    {"role": "user", "content": "Describe algunas de las aplicaciones comerciales de la IA generativa."}
]

messages2 = [
    {"role": "user", "content": "Eres un asistente que analiza el contenido de un sitio web \
y proporciona un breve resumen, ignorando el texto que podría estar relacionado con la navegación. \
Responder en Markdown."}
]

''' 
# Crea una lista de mensajes utilizando el mismo formato que usamos para OpenAI. Metodo Largo

payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False
    }

resp = requests.post(OLLAMA_API, headers=HEADERS, json=payload)

print("status:", resp.status_code)
#print("raw json:", json.dumps(resp.json(), indent=2, ensure_ascii=False))

# Sólo si la petición fue correcta y existe la clave `message`
if resp.ok and 'message' in resp.json():
    print(resp.json()['message']['content'])
else:
    # Manejo de error: muestra mensaje de Ollama
    print("Error:", resp.json().get('error', 'Respuesta inesperada'))
'''

# Metodo corto con libreria Python
import ollama
response=ollama.chat(model=MODEL, messages=messages2)
print(response['message']['content'])

