# Librerias

import os
import requests
import json
from typing import List
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from IPython.display import Markdown, display, update_display
from openai import OpenAI

# Cargando clave OpenaAi

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

if api_key and api_key[:8] == 'sk-proj-':
    print("La clave de API parece buena")
else:
    print("¿Puede haber un problema con tu clave API? ¡Visita el cuaderno de resolución de problemas!")

MODEL = 'gpt-4o-mini'
openai = OpenAI()

# La clase para representar una Página Web
class Website:
    """
    Una clase de utilidad para representar un sitio web que hemos scrappeado, ahora con enlaces
    """

    def __init__(self, url):
        self.url = url
        response = requests.get(url)
        self.body = response.content
        soup = BeautifulSoup(self.body, 'html.parser')
        self.title = soup.title.string if soup.title else "Sin título"
        if soup.body:
            for irrelevant in soup.body(["script", "style", "img", "input"]):
                irrelevant.decompose()
            self.text = soup.body.get_text(separator="\n", strip=True)
        else:
            self.text = ""
        links = [link.get('href') for link in soup.find_all('a')]
        self.links = [link for link in links if link]

    def get_contents(self):
        return f"Título de la Web:\n{self.title}\nContenido de la Web:\n{self.text}\n\n"

noticias = Website("https://cnn.com")
print(noticias.get_contents())
print(f"Enlaces:", noticias.links)


link_system_prompt = "Se te proporciona una lista de enlaces que se encuentran en una página web. \
Puedes decidir cuáles de los enlaces serían los más relevantes para incluir en un folleto sobre la empresa, \
como enlaces a una página Acerca de, una página de la empresa, las carreras/empleos disponibles o páginas de Cursos/Packs.\n"
link_system_prompt += "Debes responder en JSON como en este ejemplo:"
link_system_prompt += """
{
    "links": [
        {"type": "Pagina Sobre nosotros", "url": "https://url.completa/aqui/va/sobre/nosotros"},
        {"type": "Pagina de Cursos": "url": "https://otra.url.completa/courses"}
    ]
}
"""