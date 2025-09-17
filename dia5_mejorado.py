# Librerías
import os
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
from bs4 import BeautifulSoup

# Carga de variables de entorno
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
if api_key and api_key.startswith('sk-proj-'):
    print("La clave de API parece buena")
else:
    print("¿Puede haber un problema con tu clave API? ¡Visita el cuaderno de resolución de problemas!")


class Website:
    def __init__(self, url, wait_time=30_000):
        self.url = url

        with sync_playwright() as p:
            # Lanzamos Chromium en modo headless
            browser = p.chromium.launch(headless=True)
            # Creamos un contexto con un user-agent realista
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/115.0.0.0 Safari/537.36"
                ),
                locale="es-ES",
                timezone_id="Europe/Madrid",
                java_script_enabled=True,
            )

            # Inyectamos scripts de stealth ANTES de cualquier otro script de la página
            context.add_init_script("""
                // 1) Eliminar navigator.webdriver
                Object.defineProperty(navigator, 'webdriver', {get: () => false});
                // 2) Definir un array de idiomas
                Object.defineProperty(navigator, 'languages', {get: () => ['es-ES', 'es']});
                // 3) Simular propiedades de plugins
                Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
                // 4) Simular objeto chrome
                window.chrome = { runtime: {} };
                // 5) WebGL vendor/renderer
                const getParameter = WebGLRenderingContext.prototype.getParameter;
                WebGLRenderingContext.prototype.getParameter = function(param) {
                    if (param === 37445) return 'Intel Inc.';      // UNMASKED_VENDOR_WEBGL
                    if (param === 37446) return 'Intel Iris OpenGL Engine'; // UNMASKED_RENDERER_WEBGL
                    return getParameter.call(this, param);
                };
            """)

            page = context.new_page()

            # Intentamos cargar la página y esperar al <body>
            try:
                page.goto(self.url, timeout=wait_time)
                page.wait_for_selector("body", timeout=wait_time)
            except PlaywrightTimeout:
                print("⚠️ Tiempo de espera excedido al cargar la página.")

            # Extraemos el HTML procesado
            html = page.content()
            soup = BeautifulSoup(html, "html.parser")

            # Título
            self.title = soup.title.string.strip() if soup.title else "Sin título"

            # Texto limpio
            if soup.body:
                for tag in soup.body(["script", "style", "img", "input"]):
                    tag.decompose()
                self.text = soup.body.get_text(separator="\n", strip=True)
            else:
                self.text = ""

            # Enlaces
            raw_links = [a.get("href") for a in soup.find_all("a")]
            self.links = [link for link in raw_links if link]

            # Cierre
            context.close()
            browser.close()

    def get_contents(self):
        parts = [
            f"Título de la Web: {self.title}",
            "",
            "Contenido de la Web:",
            self.text,
            "",
            f"Enlaces encontrados ({len(self.links)}):"
        ]
        parts.extend(self.links)
        return "\n".join(parts)


if __name__ == "__main__":
    url = "https://cursos.frogamesformacion.com"
    site = Website(url)
    print(site.get_contents())
