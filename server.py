"""
server.py — Sirve el index.html en http://localhost:3000
Correr junto a Streamlit: python server.py
"""
from http.server import HTTPServer, SimpleHTTPRequestHandler
import webbrowser

PORT = 3000

class Handler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Silenciar logs del servidor

if __name__ == "__main__":
    server = HTTPServer(("localhost", PORT), Handler)
    print(f" Página Aracari corriendo en http://localhost:{PORT}")
    print(f"   Asegúrate de tener Streamlit corriendo también: streamlit run app.py")
    webbrowser.open(f"http://localhost:{PORT}")
    server.serve_forever()