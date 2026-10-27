"""Punto de entrada WSGI para gunicorn y desarrollo local."""

import logging

# Configura logging para que gunicorn lo capture y Cloud Run lo muestre
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
