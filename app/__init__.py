"""Flask application factory."""

from flask import Flask
from flask_caching import Cache

cache = Cache()


def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.update(
        CACHE_TYPE="SimpleCache",
        CACHE_DEFAULT_TIMEOUT=3600,
    )
    cache.init_app(app)

    from app.routes.main import bp as main_bp
    from app.routes.exports import bp as exports_bp
    from app.routes.banco import bp as banco_bp
    app.register_blueprint(main_bp)
    app.register_blueprint(exports_bp)
    app.register_blueprint(banco_bp)

    return app
