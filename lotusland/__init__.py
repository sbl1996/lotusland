import os

from flask import Flask
from flask_cors import CORS
from PIL import ImageFile


def create_app(test_config=None):
    """Create and configure an instance of the Flask application."""

    app = Flask(__name__, instance_relative_config=True,
                static_folder=os.path.join("public", "static"))
    CORS(app)
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    app.config.from_pyfile('config.py', silent=True)

    # ensure the instance folder exists
    os.makedirs(app.instance_path, exist_ok=True)

    # register the database commands
    from lotusland import serving, db, web
    with app.app_context():
        db.init_app()
        serving.init_app()

    app.register_blueprint(serving.bp)
    app.register_blueprint(web.bp)

    return app
