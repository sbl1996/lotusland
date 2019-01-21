import os
from flask import current_app as app, Blueprint, send_from_directory, jsonify
from lotusland import db

bp = Blueprint("web", __name__)


@bp.route('/', defaults={'path': ''})
@bp.route('/<path:path>')
def catch_all(path):
    print(path)
    if '.' not in path:
        return send_from_directory(os.path.join(app.root_path, 'public'), "index.html")
    else:
        return send_from_directory(os.path.join(app.root_path, 'public'), path)


@bp.route("/img/<path:path>")
def images(path):
    return send_from_directory(os.path.join(app.root_path, 'public', "img"), path)


@bp.route("/api/posts/latest", methods=["GET"])
def latest_post():
    p = db.get_latest_post()
    return jsonify(p)
