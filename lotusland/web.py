from datetime import datetime
import os

from flask import current_app as app, Blueprint, send_from_directory, jsonify, request

from captcha.image import ImageCaptcha
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


@bp.route("/api/posts", methods=["GET"])
def get_posts_by_month():
    month = request.args.get('month')
    month = datetime.strptime(month, '%Y%m')
    posts = db.get_posts_by_month(month)
    return jsonify(posts)


@bp.route("/api/posts/search", methods=["GET"])
def search_posts():
    keyword = request.args.get('keyword')
    posts = db.search_posts(keyword)
    return jsonify(posts)


@bp.route("/api/posts/latest", methods=["GET"])
def get_latest_post():
    p = db.get_latest_post()
    return jsonify(p)


@bp.route("/api/cv", methods=["GET"])
def get_resume():
    return send_from_directory(app.root_path, 'CV.pdf')
