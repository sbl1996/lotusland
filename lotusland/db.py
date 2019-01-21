from flask import current_app as app, g
import pymongo
from pymongo import MongoClient
import os


def get_db():
    if 'mongo_client' not in g:
        g.mongo_client = MongoClient(
            username=app.config["MONGO_USERNAME"],
            password=app.config["MONGO_PASSWORD"],
        )
    return g.mongo_client.blog


def get_latest_post():
    db = get_db()
    coll = db.posts
    post = coll.find().sort('date', pymongo.DESCENDING).next()
    del post['_id']
    return post


def init_app():
    pass
