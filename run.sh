#! /bin/bash 
# file: run.sh

export FLASK_APP=lotusland
export FLASK_ENV=development
flask run --host=0.0.0.0 --port=8080