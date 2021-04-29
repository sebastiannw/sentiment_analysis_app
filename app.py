from flask_cors import CORS, cross_origin
from flask import Flask, request, jsonify
import sys

sys.path.insert(0, 'code/')
from predict import main

app = Flask(__name__)
cors = CORS(app)
#app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/')
def hello():
    """Return a friendly HTTP greeting."""
    print("I am inside hello world")
    return 'Hello World! CD'

@app.route("/predict/<review>", methods=["GET"])
#@cross_origin
def predict(review):
    print(review)
    score = main(review)
    return jsonify({"reviewValue": score})


if __name__ == '__main__':
   # Setting debug to True enables debug output. This line should be
   # removed before deploying a production app.
   #app.debug = True
   app.run(host='0.0.0.0', port=8080, debug=True)