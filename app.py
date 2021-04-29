from flask_cors import CORS, cross_origin
from flask import Flask, request, jsonify
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/')
def hello():
    """Return a friendly HTTP greeting."""
    print("I am inside hello world")
    return 'Hello World! CD'

@app.route("/analyze/<review>", methods=["GET"])
@cross_origin
def predict(review):
    print(f"This was placed in the url: new-{name}")
    val = {"new-name": name}
    return jsonify(val)


if __name__ == '__main__':
   # Setting debug to True enables debug output. This line should be
   # removed before deploying a production app.
   #app.debug = True
   app.run(host='0.0.0.0', port=8080, debug=True)