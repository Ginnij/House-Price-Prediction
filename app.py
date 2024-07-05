import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os

# Create flask app
app = Flask(__name__)
housing = pickle.load(open("housing.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = housing.predict(features)
    return render_template("index.html", prediction_text = "Price of House will be approx {}".format(prediction))

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)