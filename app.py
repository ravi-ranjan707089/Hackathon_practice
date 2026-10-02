from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    features = data["features"]

    prediction = model.predict([features])

    return jsonify({
        "prediction": int(prediction[0])
    })


if __name__ == "__main__":
    app.run(debug=True)