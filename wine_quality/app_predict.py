from flask import Flask, request, jsonify
import pandas as pd
import joblib

app=Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    dados = request.get_json()
    data=dados["data"]
    df = pd.DataFrame.from_dict(data)

    model=joblib.load("models/model_pipeline.pkl")

    predictions=model.predict(df)
    output = predictions[0]

    return "Wine Quality Prediction: {}".format(output)

if __name__ == "__main__":
    app.run()