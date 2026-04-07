import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)
# Load model
model = joblib.load("models/model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        df = pd.DataFrame([data])
        df = pd.get_dummies(df)
        feature_columns = getattr(model, "feature_names_in_", df.columns)
        df = df.reindex(columns=feature_columns, fill_value=0)

        prediction = model.predict(df)[0]

        return jsonify({
            "status": "success",
            "prediction": str(prediction)
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })