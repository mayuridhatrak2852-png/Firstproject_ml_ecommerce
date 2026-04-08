from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import random

# ==============================
# 🚀 INIT
# ==============================
app = Flask(__name__)

# ==============================
# 📦 LOAD MODEL + DATA
# ==============================
model = joblib.load("../models/model.pkl")
products_df = pd.read_csv("../data/realistic_ecommerce_data.csv")

# ==============================
# 🤖 BUILD KNN MODEL
# ==============================
features = products_df[["Recency", "Frequency", "Monetary"]]

knn = NearestNeighbors(n_neighbors=5, metric="euclidean")
knn.fit(features)

# ==============================
# 🤖 AI RECOMMENDATION
# ==============================
def recommend_products_ai(data):

    user_vector = [[
        data["Days_Since_Last_Purchase"],
        data["Num_Transactions"],
        data["Total_Spend"]
    ]]

    distances, indices = knn.kneighbors(user_vector)

    similar_users = products_df.iloc[indices[0]]

    categories = similar_users["Product_Category"].dropna()

    if len(categories) > 0:
        # Get ranked categories
        recommendations = categories.value_counts().index.tolist()

        # Remove duplicates safely
        recommendations = list(dict.fromkeys(recommendations))

        # Add randomness (AI-like behavior)
        if len(recommendations) >= 3:
            recommendations = random.sample(recommendations, 3)
    else:
        recommendations = ["Electronics", "Fashion", "Home"]

    return recommendations

# ==============================
# 🏠 HOME
# ==============================
@app.route("/")
def home():
    return "✅ AI Recommendation API Running"

# ==============================
# 🔥 FINAL API
# ==============================
@app.route("/predict_all", methods=["POST"])
def predict_all():
    try:
        data = request.json

        # ==============================
        # INPUT → RFM
        # ==============================
        df = pd.DataFrame([{
            "Recency": data["Days_Since_Last_Purchase"],
            "Frequency": data["Num_Transactions"],
            "Monetary": data["Total_Spend"]
        }])

        # ==============================
        # ✅ CLASSIFIER PROBABILITY
        # ==============================
        prob = model.predict_proba(df)[0][1]

        # ==============================
        # CUSTOMER SEGMENT
        # ==============================
        if prob > 0.7:
            score = "🔥 High Value Customer"
        elif prob > 0.4:
            score = "⚡ Medium Value Customer"
        else:
            score = "🧊 Low Value Customer"

        # ==============================
        # 🤖 AI RECOMMENDATION
        # ==============================
        recommended_products = recommend_products_ai(data)

        return jsonify({
            "status": "success",
            "purchase_probability": float(round(prob, 3)),
            "customer_score": score,
            "recommended_products": recommended_products
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

# ==============================
# ▶️ RUN
# ==============================
if __name__ == "__main__":
    app.run(debug=True)