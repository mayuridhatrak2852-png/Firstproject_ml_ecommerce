from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# ==============================
# 🚀 INIT
# ==============================
app = Flask(__name__)

# ==============================
# 📦 LOAD MODEL + DATA
# ==============================
model = joblib.load("../models/reg_model.pkl")
products_df = pd.read_csv("../data/realistic_ecommerce_data.csv")

# ==============================
# 🤖 BUILD KNN MODEL (AI CORE)
# ==============================
features = products_df[["Recency", "Frequency", "Monetary"]]

knn = NearestNeighbors(n_neighbors=5, metric="euclidean")
knn.fit(features)

# ==============================
# 🤖 AI RECOMMENDATION FUNCTION
# ==============================
def recommend_products_ai(data):

    user_vector = [[
        data["Days_Since_Last_Purchase"],
        data["Num_Transactions"],
        data["Total_Spend"]
    ]]

    distances, indices = knn.kneighbors(user_vector)

    similar_users = products_df.iloc[indices[0]]

    recommendations = (
        similar_users["Product_Category"]
        .value_counts()
        .head(3)
        .index
        .tolist()
    )

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
        print("🔥 HIT API")

        data = request.json

        # ==============================
        # RFM INPUT
        # ==============================
        df = pd.DataFrame([{
            "Recency": data["Days_Since_Last_Purchase"],
            "Frequency": data["Num_Transactions"],
            "Monetary": data["Total_Spend"]
        }])

        # ==============================
        # ML PREDICTION
        # ==============================
        prob = model.predict_proba(df)[0][1]

        # ==============================
        # CUSTOMER SCORE
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

        print("PROB:", prob)
        print("SCORE:", score)
        print("AI RECOMMEND:", recommended_products)

        return jsonify({
            "status": "success",
            "purchase_probability": float(prob),
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