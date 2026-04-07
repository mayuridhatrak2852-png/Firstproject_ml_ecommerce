from flask import Flask, request, jsonify
import joblib
import pandas as pd

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
# 🛍️ RECOMMENDATION FUNCTION (FINAL)
# ==============================
def recommend_products(data, prob):

    # High value customers
    if prob > 0.7:
        filtered = products_df[
            products_df["Monetary"] > products_df["Monetary"].median()
        ]

    # Medium customers
    elif prob > 0.4:
        filtered = products_df

    # Low customers
    else:
        filtered = products_df[
            products_df["Monetary"] < products_df["Monetary"].median()
        ]

    # Similarity filter
    filtered = filtered[
        (filtered["Monetary"] >= data["Total_Spend"] * 0.5) &
        (filtered["Monetary"] <= data["Total_Spend"] * 1.5)
    ]

    if filtered.empty:
        filtered = products_df

    recommendations = (
        filtered["Product_Category"]
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
    return "✅ FINAL ML API RUNNING"

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
        # PREDICTION
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
        # RECOMMENDATION
        # ==============================
        recommended_products = recommend_products(data, prob)

        print("PROB:", prob)
        print("SCORE:", score)
        print("RECOMMEND:", recommended_products)

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