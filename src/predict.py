import pickle
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Load models
clf_model = pickle.load(open("models/model.pkl", "rb"))
reg_model = pickle.load(open("models/regression_model.pkl", "rb"))

# Load dataset for recommendation
df = pd.read_csv("data/processed_data.csv")

# Prepare KNN
knn_features = ["Recency", "Frequency", "Monetary"]
knn = NearestNeighbors(n_neighbors=5)
knn.fit(df[knn_features])


def predict_all(data):
    try:
        # Extract input
        total_spend = data.get("Total_Spend", 0)
        num_transactions = data.get("Num_Transactions", 0)
        days_since_last = data.get("Days_Since_Last_Purchase", 0)

        # Create features
        recency = days_since_last
        frequency = num_transactions
        monetary = total_spend

        input_df = pd.DataFrame([{
            "Recency": recency,
            "Frequency": frequency,
            "Monetary": monetary
        }])

        # Prediction
        customer_pred = clf_model.predict(input_df)[0]
        purchase_prob = reg_model.predict(input_df)[0]

        # Clip probability
        purchase_prob = max(0, min(1, purchase_prob))

        # Customer label
        if purchase_prob > 0.7:
            customer_score = "🔥 High Value Customer"
        elif purchase_prob > 0.3:
            customer_score = "❄️ Medium Value Customer"
        else:
            customer_score = "🧊 Low Value Customer"

        # =========================
        # 🔥 AI RECOMMENDATION FIX
        # =========================
        distances, indices = knn.kneighbors(input_df)

        similar_users = df.iloc[indices[0]]

        categories = similar_users["Product_Category"].dropna()

        if len(categories) > 0:
            # Add randomness + diversity
            recommended = (
                categories.value_counts()
                .sample(n=min(3, len(categories)), replace=True)
                .index
                .tolist()
            )
        else:
            recommended = ["Electronics", "Fashion", "Home"]

        return {
            "customer_score": customer_score,
            "purchase_probability": float(round(purchase_prob, 3)),
            "recommended_products": recommended,
            "status": "success"
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }