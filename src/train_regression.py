import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load data
df = pd.read_csv("data/realistic_ecommerce_data.csv")

# ==============================
# 🔥 STEP 1: DROP USELESS COLUMNS
# ==============================
df = df.drop(["Customer_ID"], axis=1)

# ==============================
# 🔥 STEP 2: HANDLE MISSING VALUES
# ==============================
num_cols = df.select_dtypes(include=["int64", "float64"]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

cat_cols = df.select_dtypes(include=["object"]).columns
df[cat_cols] = df[cat_cols].fillna("Unknown")

# ==============================
# 🎯 TARGET = Purchase Probability
# ==============================
y = df["Purchase_Probability"]
X = df.drop("Purchase_Probability", axis=1)

# Remove Engagement_Level (optional but better)
if "Engagement_Level" in X.columns:
    X = X.drop("Engagement_Level", axis=1)

# ==============================
# 🔁 ENCODING
# ==============================
X = pd.get_dummies(X)

# ==============================
# ✂️ SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 🤖 MODEL
# ==============================
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

# ==============================
# 🚀 TRAIN
# ==============================
model.fit(X_train, y_train)

# ==============================
# 💾 SAVE
# ==============================
joblib.dump(model, "models/reg_model.pkl")
joblib.dump(X.columns.tolist(), "models/reg_columns.pkl")

print("✅ Regression model trained and saved!")