import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv("data/realistic_ecommerce_data.csv")

print("📊 Columns:", df.columns.tolist())

# ==============================
# CLEAN
# ==============================
df = df.drop(columns=["Customer_ID"], errors="ignore")

# ==============================
# 🔥 FIX NUMERIC CONVERSION (FINAL FIX)
# ==============================
for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    except:
        pass

# ==============================
# ✅ USE EXISTING RFM (YOU ALREADY HAVE THEM!)
# ==============================
# Your dataset already has:
# Recency, Frequency, Monetary

df["Recency"].fillna(df["Recency"].median(), inplace=True)
df["Frequency"].fillna(df["Frequency"].median(), inplace=True)
df["Monetary"].fillna(df["Monetary"].median(), inplace=True)

# ==============================
# 🎯 TARGET (SMART SEGMENTATION)
# ==============================
df["Target"] = 0

df.loc[
    (df["Monetary"] > df["Monetary"].quantile(0.7)) &
    (df["Frequency"] > df["Frequency"].quantile(0.7)) &
    (df["Recency"] < df["Recency"].quantile(0.3)),
    "Target"
] = 1

# ==============================
# FEATURES
# ==============================
X = df[["Recency", "Frequency", "Monetary"]]
y = df["Target"]

# ==============================
# MODEL
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# ==============================
# SAVE MODEL
# ==============================
joblib.dump(model, "models/reg_model.pkl")

print("✅ MODEL TRAINED SUCCESSFULLY")