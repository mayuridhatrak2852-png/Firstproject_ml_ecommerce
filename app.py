"""
AI Intelligence Platform - Flask Backend
=========================================
Combined web app with two ML modules:
  1. Customer Intelligence  - RFM segmentation + purchase probability
  2. Product Recommendation - KMeans clustering + TF-IDF + KNN similarity

Run:  python app.py
Open:  http://127.0.0.1:5000
"""

import os
import pickle
import random
import warnings

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE, "data")
MODEL_DIR = os.path.join(BASE, "models")

DMART_CSV    = os.path.join(DATA_DIR,  "D_Mart.csv")
CUST_CSV     = os.path.join(DATA_DIR,  "ecommerce_customers.csv")
CLF_PKL      = os.path.join(MODEL_DIR, "customer_classifier.pkl")
REG_PKL      = os.path.join(MODEL_DIR, "purchase_regressor.pkl")
REG_COLS_PKL = os.path.join(MODEL_DIR, "regressor_columns.pkl")
CACHE_PKL    = os.path.join(BASE,      "model_cache.pkl")


# ═════════════════════════════════════════════════════════════════
#  SECTION 1 : CUSTOMER INTELLIGENCE MODULE
# ═════════════════════════════════════════════════════════════════

# load customer models once
_clf       = joblib.load(CLF_PKL)
_reg       = joblib.load(REG_PKL)
_reg_cols  = joblib.load(REG_COLS_PKL)

_cust_df   = pd.read_csv(CUST_CSV).dropna(subset=["Recency","Frequency","Monetary"]).reset_index(drop=True)
_cust_knn  = NearestNeighbors(n_neighbors=6, metric="euclidean")
_cust_knn.fit(_cust_df[["Recency","Frequency","Monetary"]].values)


def _build_reg_input(data):
    """Map simple form inputs to the 6531-column one-hot frame the regressor needs."""
    row = {col: 0.0 for col in _reg_cols}

    r = float(data.get("Days_Since_Last_Purchase", data.get("Recency", 30)))
    f = float(data.get("Num_Transactions",         data.get("Frequency", 5)))
    m = float(data.get("Total_Spend",              data.get("Monetary", 500)))

    nums = {
        "Recency": r, "Frequency": f, "Monetary": m,
        "Total_Spend": m, "Total_Purchases": f,
        "Age": float(data.get("Age", 30)),
        "Session_Count":       f * 3,
        "Avg_Session_Duration": 8.0,
        "Pages_Viewed":        f * 5,
        "Product_Views":       f * 4,
        "Add_to_Cart_Count":   f * 2,
        "Avg_Order_Value":     m / max(f, 1),
        "Interaction_Score":   f * 10,
    }
    for k, v in nums.items():
        if k in row:
            row[k] = v

    # set one-hot flags
    for prefix, value in [("Gender", data.get("Gender","Male")),
                          ("Location", data.get("Location","Mumbai")),
                          ("Device_Type", data.get("Device_Type","Mobile"))]:
        col = f"{prefix}_{value}"
        if col in row:
            row[col] = 1.0

    return pd.DataFrame([row], columns=_reg_cols)


def predict_customer(data):
    """Predict customer segment + purchase probability from form input."""
    try:
        r = float(data.get("Days_Since_Last_Purchase", data.get("Recency", 30)))
        f = float(data.get("Num_Transactions",         data.get("Frequency", 5)))
        m = float(data.get("Total_Spend",              data.get("Monetary", 500)))

        # classifier
        rfm_df     = pd.DataFrame([{"Recency": r, "Frequency": f, "Monetary": m}])
        clf_proba  = float(_clf.predict_proba(rfm_df)[0][1])

        # regressor
        reg_input    = _build_reg_input(data)
        purchase_prob = float(np.clip(_reg.predict(reg_input)[0], 0, 1))

        # segment from classifier confidence
        if clf_proba > 0.65:
            seg, emoji, color = "High Value Customer", "\U0001f525", "high"
        elif clf_proba > 0.35:
            seg, emoji, color = "Medium Value Customer", "\u26a1", "medium"
        else:
            seg, emoji, color = "Low Value Customer", "\U0001f9ca", "low"

        # category recommendation via KNN on customer data
        dists, idxs = _cust_knn.kneighbors([[r, f, m]])
        cats = _cust_df.iloc[idxs[0]]["Product_Category"].dropna()
        top_cats = list(dict.fromkeys(cats.value_counts().index.tolist()))
        n = min(3, len(top_cats))
        reco_cats = random.sample(top_cats, n) if len(top_cats) >= n else top_cats

        return {
            "status": "success",
            "customer_segment": seg,
            "segment_emoji": emoji,
            "segment_color": color,
            "segment_label": f"{emoji} {seg}",
            "purchase_probability": round(purchase_prob, 3),
            "clf_confidence": round(clf_proba, 3),
            "recommended_categories": reco_cats,
            "rfm_scores": {"recency": r, "frequency": f, "monetary": m},
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ═════════════════════════════════════════════════════════════════
#  SECTION 2 : PRODUCT RECOMMENDATION MODULE
# ═════════════════════════════════════════════════════════════════

_prd_df      = None   # product dataframe
_prd_X       = None   # feature matrix for KNN
_prd_kmeans  = None
_prd_k       = None
_prd_tfidf   = None   # TfidfVectorizer
_prd_tfidf_m = None   # sparse TF-IDF matrix
_prd_idx     = None   # {name_lower: row_index}
_prd_loaded  = False


def _freq_enc(series):
    freq = series.value_counts(normalize=True)
    return series.map(freq).fillna(0).values.astype(float)


def _load_products():
    """Load, clean, engineer features, train KMeans, build TF-IDF. Cached to disk."""
    global _prd_df, _prd_X, _prd_kmeans, _prd_k, _prd_tfidf, _prd_tfidf_m, _prd_idx, _prd_loaded
    if _prd_loaded:
        return

    # try cache first
    if os.path.exists(CACHE_PKL):
        try:
            c = pickle.load(open(CACHE_PKL, "rb"))
            if "tfidf" in c:
                _prd_df, _prd_X       = c["df"], c["X"]
                _prd_kmeans, _prd_k   = c["kmeans"], c["best_k"]
                _prd_tfidf, _prd_tfidf_m = c["tfidf"], c["name_matrix"]
                _prd_idx = {n.lower(): i for i, n in enumerate(_prd_df["name"])}
                _prd_loaded = True
                print(f"[Products] Loaded cache. K={_prd_k}")
                return
        except Exception:
            os.remove(CACHE_PKL)

    print("[Products] Building models...")

    # ── clean data ──
    df = pd.read_csv(DMART_CSV)
    df.columns = [c.strip().lower().replace(" ", "") for c in df.columns]
    df.drop(columns=[c for c in ["description","breadcrumbs"] if c in df.columns], inplace=True, errors="ignore")

    for c in ["name","brand","category","subcategory"]:
        df[c] = df.get(c, pd.Series(dtype=str)).fillna("Unknown").astype(str).str.strip()
    for c in ["price","discountedprice"]:
        df[c] = pd.to_numeric(df.get(c, pd.Series(dtype=float)), errors="coerce")
    df["price"]           = df["price"].fillna(df["price"].median())
    df["discountedprice"] = df["discountedprice"].fillna(df["discountedprice"].median())

    if "quantity" in df.columns:
        qn = df["quantity"].astype(str).str.extract(r"([\d.]+)", expand=False).astype(float)
        qu = df["quantity"].astype(str).str.lower()
        qn = qn * np.where(qu.str.contains(r"\bkg\b|\ k"), 1000, np.where(qu.str.contains(r"\bl\b"), 1000, 1))
        df["quantity_num"] = qn.fillna(float(np.nanmedian(qn.dropna())))
    else:
        df["quantity_num"] = 500.0

    df["discount_pct"] = np.clip(
        np.where(df["price"] > 0, (df["price"] - df["discountedprice"]) / df["price"], 0), 0, 1
    ).astype(float)
    df = df.reset_index(drop=True)

    # ── feature matrix (weighted so KNN respects product type, not just price) ──
    le_c  = LabelEncoder(); le_s = LabelEncoder()
    cat_l  = le_c.fit_transform(df["category"])    / max(len(le_c.classes_)-1, 1)
    sub_l  = le_s.fit_transform(df["subcategory"]) / max(len(le_s.classes_)-1, 1)

    nums = df[["discountedprice","quantity_num","discount_pct"]].values.astype(float)
    for j in range(nums.shape[1]):
        med = np.nanmedian(nums[:, j])
        nums[np.isnan(nums[:, j]), j] = med

    sc = StandardScaler()
    X  = np.nan_to_num(np.column_stack([
        _freq_enc(df["brand"]),
        _freq_enc(df["category"]),
        _freq_enc(df["subcategory"]),
        cat_l * 2.0,
        sub_l * 3.0,
        sc.fit_transform(nums),
    ]))

    # ── TF-IDF on product names ──
    tfidf = TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=1,
                            sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
    tfidf_m = tfidf.fit_transform(df["name"].str.lower())

    # ── KMeans (elbow, min 12) ──
    best_k, max_drop = 2, 0
    inertias = []
    for k in range(2, 21):
        km = KMeans(n_clusters=k, random_state=42, n_init=10); km.fit(X)
        inertias.append((k, km.inertia_))
    for i in range(1, len(inertias)-1):
        drop = (inertias[i-1][1] - inertias[i][1]) / (inertias[i-1][1] + 1e-9)
        if drop > max_drop:
            max_drop, best_k = drop, inertias[i][0]
    if best_k < 12:
        best_k = 12
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10); km.fit(X)
    df["cluster"] = km.labels_

    # save cache
    pickle.dump({"df": df, "X": X, "kmeans": km, "best_k": best_k,
                 "tfidf": tfidf, "name_matrix": tfidf_m}, open(CACHE_PKL, "wb"))

    _prd_df, _prd_X       = df, X
    _prd_kmeans, _prd_k   = km, best_k
    _prd_tfidf, _prd_tfidf_m = tfidf, tfidf_m
    _prd_idx = {n.lower(): i for i, n in enumerate(df["name"])}
    _prd_loaded = True
    print(f"[Products] Done. K={best_k}, vocab={len(tfidf.vocabulary_)}")


def _fmt(row):
    return {
        "name": str(row["name"]), "brand": str(row.get("brand","")),
        "category": str(row.get("category","")), "subcategory": str(row.get("subcategory","")),
        "price": float(row.get("price",0)), "discounted_price": float(row.get("discountedprice",0)),
        "discount_pct": round(float(row.get("discount_pct",0)) * 100, 1),
    }


def _knn_pool(mask, anchor, need, seen):
    """KNN within a boolean-masked subset of the product dataset."""
    gidx = np.where(mask)[0]
    lpos = np.where(gidx == anchor)[0]
    if len(gidx) < 2 or len(lpos) == 0:
        return []
    pX   = _prd_X[mask]
    nn   = NearestNeighbors(n_neighbors=min(need*6+10, len(gidx)), algorithm="ball_tree", metric="euclidean")
    nn.fit(pX)
    _, nbrs = nn.kneighbors(pX[int(lpos[0])].reshape(1,-1))
    out = []
    for li in nbrs[0]:
        nl = _prd_df.iloc[gidx[li]]["name"].lower()
        if nl in seen: continue
        seen.add(nl)
        out.append(_fmt(_prd_df.iloc[gidx[li]]))
        if len(out) >= need: break
    return out


def recommend(product_name, top_n=5):
    """
    4-stage recommendation pipeline:
      1. TF-IDF cosine similarity on product names (most relevant first)
      2. Same subcategory KNN
      3. Same category KNN
      4. Cluster KNN fallback
    """
    _load_products()
    q = product_name.strip().lower()

    # find anchor product
    anchor = _prd_idx.get(q)
    if anchor is None:
        subs = [(n, i) for n, i in _prd_idx.items() if q in n]
        if subs:
            subs.sort(key=lambda x: len(x[0]))
            anchor = subs[0][1]

    qv       = _prd_tfidf.transform([q])
    all_sims = cosine_similarity(qv, _prd_tfidf_m).flatten()

    if anchor is None:
        best = int(np.argmax(all_sims))
        if all_sims[best] < 0.05:
            return None, f"No product matching '{product_name}' found."
        anchor = best

    row     = _prd_df.iloc[anchor]
    seen    = {row["name"].lower()}
    results = []

    # stage 1: TF-IDF name similarity
    thresh  = max(0.05, float(all_sims[anchor]) * 0.25)
    for gi in np.argsort(all_sims)[::-1]:
        if all_sims[gi] < thresh: break
        if gi == anchor: continue
        nl = _prd_df.iloc[gi]["name"].lower()
        if nl in seen: continue
        seen.add(nl)
        results.append(_fmt(_prd_df.iloc[gi]))
        if len(results) >= top_n: break

    # stage 2: same subcategory KNN
    if len(results) < top_n:
        m = (_prd_df["subcategory"] == row["subcategory"]).values
        results.extend(_knn_pool(m, anchor, top_n - len(results), seen))

    # stage 3: same category KNN
    if len(results) < top_n:
        m = (_prd_df["category"] == row["category"]).values
        results.extend(_knn_pool(m, anchor, top_n - len(results), seen))

    # stage 4: cluster KNN
    if len(results) < top_n:
        m = (_prd_df["cluster"] == int(row["cluster"])).values
        results.extend(_knn_pool(m, anchor, top_n - len(results), seen))

    if not results:
        return None, "No similar products found."
    return results[:top_n], None


# ═════════════════════════════════════════════════════════════════
#  SECTION 3 : FLASK APP + ROUTES
# ═════════════════════════════════════════════════════════════════

app = Flask(__name__)
CORS(app)

# pre-load product models at startup
_load_products()


@app.route("/")
@app.route("/dashboard")
def dashboard():
    return render_template("index.html")


@app.route("/predict_customer", methods=["POST"])
def api_predict_customer():
    data = request.get_json(force=True, silent=True) or {}
    if not data:
        return jsonify({"status": "error", "message": "No JSON body"}), 400
    result = predict_customer(data)
    code = 500 if result.get("status") == "error" else 200
    return jsonify(result), code


@app.route("/recommend", methods=["POST"])
def api_recommend():
    data = request.get_json(force=True, silent=True) or {}
    name = data.get("product_name", "").strip()
    if not name:
        return jsonify({"error": "product_name is required"}), 400
    results, err = recommend(name)
    if err:
        return jsonify({"error": err}), 404
    return jsonify({"recommendations": results}), 200


@app.route("/products", methods=["GET"])
def api_products():
    _load_products()
    names = sorted(_prd_df["name"].dropna().unique().tolist())
    return jsonify({"products": names}), 200


@app.route("/stats", methods=["GET"])
def api_stats():
    _load_products()
    return jsonify({
        "total_products":    len(_prd_df),
        "total_clusters":    int(_prd_k),
        "unique_brands":     int(_prd_df["brand"].nunique()),
        "unique_categories": int(_prd_df["category"].nunique()),
    }), 200


if __name__ == "__main__":
    app.run(debug=True, port=5000)
