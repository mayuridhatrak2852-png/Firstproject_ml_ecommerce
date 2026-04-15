# AI Intelligence Platform

A combined ML-powered web application that does two things:

1. **Customer Intelligence** — Takes customer purchase data (how recently they bought, how often, how much they spent) and predicts whether they are a high, medium, or low value customer. Also estimates the probability of their next purchase and suggests product categories they might be interested in.

2. **Product Recommendation Engine** — Given any product name from the D-Mart catalog, it finds and ranks the most similar products using a combination of text matching (TF-IDF) and nearest-neighbor search (KNN) within product clusters.

Both modules run together in a single Flask app with one dashboard UI.

---

## Folder Structure

```
.
├── app.py                  # main Flask app (single file, all backend logic)
├── requirements.txt        # python dependencies
├── .gitignore
│
├── data/
│   ├── D_Mart.csv              # ~5200 D-Mart products (name, brand, category, price, etc.)
│   └── ecommerce_customers.csv # ~5000 customer records (RFM scores, demographics, purchase history)
│
├── models/
│   ├── customer_classifier.pkl     # RandomForest classifier - predicts high/low value customer (trained on RFM)
│   ├── purchase_regressor.pkl      # RandomForest regressor - predicts purchase probability (uses full feature set)
│   └── regressor_columns.pkl       # column names the regressor expects (saved during training)
│
└── templates/
    └── index.html          # frontend dashboard (single page, no framework, vanilla JS)
```

That's it. No extra folders, no scattered scripts.

---

## How It Works

### Customer Intelligence Module

- The user enters three main inputs: days since last purchase (Recency), number of transactions (Frequency), and total spend (Monetary). Optional fields include age, gender, location, and device type.
- **Classifier** (`customer_classifier.pkl`): a RandomForest trained on RFM features. Uses `predict_proba` to get a confidence score, which determines the customer segment (High/Medium/Low value).
- **Regressor** (`purchase_regressor.pkl`): a RandomForest trained on the full feature set (6531 one-hot encoded columns). A preprocessing adapter maps the few UI inputs into the full feature frame by filling defaults for missing columns.
- **Category Recommendations**: KNN on the customer dataset finds similar customers and returns their most-purchased product categories.

### Product Recommendation Module

The recommendation uses a 4-stage pipeline, in priority order:

1. **TF-IDF name matching** — builds a vocabulary of ~15k word tokens from all product names. When the user searches, it computes cosine similarity between the query and every product name. Products above a threshold are returned, sorted by relevance. This handles exact names, partial matches, and multi-word queries well.

2. **Same subcategory KNN** — if stage 1 didn't fill all slots, it runs KNN within the same subcategory (e.g. "Dry Fruits" or "Cooking Oil") using a feature matrix that weights subcategory and category membership heavily.

3. **Same category KNN** — expands to the broader category if still not enough results.

4. **Cluster KNN fallback** — falls back to the K-Means cluster the anchor product belongs to.

The feature matrix for KNN includes frequency-encoded brand/category/subcategory, label-encoded category (weight 2x) and subcategory (weight 3x), plus StandardScaler-normalized price, quantity, and discount. This means KNN keeps products of the same type together even when prices differ.

K-Means uses K=12 (found via elbow method, with a minimum of 12 to keep clusters granular enough for a 5k product catalog).

---

## Setup & Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
python app.py
```

The server starts at `http://127.0.0.1:5000`.  
On first run it builds and caches the product models (takes ~30-60 seconds). Subsequent runs load from cache instantly.

### 3. Use it

- Open `http://127.0.0.1:5000` in your browser
- **Top section**: fill in customer details, click "Analyze Customer"
- **Bottom section**: type a product name, click "Get Recommendations"

---

## API Endpoints

| Endpoint             | Method | What it does                              |
|----------------------|--------|-------------------------------------------|
| `/`                  | GET    | Dashboard UI                              |
| `/predict_customer`  | POST   | Customer segment + purchase probability   |
| `/recommend`         | POST   | Product recommendations (JSON: `{"product_name": "..."}`) |
| `/products`          | GET    | All product names (for autocomplete)      |
| `/stats`             | GET    | Catalog stats (product count, clusters, brands, categories) |

---

## Tech Stack

- **Backend**: Python, Flask, scikit-learn, pandas, numpy
- **ML**: RandomForest (classification + regression), K-Means, KNN, TF-IDF
- **Frontend**: HTML, CSS, vanilla JavaScript (single file, no build tools)
- **Models**: pre-trained and saved as pickle/joblib files

---

## Notes

- `model_cache.pkl` is auto-generated on first run and listed in `.gitignore`. If anything seems off with recommendations, delete it and restart — it rebuilds automatically.
- The regressor expects 6531 columns because it was trained on one-hot encoded data. The app handles this internally; the user only needs to fill 3-7 simple form fields.
- Both modules are independent — you could remove one without breaking the other.
