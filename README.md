🚀 E-Commerce ML System with AI Recommendations

An end-to-end Machine Learning project that predicts customer value, purchase probability, and provides AI-driven product recommendations using real-world data patterns.

📌 Project Overview

This project simulates a real-world e-commerce intelligence system that helps businesses:

Identify high-value customers
Predict purchase likelihood
Recommend personalized products using AI

Built with a combination of Machine Learning, Feature Engineering, and Flask API.

🔥 Key Features
🧠 Customer Intelligence
Classifies customers into:
🔥 High Value
❄️ Medium Value
🧊 Low Value
📈 Purchase Prediction
Predicts probability (0 → 1) of customer making a purchase
🛍️ AI-Based Recommendation System
Uses similarity-based learning (not hardcoded rules)
Suggests products dynamically based on behavior
🌐 REST API (Flask)
Real-time predictions via API endpoints

🏗️ Project Structure

ecommerce-ml-project/
│
├── app/
│   └── app.py                # Flask API
│
├── src/
│   ├── train.py             # Classification model
│   ├── train_regression.py  # Purchase prediction model
│   ├── preprocessing.py     # Data processing
│   └── predict.py           # Prediction logic
│
├── notebooks/
│   └── realistic_ecommerce_data.ipynb
│
├── models/                  # Saved ML models
├── requirements.txt
├── .gitignore
└── README.md

⚙️ Tech Stack
Python 🐍
Pandas & NumPy
Scikit-learn
Flask
Machine Learning (Classification + Regression + KNN)

📊 Machine Learning Approach
🔹 Feature Engineering (RFM Model)
Recency – Days since last purchase
Frequency – Number of transactions
Monetary – Total spend
🔹 Models Used
Classification → Customer Segmentation
Regression → Purchase Probability
KNN → Product Recommendation

🔌 API Endpoints
🔹 Predict Customer + Recommendation

POST /predict_all

📥 Sample Input

{
  "Total_Spend": 30000,
  "Num_Transactions": 60,
  "Days_Since_Last_Purchase": 2
}

📤 Sample Output

{
  "customer_score": "🔥 High Value Customer",
  "purchase_probability": 0.91,
  "recommended_products": [
    "Beauty",
    "Sports",
    "Fashion"
  ],
  "status": "success"
}

▶️ How to Run
1. Clone repo
   git clone https://github.com/your-username/ecommerce-ml-project.git
   cd ecommerce-ml-project

2. Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

3. Install dependencies
pip install -r requirements.txt

4. Train models
  python src/train.py
  python src/train_regression.py

5. Run API
  python app/app.py

💡 Why This Project Stands Out
✅ End-to-end ML pipeline (data → model → API)
✅ Multiple ML models working together
✅ Real-world business use case
✅ AI-based recommendation (not rule-based)
✅ Production-style architecture

🚀 Future Improvements
Add Streamlit dashboard (UI)
Deploy on AWS / Render
Add real-time user tracking
Improve recommendation using deep learning

👨‍💻 Author

Koshlesh Wandhe
https://github.com/Koshleshwandhe