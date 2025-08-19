#train_model.py
import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
#from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------------
# Part 1: Transaction Categorization Model
# -------------------------------

# Sample data
transaction_data = pd.DataFrame({
    "Description": [
        "Salary Credit", "Rent Payment", "EMI Payment",
        "Shopping", "Utility Bill", "Loan Repayment", "Payroll Deposit"
    ],
    "Category": ["Income", "Expense", "Debt", "Expense", "Expense", "Debt", "Income"]
})

# Vectorize text
vectorizer = TfidfVectorizer()
X_trans = vectorizer.fit_transform(transaction_data["Description"])

# Train classifier
category_classifier = RandomForestClassifier()
category_classifier.fit(X_trans, transaction_data["Category"])

# If label encoding used, replace `None` below with the actual encoder
label_encoder = None  # Placeholder

# Save models
os.makedirs("models", exist_ok=True)
with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f, protocol=4)
with open("models/classifier.pkl", "wb") as f:
    pickle.dump(category_classifier, f, protocol=4)
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f, protocol=4)

print("✅ Categorization model saved")

# -------------------------------
# Part 2: Loan Approval Model
# -------------------------------

# 1. Generate Mock Data
np.random.seed(42)
n_samples = 500

data = {
    "avg_income": np.random.normal(45000, 15000, n_samples).clip(10000, 120000),
    "net_surplus": np.random.normal(10000, 7000, n_samples).clip(-5000, 50000),
    "dti": np.random.normal(35, 15, n_samples).clip(0, 100),
    "savings_rate": np.random.normal(15, 10, n_samples).clip(0, 80),
    "red_flag_count": np.random.poisson(1, n_samples).clip(0, 6),
    "cibil": np.random.normal(700, 60, n_samples).clip(300, 900),
    "income_trend": np.random.uniform(-0.1, 0.2, n_samples),
    "emi_bounce_count": np.random.poisson(0.5, n_samples).clip(0, 5),
    "income_stability": np.random.choice([0, 1], size=n_samples),  # NEW FEATURE
    "discretionary_expenses": np.random.normal(5000, 2000, n_samples).clip(0, 30000)  # NEW FEATURE
}

# Approval logic for mock label
labels = (
    (data["cibil"] > 700) &
    (data["net_surplus"] > 8000) &
    (data["dti"] < 40) &
    (np.array(data["red_flag_count"]) <= 2)
).astype(int)

df = pd.DataFrame(data)
df["label"] = labels

# 2. Prepare Features and Target
X = df.drop("label", axis=1)
y = df["label"]

# 3. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Evaluate Multiple Models
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss',verbosity=0),
    #"LightGBM": LGBMClassifier()
}

print("\n🔍 Cross-validation Results:")
for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
    print(f"{name}: Accuracy = {scores.mean():.3f} ± {scores.std():.3f}")

# 5. Grid Search for Best RandomForest
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_scaled, y)

best_model = grid_search.best_estimator_
print(f"\n✅ Best RandomForest Params: {grid_search.best_params_}")

# 6. Final Evaluation
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# 7. Save Loan Approval Model and Scaler
with open("models/loan_approval_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\n📁 Model and Scaler saved in 'models/' directory.")
