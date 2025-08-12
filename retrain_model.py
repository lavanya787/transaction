import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle

# Default training data (expanded to include "Bonus" and "Fixed Deposit")
data = pd.DataFrame({
    "Description": [
        "Salary Credit", "Rent Payment", "EMI Payment", "Shopping", "Utility Bill",
        "Loan Repayment", "Payroll Deposit", "Bonus", "Fixed Deposit", "Overdraft Fee",
        "NEFT CR Salary", "UPI Shopping Amazon", "IMPS CR Bonus"
    ],
    "Category": [
        "Income", "Fixed Expenses", "Debt", "Discretionary Expenses", "Fixed Expenses",
        "Debt", "Income", "Income", "Savings", "Red Flags",
        "Income", "Discretionary Expenses", "Income"
    ]
})

# Train model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["Description"])
model = RandomForestClassifier(random_state=42)
model.fit(X, data["Category"])

# Save model
os.makedirs(r"C:\Users\lavan\OneDrive\Desktop\bank_transaction_tool\models", exist_ok=True)
model_path = r"C:\Users\lavan\OneDrive\Desktop\bank_transaction_tool\models\category_classifier.pkl"
with open(model_path, "wb") as f:
    pickle.dump({
        "vectorizer": vectorizer,
        "classifier": model,
        "label_encoder": None  # No LabelEncoder needed for string categories
    }, f, protocol=4)

print("✅ Categorization model retrained and saved to", model_path)