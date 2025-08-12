import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import re
import json
from datetime import datetime

# Load category rules
def load_category_rules(config_path="category_rules.json"):
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading category_rules.json: {e}")
        return {
            "Income": ["salary", "bonus", "freelance", "investment", "neft.*cr", "imps.*cr", "upi.*cr", "\\[Salary\\]", "\\[Bonus\\]"],
            "Fixed Expenses": ["rent", "emi", "insurance", "electricity", "phone", "bill", "loan", "\\[Rent\\]", "\\[Electricity\\]"],
            "Discretionary Expenses": ["shopping", "dining", "entertainment", "travel", "amazon", "zomato", "\\[Shopping\\]", "\\[Dining\\]"],
            "Savings": ["fd", "rd", "mutual fund", "deposit", "sip", "\\[FD\\]", "\\[RD\\]"],
            "Red Flags": ["overdraft", "bounce", "insufficient funds", "cash.*\\d{4,}", "upi.*\\d{5,}.*cash"]
        }

CATEGORY_RULES = load_category_rules()

# Rule-based categorization for labeling
def categorize_description(desc, amount=0.0, is_debit=False):
    desc = str(desc).lower().strip()
    if not desc:
        return "Uncategorized"
    for category, patterns in CATEGORY_RULES.items():
        for pattern in patterns:
            if re.search(pattern, desc, re.IGNORECASE):
                return category
    if is_debit and amount > 10000 and any(k in desc for k in ["cash", "atm"]):
        return "Red Flags"
    return "Uncategorized"

# Aggregate CSVs from cleaned_csvs/
def load_transaction_data(input_dir="cleaned_csvs"):
    dataframes = []
    required_columns = ["Date", "Description", "Debit", "Credit", "Balance"]
    for file in os.listdir(input_dir):
        if file.endswith(".csv"):
            try:
                df = pd.read_csv(os.path.join(input_dir, file))
                # Validate columns
                missing_cols = [col for col in required_columns if col not in df.columns]
                if missing_cols:
                    print(f"⚠️ Skipping {file}: Missing columns {missing_cols}")
                    continue
                df["Category"] = df.apply(
                    lambda row: categorize_description(
                        row["Description"], 
                        row["Debit"] if row["Debit"] > 0 else row["Credit"],
                        row["Debit"] > 0
                    ), axis=1
                )
                dataframes.append(df[required_columns + ["Category"]])
            except Exception as e:
                print(f"⚠️ Error reading {file}: {e}")
    if not dataframes:
        raise ValueError("No valid CSVs found in cleaned_csvs/")
    return pd.concat(dataframes, ignore_index=True)

# Feature engineering
def extract_features(df):
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", max_features=1000)
    text_features = vectorizer.fit_transform(df["Description"]).toarray()
    numerical_features = df.apply(
        lambda row: row["Debit"] if row["Debit"] > 0 else row["Credit"], axis=1
    ).values
    is_debit = (df["Debit"] > 0).astype(int).values
    dates = pd.to_datetime(df["Date"], errors="coerce")
    day_of_month = dates.dt.day.fillna(0).values
    is_month_end = (dates.dt.day >= 25).astype(int).values
    is_weekend = dates.dt.weekday.isin([5, 6]).astype(int).values
    return np.concatenate([
        text_features,
        numerical_features.reshape(-1, 1),
        is_debit.reshape(-1, 1),
        day_of_month.reshape(-1, 1),
        is_month_end.reshape(-1, 1),
        is_weekend.reshape(-1, 1)
    ], axis=1), vectorizer

# Train model
def train_model(input_dir="cleaned_csvs"):
    df = load_transaction_data(input_dir)
    if df.empty:
        raise ValueError("No valid transaction data for training")
    
    X, vectorizer = extract_features(df)
    y = df["Category"]
    
    classifier = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    classifier.fit(X, y)
    
    os.makedirs("models", exist_ok=True)
    with open("models/category_classifier.pkl", "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "classifier": classifier}, f)
    print(f"✅ ML model trained and saved to models/category_classifier.pkl")
    print(f"Training data size: {len(df)} transactions, {len(set(y))} categories")
    return df

if __name__ == "__main__":
    train_model()
