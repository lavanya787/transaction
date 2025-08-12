import pandas as pd
import pickle
import spacy
import re
from datetime import datetime
from fuzzywuzzy import fuzz
from collections import Counter
import numpy as np
import os

# Initialize spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = None
    print("⚠️ spaCy model not available, skipping NLP fallback.")

# Category rules
category_rules = {
    "Income": ["salary", "bonus", "freelance", "investment", "neft.*cr", "imps.*cr", "upi.*cr", r"\[Salary\]", r"\[Bonus\]", r"\[Freelance\]"],
    "Fixed Expenses": ["rent", "emi", "insurance", "electricity", "phone", "bill", "loan", r"\[Rent\]", r"\[Electricity\]"],
    "Discretionary Expenses": ["shopping", "dining", "entertainment", "travel", "amazon", "zomato", r"\[Shopping\]", r"\[Dining\]", r"\[Entertainment\]"],
    "Savings": ["fd", "rd", "mutual fund", "deposit", "sip", r"\[FD\]", r"\[RD\]", r"\[Mutual Fund\]"],
    "Red Flags": ["overdraft", "bounce", "insufficient funds", r"cash.*\d{4,}", r"upi.*\d{5,}.*cash"]
}
# File path for automatic overrides
AUTO_OVERRIDE_FILE = "auto_overrides.csv"

def load_models():
    """Load vectorizer, classifier, and label encoder from disk."""
    model_dir = "models"
    vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")
    classifier_path = os.path.join(model_dir, "classifier.pkl")
    label_encoder_path = os.path.join(model_dir, "label_encoder.pkl")

    vectorizer, classifier, label_encoder = None, None, None

    try:
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
        print(f"[INFO] Loaded vectorizer from {vectorizer_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load vectorizer: {e}")

    try:
        with open(classifier_path, "rb") as f:
            classifier = pickle.load(f)
        print(f"[INFO] Loaded classifier from {classifier_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load classifier: {e}")

    try:
        with open(label_encoder_path, "rb") as f:
            label_encoder = pickle.load(f)
        print(f"[INFO] Loaded label encoder from {label_encoder_path}")
    except Exception as e:
        print(f"[WARNING] Failed to load label encoder: {e} — Proceeding without it")

    return vectorizer, classifier, label_encoder

# Standardize column names
def standardize_columns(df):
    """Standardize column names to handle variations."""
    column_mapping = {
        'debit': 'Debit', 'DEBIT': 'Debit', 'withdrawal': 'Debit',
        'credit': 'Credit', 'CREDIT': 'Credit', 'deposit': 'Credit',
        'balance': 'Balance', 'BALANCE': 'Balance',
        'date': 'Date', 'DATE': 'Date', 'transaction date': 'Date',
        'description': 'Description', 'DESCRIPTION': 'Description', 'particulars': 'Description',
        'category': 'Category', 'CATEGORY': 'Category'
    }
    df.columns = [column_mapping.get(col.lower(), col) for col in df.columns]
    return df

# Preprocess description
def preprocess_description(description):
    """Clean transaction description for ML model."""
    if not isinstance(description, str):
        return ""
    description = description.lower()
    description = re.sub(r"[a-zA-Z0-9]{10,}", "", description)  # Remove long codes
    description = re.sub(r"(txnid|utr|branch|upi)\W.*", "", description)  # Remove transaction metadata
    description = re.sub(r"\[.*?\]", "", description)  # Remove bracketed categories
    description = re.sub(r"[^a-z\s]", "", description)  # Keep letters and spaces
    description = re.sub(r"\s+", " ", description).strip()
    return description or "unknown"

# Infer date formats
def infer_date(date_str):
    """Attempt to infer date from string, handling common formats."""
    if pd.isna(date_str) or not str(date_str).strip():
        return pd.NaT
    date_str = str(date_str).strip()
    formats = [
        "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d",
        "%d %b %Y", "%d-%b-%Y", "%d %B %Y", "%Y%m%d",
        "%d%m%Y", "%b %d, %Y", "%B %d, %Y"
    ]
    for fmt in formats:
        try:
            return pd.to_datetime(date_str, format=fmt, errors="coerce")
        except:
            continue
    try:
        return pd.to_datetime(date_str, errors="coerce")
    except:
        return pd.NaT

# Detect recurring transactions
def detect_recurring_transactions(df, keywords):
    """Identify recurring transactions based on description and amount."""
    desc_amount = df[["Description", "Debit"]].dropna()
    counts = Counter(desc_amount.apply(lambda x: f"{x['Description']}_{x['Debit']}", axis=1))
    recurring = {k: v for k, v in counts.items() if v > 1}
    return df["Description"].apply(
        lambda x: any(kw in str(x).lower() for kw in keywords) and 
        f"{x}_{df.loc[df['Description'] == x, 'Debit'].iloc[0]}" in recurring
    )

# Rule-based categorization
def rule_based_categorize(description):
    """Categorize using predefined rules."""
    doc = nlp(description.lower()) if nlp else None
    for category, patterns in category_rules.items():
        for pattern in patterns:
            if re.search(pattern, description.lower()):
                return category
    return "Other"

# Categorize a single transaction
def categorize_description(row, vectorizer=None, classifier=None, label_encoder=None):
    """Categorize a single transaction using ML, rules, or NLP."""
    desc = str(row["Description"]).lower().strip()
    if not desc:
        return "Uncategorized"

    # ML-based categorization
    if vectorizer and classifier and label_encoder:
        try:
            features = vectorizer.transform([preprocess_description(desc)]).toarray()
            prob = classifier.predict_proba(features)[0]
            prediction = label_encoder.inverse_transform([classifier.predict(features)[0]])[0]
            if max(prob) > 0.80:  # Confidence threshold
                return prediction
        except Exception as e:
            print(f"[WARNING] ML categorization failed: {e}")

    # Rule-based matching
    for category, patterns in category_rules.items():
        for pattern in patterns:
            if re.search(pattern, desc, re.IGNORECASE):
                return category

    # NLP fallback
    if nlp:
        try:
            doc = nlp(desc)
            for ent in doc.ents:
                if ent.label_ in ["ORG", "PERSON"]:
                    if any(k in desc for k in ["transfer", "deposit", "sip", "fd", "rd", "ppf", "nps"]):
                        return "Savings"
                    if any(k in desc for k in ["emi", "rent", "loan", "bill", "insurance", "subscription"]):
                        return "Fixed Expenses"
                    if any(k in desc for k in ["shopping", "dining", "travel", "restaurant", "amazon", "zomato", "swiggy", "flipkart", "paytm", "phonepe"]):
                        return "Discretionary Expenses"
            for token in doc:
                if token.dep_ == "ROOT" and token.lemma_ in ["pay", "transfer", "deposit"]:
                    if any(k in desc for k in ["savings", "fd", "rd", "sip"]):
                        return "Savings"
                    if any(k in desc for k in ["rent", "emi", "bill"]):
                        return "Fixed Expenses"
        except Exception as e:
            print(f"[WARNING] NLP processing failed: {e}")

    # Amount-based and recurrence checks
    amount = row["Debit"] if row["Debit"] > 0 else row["Credit"]
    is_debit = row["Debit"] > 0
    date = infer_date(row["Date"])
    
    if is_debit and amount > 10000 and any(k in desc for k in ["cash", "atm", "upi.*cash"]):
        return "Red Flags"
    
    if is_debit and pd.notna(date) and date.day >= 25:
        emi_keywords = ["emi", "rent", "bill", "subscription", "loan"]
        if any(k in desc for k in emi_keywords):
            df_temp = pd.DataFrame([row])
            if detect_recurring_transactions(df_temp, emi_keywords).iloc[0]:
                return "Fixed Expenses"

    # Fuzzy matching for merchants
    merchant_keywords = ["zomato", "swiggy", "amazon", "flipkart", "paytm", "phonepe"]
    if any(fuzz.partial_ratio(desc, kw) > 85 for kw in merchant_keywords):
        return "Discretionary Expenses"

    return "Uncategorized"\
    
def load_auto_overrides():
    """Load automatic overrides from CSV."""
    overrides = {}
    if os.path.exists(AUTO_OVERRIDE_FILE):
        try:
            df = pd.read_csv(AUTO_OVERRIDE_FILE)
            if {"Description", "Category"}.issubset(df.columns):
                overrides = dict(zip(df["Description"].str.lower(), df["Category"]))
                print(f"[INFO] Loaded {len(overrides)} automatic overrides from {AUTO_OVERRIDE_FILE}")
        except Exception as e:
            print(f"[WARNING] Could not load automatic overrides: {e}")
    return overrides

def update_auto_overrides(df, threshold=3):
    """
    Add recurring uncategorized or low-confidence transactions to automatic overrides.
    
    Args:
        df (pd.DataFrame): DataFrame after categorization.
        threshold (int): Minimum recurring count to trigger override addition.
    """
    # Filter uncategorized or low confidence transactions
    problematic = df[(df["Category"] == "Uncategorized") | (df["Confidence"] < 0.7)]
    if problematic.empty:
        return

    # Count frequency of descriptions
    desc_counts = problematic["Description"].str.lower().value_counts()

    # Load existing overrides
    existing_overrides = load_auto_overrides()

    new_overrides = {}
    for desc, count in desc_counts.items():
        if count >= threshold and desc not in existing_overrides:
            # Assign "Other" category by default; can be customized
            new_overrides[desc] = "Other"

    if not new_overrides:
        return

    # Append new overrides to CSV
    new_df = pd.DataFrame(list(new_overrides.items()), columns=["Description", "Category"])
    if os.path.exists(AUTO_OVERRIDE_FILE):
        new_df.to_csv(AUTO_OVERRIDE_FILE, mode="a", header=False, index=False)
    else:
        new_df.to_csv(AUTO_OVERRIDE_FILE, index=False)
    
    print(f"[INFO] Added {len(new_overrides)} new automatic overrides to {AUTO_OVERRIDE_FILE}")

#categorize transactions in CSV
def categorize_transactions(input_file, output_file, vectorizer=None, classifier=None, label_encoder=None, override_file="overrides.csv"):
    """
    Categorize transactions in the input CSV with automated overrides and save to output CSV.
    """
    # Read input CSV
    try:
        df = pd.read_csv(input_file)
        print(f"[DEBUG] Loaded input file: {input_file}, columns: {df.columns}")
    except Exception as e:
        print(f"[ERROR] Failed to load input file {input_file}: {e}")
        return pd.DataFrame()

    # Standardize columns
    df = standardize_columns(df)

    # Clean numeric columns
    for col in ["Debit", "Credit", "Balance"]:
        if col in df.columns:
            df[col] = df[col].replace({',': '', '₹': '', 'INR': ''}, regex=True)
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Validate required columns
    required_cols = {"Date", "Description", "Debit", "Credit", "Balance"}
    if not required_cols.issubset(df.columns):
        print(f"[ERROR] Missing required columns: {required_cols - set(df.columns)}")
        return pd.DataFrame()

    # Ensure Category & Confidence columns exist
    if "Category" not in df.columns:
        df["Category"] = "Uncategorized"
    if "Confidence" not in df.columns:
        df["Confidence"] = 0.0

    # Rule-based overrides
    rule_based_overrides = {
        r"salary|wages|payroll": "Income",
        r"rent|mortgage|utility|electricity|water|gas|internet": "Fixed Expenses",
        r"amazon|flipkart|shopping|restaurant|entertainment|movie": "Discretionary Expenses",
        r"savings|investment|fixed deposit|mutual fund": "Savings",
        r"overdraft|insufficient funds|bounce|penalty|late fee": "Red Flags"
    }
    for pattern, category in rule_based_overrides.items():
        mask = df["Description"].str.lower().str.contains(pattern, case=False, na=False)
        df.loc[mask, "Category"] = category
        df.loc[mask, "Confidence"] = 1.0
        print(f"[DEBUG] Applied rule-based override: {pattern} -> {category} for {mask.sum()} transactions")

    # Apply automatic overrides
    overrides = load_auto_overrides()
    if overrides:
        override_mask = df["Description"].str.lower().isin(overrides)
        df.loc[override_mask, "Category"] = df.loc[override_mask, "Description"].str.lower().map(overrides)
        df.loc[override_mask, "Confidence"] = 1.0
        print(f"[DEBUG] Applied automatic overrides to {override_mask.sum()} transactions")

    # ML classification for remaining uncategorized
    uncategorized_idx = df.index[df["Category"] == "Uncategorized"]
    if len(uncategorized_idx) > 0 and vectorizer is not None and classifier is not None:
        try:
            X = vectorizer.transform(df.loc[uncategorized_idx, "Description"].fillna(""))
            expected = getattr(classifier, "n_features_in_", None)

            # Align features if needed
            X = X.toarray()
            if expected is not None:
                if X.shape[1] < expected:
                    X = np.pad(X, ((0, 0), (0, expected - X.shape[1])), mode="constant")
                elif X.shape[1] > expected:
                    X = X[:, :expected]

            predicted_encoded = classifier.predict(X)
            predicted_categories = (
                label_encoder.inverse_transform(predicted_encoded)
                if label_encoder is not None
                else predicted_encoded
            )
            confidence_scores = (
                classifier.predict_proba(X).max(axis=1)
                if hasattr(classifier, "predict_proba")
                else np.ones(len(predicted_categories)) * 0.5
            )

            df.loc[uncategorized_idx, "Category"] = predicted_categories
            df.loc[uncategorized_idx, "Confidence"] = confidence_scores
            print(f"[DEBUG] Categorized {len(uncategorized_idx)} transactions using classifier")
        except Exception as e:
            print(f"[WARNING] Failed to categorize transactions: {e}")

    # Validate categories
    valid_categories = ["Income", "Fixed Expenses", "Discretionary Expenses", "Savings", "Red Flags", "Uncategorized"]
    df["Category"] = df["Category"].apply(lambda x: x if x in valid_categories else "Uncategorized")

    # Save results
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"[DEBUG] Saved categorized transactions to {output_file}")
    except Exception as e:
        print(f"[ERROR] Failed to save categorized file {output_file}: {e}")

    # Log low-confidence and uncategorized
    low_confidence = df[df["Confidence"] < 0.7]
    uncategorized_count = len(df[df["Category"] == "Uncategorized"])
    if uncategorized_count > 0 or not low_confidence.empty:
        with open("outputs/uncategorized.log", "a") as f:
            f.write(f"File: {input_file}, Timestamp: {datetime.now()}, Uncategorized: {uncategorized_count}, Low Confidence: {len(low_confidence)}\n")
        if not low_confidence.empty:
            low_confidence[["Description", "Category", "Confidence"]].to_csv(
                "outputs/low_confidence.csv", mode="a", header=not os.path.exists("outputs/low_confidence.csv"), index=False
            )

    # Update auto overrides
    update_auto_overrides(df)
    return df


def refine_uncategorized(df):
    for idx, row in df[df["Category"] == "Other"].iterrows():
        if "UPI" in row["Description"] and row["Debit"] > 5000:
            df.loc[idx, "Category"] = "Discretionary Expenses"
        elif "UPI" in row["Description"] and row["Debit"] < 1000:
            df.loc[idx, "Category"] = "Fixed Expenses"
    return df

def suggest_dynamic_emi(income_stability, surplus, proposed_emi):
    if not income_stability and surplus >= proposed_emi:
        return proposed_emi * 0.7, "Step-up EMI recommended"
    return proposed_emi, "Standard EMI"

def early_warning(df, metrics_df):
    if metrics_df["Red Flag Count"].iloc[0] > 2 or metrics_df["Net Surplus"].iloc[0] < 0:
        return "Warning: High risk of default"
    return "No immediate concerns"

# Process all CSVs in a folder
def process_folder(input_folder, output_folder, vectorizer=None, classifier=None, label_encoder=None):
    """Process all CSVs in the input folder."""
    os.makedirs(output_folder, exist_ok=True)
    processed_files = []
    for file in os.listdir(input_folder):
        if file.endswith(".csv"):
            input_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, file)
            try:
                categorize_transactions(input_path, output_path, vectorizer, classifier, label_encoder)
                processed_files.append(file)
            except Exception as e:
                print(f"⚠️ Failed to process {file}: {e}")
                os.makedirs("outputs", exist_ok=True)
                with open("outputs/extraction_errors.log", "a") as f:
                    f.write(f"{datetime.now()}: Failed to process {file}: {e}\n")
    return processed_files

# CLI use
if __name__ == "__main__":
    vectorizer, classifier, label_encoder = load_models()
    input_dir = "cleaned_csvs"
    output_dir = "categorized_csvs"
    process_folder(input_dir, output_dir, vectorizer, classifier, label_encoder)
