#preprocess_transactions.py
"""
Preprocess extracted bank transactions for categorization & analysis.
"""
import pandas as pd
import numpy as np
import re
from datetime import datetime

def make_arrow_compatible(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures DataFrame columns are Arrow-compatible for Streamlit display.
    Fixes datetime, numeric, boolean, and string types.
    """
    df = df.copy()

    # Convert datetime → string
    for col in df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
        df[col] = df[col].astype(str)

    # Explicitly handle 'Date' column
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # Convert monetary/metric columns to float64
    money_cols = [
        "Debit", "Credit", "Balance", "Amount", "Value",
        "Average Monthly Income", "Average Monthly Expenses", "Net Surplus",
        "DTI Ratio", "Discretionary Expenses", "Savings Rate", "Cumulative Savings",
        "Average Monthly EMI", "Credit Utilization", "Average Monthly Balance",
        "Cash Withdrawals"
    ]
    for col in money_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r"[₹,$£,%]", "", regex=True)  # also remove % symbols
                .str.strip()
            )
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype("float64")
            except Exception as e:
                print(f"[ERROR] Failed to convert {col} to numeric: {e}")
                df[col] = 0.0

    # Convert integer columns
    int_cols = ["Year", "Month", "Red Flag Count", "Number of Open Credit Accounts"]
    for col in int_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
            except Exception as e:
                print(f"[ERROR] Failed to convert {col} to int64: {e}")
                df[col] = 0

    # Convert booleans → int (Arrow safe)
    for col in df.select_dtypes(include=["bool"]).columns:
        df[col] = df[col].astype(int)

    # Convert all remaining object/string → plain str
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).replace(["nan", "NaN", "<NA>", ""], "Unknown")

    print(f"[DEBUG] Arrow-compatible df types: {df.dtypes}")
    return df


def detect_recurring_transactions(df: pd.DataFrame, min_occurrences=3, date_window=35) -> set:
    """
    Detect recurring transactions based on similar description & periodic intervals.

    Args:
        df: DataFrame of transactions
        min_occurrences: Minimum number of times a transaction must appear to be considered recurring
        date_window: Days allowed between occurrences for it to be considered periodic

    Returns:
        set: Descriptions considered recurring
    """
    if "Description_clean" not in df.columns or "Date" not in df.columns:
        print("[WARNING] Missing required columns for recurring transaction detection")
        return set()

    recurring_descriptions = set()
    # Normalize descriptions for grouping (remove variable numbers)
    df["Description_normalized"] = df["Description_clean"].str.replace(r"\d+", "", regex=True).str.strip()
    grouped = df.groupby("Description_normalized")

    for desc, group in grouped:
        if len(group) >= min_occurrences:
            try:
                sorted_dates = pd.to_datetime(group["Date"]).sort_values().dropna()
                if len(sorted_dates) < 2:
                    continue
                gaps = [(sorted_dates.iloc[i + 1] - sorted_dates.iloc[i]).days 
                        for i in range(len(sorted_dates) - 1)]
                if gaps and all(abs(g - np.mean(gaps)) <= date_window for g in gaps):
                    recurring_descriptions.update(group["Description_clean"].unique())
            except Exception as e:
                print(f"[ERROR] Failed to detect recurring transactions for {desc}: {e}")
    
    return recurring_descriptions

def preprocess_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize extracted bank transactions DataFrame.
    Adds derived fields for analysis and flags recurring transactions.
    """
    if df.empty:
        print("[WARNING] Empty DataFrame in preprocess_transactions")
        return pd.DataFrame(columns=["Date", "Description", "Debit", "Credit", "Balance", 
                                    "Description_clean", "Year", "Month", "Month_Name", 
                                    "DayOfWeek", "Type", "Recurring_Tag"])
    
    # --- 1. Basic cleaning ---
    df = df.copy()
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()
    column_mapping = {
        'debit': 'Debit', 'withdrawal': 'Debit',
        'credit': 'Credit', 'deposit': 'Credit',
        'balance': 'Balance',
        'date': 'Date', 'transaction date': 'Date',
        'description': 'Description', 'particulars': 'Description', 'narration': 'Description',
        'category': 'Category'
    }
    df.columns = [column_mapping.get(col, col.title()) for col in df.columns]
    
    # Ensure required columns
    required_cols = ["Date", "Description", "Debit", "Credit", "Balance"]
    for col in required_cols:
        if col not in df.columns:
            print(f"[WARNING] Missing column {col}, adding with default values")
            if col in ["Debit", "Credit", "Balance"]:
                df[col] = 0.0
            elif col == "Description":
                df[col] = "Unknown"
            elif col == "Date":
                df[col] = pd.NaT

    # Date parsing
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).reset_index(drop=True)
    if df.empty:
        print("[WARNING] All rows dropped due to invalid dates")
        return pd.DataFrame(columns=["Date", "Description", "Debit", "Credit", "Balance", 
                                    "Description_clean", "Year", "Month", "Month_Name", 
                                    "DayOfWeek", "Type", "Recurring_Tag"])

    # Ensure numeric fields
    numeric_cols = ["Debit", "Credit", "Balance", "Amount"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r"[₹,$£,]", "", regex=True)  # Handle multiple currency symbols
                .str.strip()
            )
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype("float64")
            except Exception as e:
                print(f"[ERROR] Failed to convert {col} to numeric: {e}")
                df[col] = 0.0
        else:
            df[col] = 0.0

    # Normalize description
    df["Description"] = df["Description"].fillna("Unknown").astype(str).str.strip()
    df["Description_clean"] = (
        df["Description"]
        .str.lower()
        .str.replace(r"[^a-z0-9\s]", "", regex=True)
        .str.strip()
        .replace("", "unknown")
    )

    # --- 2. Derived date fields ---
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Month_Name"] = df["Date"].dt.strftime("%B")
    df["DayOfWeek"] = df["Date"].dt.day_name()

    # --- 3. Income / Expense classification ---
    df["Type"] = np.where(df["Credit"] > 0, "Income", "Expense")

    # --- 4. Recurring transaction detection ---
    recurring = detect_recurring_transactions(df, min_occurrences=3, date_window=35)
    df["Recurring_Tag"] = df["Description_clean"].apply(lambda x: "Yes" if x in recurring else "No")

    # --- 5. Initialize Category column if missing ---
    if "Category" not in df.columns:
        print("[WARNING] Adding Category column with default value")
        df["Category"] = "Uncategorized"

    # --- 6. Ensure Arrow compatibility ---
    df = make_arrow_compatible(df)
    
    print(f"[DEBUG] Preprocessed df types: {df.dtypes}")
    return df

# ---------------------------
# Example usage (standalone run)
# ---------------------------
if __name__ == "__main__":
    from utils.extract_transactions import extract_transactions

    file_path = input("Enter extracted bank statement file path: ").strip()
    raw_df = extract_transactions(file_path)
    processed_df = preprocess_transactions(raw_df)

    print(processed_df.head())
    processed_df.to_csv("processed_transactions.csv", index=False)
    print("✅ Processed data saved to processed_transactions.csv")
