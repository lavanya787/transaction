import pandas as pd
import os
from datetime import datetime
import numpy as np
from collections import Counter
from sklearn.cluster import DBSCAN
from sklearn.impute import SimpleImputer
from dateutil.parser import parse as parse_date
import re
from typing import Dict, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)

def is_classification_task(y_true):
    """
    Automatically determine if the task is classification.
    Rules:
    - If y contains only integers and number of unique values is small relative to length → classification
    - If dtype is object, bool, or category → classification
    """
    y_series = pd.Series(y_true)
    if y_series.dtype.name in ["object", "category", "bool"]:
        return True
    # Check if it's all integers and discrete labels
    unique_vals = y_series.unique()
    if np.issubdtype(y_series.dtype, np.integer) and len(unique_vals) < max(20, len(y_series) * 0.05):
        return True
    return False


def calculate_classification_metrics(y_true, y_pred, y_prob=None, average='weighted') -> Dict[str, Any]:
    """
    Calculate common classification metrics.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        y_prob (array-like, optional): Predicted probabilities for positive class (for ROC AUC).
        average (str): Averaging method for multi-class metrics.

    Returns:
        dict: Dictionary containing classification metrics.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0)    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob, multi_class='ovr')
        except ValueError:
            metrics["roc_auc"] = None
    
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    metrics["classification_report"] = classification_report(y_true, y_pred, zero_division=0)

    return metrics


def calculate_regression_metrics(y_true, y_pred) -> Dict[str, Any]:
    """
    Calculate common regression metrics.

    Args:
        y_true (array-like): True values.
        y_pred (array-like): Predicted values.

    Returns:
        dict: Dictionary containing regression metrics.
    """
    metrics = {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2_score": r2_score(y_true, y_pred)
    }
    return metrics

def calculate_metrics_auto(y_true, y_pred, y_proba=None):
    """
    Automatically detect if task is classification or regression and compute metrics.
    """
    if is_classification_task(y_true):
        return {
            "task_type": "classification",
            **calculate_classification_metrics(y_true, y_pred, y_proba)
        }
    else:
        return {
            "task_type": "regression",
            **calculate_regression_metrics(y_true, y_pred)
        }
def infer_date(date_str, file_name="unknown"):
    """Attempt to infer date from string, handling common and noisy formats."""
    if pd.isna(date_str) or not str(date_str).strip():
        with open("outputs/invalid_dates.log", "a") as f:
            f.write(f"File: {file_name}, Empty or NaN date: {date_str}, using default 2025-01-01\n")
        return pd.to_datetime("2025-01-01")
    
    date_str = str(date_str).strip()
    # Clean common OCR errors (e.g., 'O' instead of '0', 'I' instead of '1')
    date_str = re.sub(r"[Oo]", "0", date_str)
    date_str = re.sub(r"[IiLl]", "1", date_str)
    date_str = re.sub(r"[^0-9a-zA-Z\s/-:]", "", date_str)  # Remove special chars
    
    # Handle invalid months (e.g., 31/13/2025 -> 31/12/2025)
    try:
        parts = re.split(r"[-/:\s]", date_str)
        if len(parts) >= 3 and parts[1].isdigit() and int(parts[1]) > 12:
            parts[1] = "12"
            date_str = "/".join(parts[:3])
    except:
        pass

    formats = [
        "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d",
        "%d %b %Y", "%d-%b-%Y", "%d %B %Y", "%Y%m%d",
        "%d%m%Y", "%b %d, %Y", "%B %d, %Y", "%d.%m.%Y",
        "%Y.%m.%d", "%m/%d/%Y", "%m-%d-%Y", "%d %b %y",
        "%d-%b-%y", "%Y %b %d", "%Y %B %d", "%d-%b-%y %H:%M",
        "%Y/%m/%d %H:%M:%S", "%d/%m/%y %H:%M", "%d-%m-%y %H:%M",
        "%Y-%m-%d %H:%M:%S", "%d %b %Y %H:%M"
    ]
    for fmt in formats:
        try:
            parsed = pd.to_datetime(date_str, format=fmt, errors="coerce")
            if pd.notna(parsed):
                return parsed
        except:
            continue
    
    try:
        parsed = parse_date(date_str, fuzzy=True, dayfirst=True)
        if pd.notna(parsed):
            return pd.to_datetime(parsed)
        with open("outputs/invalid_dates.log", "a") as f:
            f.write(f"File: {file_name}, Failed to parse date: {date_str}, using default 2025-01-01\n")
        return pd.to_datetime("2025-01-01")
    except:
        with open("outputs/invalid_dates.log", "a") as f:
            f.write(f"File: {file_name}, Failed to parse date: {date_str}, using default 2025-01-01\n")
        return pd.to_datetime("2025-01-01")

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

def detect_recurring_transactions(group, keywords, amount_col="Debit"):
    """Detect recurring transactions using clustering and keyword matching."""
    if group.empty:
        print(f"[WARNING] Empty group in detect_recurring_transactions")
        return pd.Series([False] * len(group), index=group.index)

    # Drop rows with missing Description or amount_col
    desc_amount = group[["Description", amount_col]].dropna()
    if desc_amount.empty:
        print(f"[WARNING] No valid {amount_col} transactions for recurrence detection")
        return pd.Series([False] * len(group), index=group.index)

    # Impute missing values
    imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    # Ensure dates aligned to group's index
    dates = group["Date"].apply(infer_date)
    amounts = imputer.fit_transform(desc_amount[[amount_col]]).flatten()

    # Feature for clustering: normalized amount and day of month
    features = np.column_stack([
        amounts / (np.std(amounts) + 1e-6),
        dates.loc[desc_amount.index].dt.day.fillna(15) / 31
    ])

    # DBSCAN clustering
    try:
        clustering = DBSCAN(eps=0.5, min_samples=2).fit(features)
        labels = clustering.labels_
    except Exception as e:
        print(f"[WARNING] DBSCAN failed: {e}")
        labels = np.array([-1] * len(desc_amount))

    # Identify recurring transactions
    desc_counts = Counter(desc_amount["Description"])
    recurring = pd.Series(False, index=group.index, dtype=bool)

    # Iterate using labels/indexes so we don't mix positional ints with labels
    for idx in group.index:
        desc = group.at[idx, "Description"]
        if pd.isna(desc):
            continue
        # find first match in desc_amount index (if any)
        matches = desc_amount.index[desc_amount["Description"] == desc]
        if matches.empty:
            # no matching description in desc_amount (possibly because of dropna earlier)
            recurring.at[idx] = False
            continue
        first_match_idx = matches[0]
        pos_in_labels = list(desc_amount.index).index(first_match_idx)
        recurring.at[idx] = (
            any(kw in str(desc).lower() for kw in keywords) and
            (desc_counts[desc] > 1 or (pos_in_labels < len(labels) and labels[pos_in_labels] != -1))
        )

    return recurring

def detect_anomalies(series):
    """Identify outliers using z-score."""
    z_scores = np.abs((series - series.mean()) / (series.std() + 1e-6))
    return z_scores > 3

def clean_currency_columns(df):
    for col in df.select_dtypes(include='object'):
        # Remove currency symbols & commas, convert to float where possible
        df[col] = df[col].replace(r"[₹$,]", "", regex=True)
        try:
            df[col] = df[col].astype(float)
        except ValueError:
            pass  # leave non-numeric text as-is
    return df

def calculate_metrics(df, output_file):
    """Calculate financial metrics with advanced logic."""
    # Standardize columns
    df = standardize_columns(df)
    print(f"[DEBUG] Columns after standardization: {list(df.columns)}")
    
    # Clean all currency-like columns (₹, $, commas, etc.)
    df = clean_currency_columns(df)

    # Validate and clean columns
    required_columns = ["Date", "Description", "Debit", "Credit", "Balance", "Category"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"[WARNING] Missing columns: {missing_columns}. Filling with defaults.")
        for col in missing_columns:
            if col in ["Debit", "Credit", "Balance"]:
                df[col] = 0.0
            elif col == "Date":
                df[col] = pd.NaT
            elif col == "Description":
                df[col] = ""
            elif col == "Category":
                df[col] = "Uncategorized"

    # Clean numeric columns
    for col in ["Debit", "Credit", "Balance"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Convert and clean dates
    original_len = len(df)
    df["Date"] = df["Date"].apply(lambda x: infer_date(x, file_name=output_file))
    df = df.dropna(subset=["Date"])
    if len(df) < original_len:
        print(f"[WARNING] Dropped {original_len - len(df)} rows with invalid dates")
        os.makedirs("outputs", exist_ok=True)

        with open("outputs/invalid_dates.log", "a") as f:
            f.write(f"File: {output_file}, Dropped {original_len - len(df)} rows with invalid dates\n")

    # Return empty metrics if DataFrame is empty
    if df.empty:
        print("[WARNING] Empty DataFrame after date cleaning, returning empty metrics")
        return pd.DataFrame([{
            "Average Monthly Income": 0.0,
            "Average Monthly Expenses": 0.0,
            "Average Monthly EMI": 0.0,
            "Net Surplus": 0.0,
            "DTI Ratio": 0.0,
            "Savings Rate": 0.0,
            "Red Flag Count": 0,
            "Credit Utilization": 0.0,
            "Discretionary Expenses": 0.0
        }])

    # Group by month
    df["Month"] = df["Date"].dt.to_period("M")
    monthly_groups = df.groupby("Month")
    metrics = []
    total_recurring_amount = 0.0

    for month, group in monthly_groups:
        print(f"[DEBUG] Processing month: {month}")
        # Calculate days in month for partial month adjustment
        month_start = group["Date"].min()
        month_end = group["Date"].max()
        days_in_month = (month_end - month_start).days + 1 if pd.notna(month_start) else 30
        month_weight = min(days_in_month / 30, 1.0)

        # Seasonal adjustment
        seasonal_factor = 1.2 if month.month in [10, 11, 12] else 1.0

        # Filter categories
        income = group[group["Category"] == "Income"]["Credit"]
        fixed_expenses = group[group["Category"] == "Fixed Expenses"]["Debit"]
        discretionary_expenses = group[group["Category"] == "Discretionary Expenses"]["Debit"]
        savings_credits = group[group["Category"] == "Savings"]["Credit"]
        savings_debits = group[group["Category"] == "Savings"]["Debit"]
        red_flags = group[group["Category"] == "Red Flags"]

        # Anomaly detection
        income_outliers = detect_anomalies(income)
        fixed_outliers = detect_anomalies(fixed_expenses)
        discretionary_outliers = detect_anomalies(discretionary_expenses)

        # Remove outliers
        income_clean = income[~income_outliers].sum()
        fixed_expenses_clean = fixed_expenses[~fixed_outliers].sum()
        discretionary_expenses_clean = discretionary_expenses[~discretionary_outliers].sum()

        # Net savings
        net_savings = savings_credits.sum() - savings_debits.sum()

        # Weight expenses by recency
        group["Recency"] = (datetime.now() - group["Date"]).dt.days
        group["Weight"] = 1 / (group["Recency"] + 1)
        weighted_fixed = group[group["Category"] == "Fixed Expenses"].apply(
            lambda x: x["Debit"] * x["Weight"] / fixed_expenses_clean if fixed_expenses_clean > 0 else 0, axis=1
        ).sum()
        weighted_discretionary = group[group["Category"] == "Discretionary Expenses"].apply(
            lambda x: x["Debit"] * x["Weight"] / discretionary_expenses_clean if discretionary_expenses_clean > 0 else 0, axis=1
        ).sum()

        # Identify recurring debits
        recurring_keywords = ["emi", "loan", "mortgage", "rent", "bill", "subscription", "insurance", "payment", "utility", "premium"]
        recurring_transactions = group[
            (group["Category"].isin(["Fixed Expenses", "Discretionary Expenses"])) &
            (group["Description"].str.lower().str.contains("|".join(recurring_keywords), na=False))
        ]

        if not recurring_transactions.empty:
            recurring_mask = detect_recurring_transactions(recurring_transactions, recurring_keywords, amount_col="Debit")
            recurring_amount = recurring_transactions.loc[recurring_mask, "Debit"].sum() if recurring_mask.any() else 0.0
        else:
            recurring_amount = 0.0

        total_recurring_amount += recurring_amount

        # Calculate metrics
        total_expenses = (weighted_fixed + weighted_discretionary) * seasonal_factor
        net_surplus = income_clean - total_expenses
        dti = (recurring_amount / income_clean * 100) if income_clean > 0 else 0.0
        savings_rate = (net_savings / income_clean * 100) if income_clean > 0 else 0.0

        metrics.append({
            "Month": month.to_timestamp().strftime("%Y-%m"),
            "Average Monthly Income": income_clean / month_weight if month_weight > 0 else income_clean,
            "Average Monthly Expenses": total_expenses / month_weight if month_weight > 0 else total_expenses,
            "Net Surplus": net_surplus,
            "Discretionary Expenses": discretionary_expenses_clean / month_weight if month_weight > 0 else discretionary_expenses_clean,
            "Debt-to-Income (%)": round(dti, 2),
            "Savings Rate (%)": round(savings_rate, 2),
            "Red Flag Count": len(red_flags),
            "Outliers Detected": int(sum(income_outliers) + sum(fixed_outliers) + sum(discretionary_outliers))
        })

    metrics_df = pd.DataFrame(metrics)
    if metrics_df.empty:
        print("[WARNING] No metrics calculated, returning default metrics")
        return pd.DataFrame([{
            "Average Monthly Income": 0.0, "Average Monthly Expenses": 0.0,
            "Average Monthly EMI": 0.0, "Net Surplus": 0.0, "DTI Ratio": 0.0,
            "Savings Rate": 0.0, "Red Flag Count": 0, "Credit Utilization": 0.0,
            "Discretionary Expenses": 0.0
        }])

    # Aggregate metrics
    aggregated = {
        "Average Monthly Income": metrics_df["Average Monthly Income"].mean(),
        "Average Monthly Expenses": metrics_df["Average Monthly Expenses"].mean(),
        "Average Monthly EMI": total_recurring_amount / max(1, len(metrics_df)),  # average recurring per month as EMI proxy
        "Net Surplus": metrics_df["Net Surplus"].sum(),
        "DTI Ratio": metrics_df["Debt-to-Income (%)"].mean(),
        "Savings Rate": metrics_df["Savings Rate (%)"].mean(),
        "Red Flag Count": metrics_df["Red Flag Count"].sum(),
        "Credit Utilization": 0.0,
        "Discretionary Expenses": metrics_df["Discretionary Expenses"].mean(),
        "Income Stability": check_salary_stability(df)
    }

    aggregated_df = pd.DataFrame([aggregated])

    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    aggregated_df.to_csv(output_file, index=False)
    print(f"✅ Metrics saved to {output_file}")

    return aggregated_df
def check_salary_stability(df):
    income_df = df[df["Category"] == "Income"]
    monthly_counts = income_df.groupby(income_df["Date"].dt.to_period("M")).size()
    return monthly_counts.std() / monthly_counts.mean() < 0.2  # Low variance indicates stability
def calculate_savings_metrics(df):
    savings_df = df[df["Category"] == "Savings"]
    total_savings = savings_df["Credit"].sum()
    total_income = df[df["Category"] == "Income"]["Credit"].sum()
    savings_rate = (total_savings / total_income) * 100 if total_income > 0 else 0
    emergency_fund_months = total_savings / df[df["Category"] == "Fixed Expenses"]["Debit"].mean() if df["Debit"].mean() > 0 else 0
    return savings_rate, emergency_fund_months

def score_and_decide(metrics_df, cibil_score, loan_purpose, proposed_emi=0):
    if metrics_df.empty:
        return {"Total Score": 0, "Risk Level": "High", "Action": "Reject", "Reason": "No valid metrics"}
    
    avg_income = metrics_df["Average Monthly Income"].iloc[0] if "Average Monthly Income" in metrics_df.columns else 0
    avg_expenses = metrics_df["Average Monthly Expenses"].iloc[0] if "Average Monthly Expenses" in metrics_df.columns else 0
    avg_emi = metrics_df["Average Monthly EMI"].iloc[0] if "Average Monthly EMI" in metrics_df.columns else 0
    net_surplus = metrics_df["Net Surplus"].iloc[0] if "Net Surplus" in metrics_df.columns else 0
    dti_ratio = metrics_df["DTI Ratio"].iloc[0] if "DTI Ratio" in metrics_df.columns else 0
    savings_rate = metrics_df["Savings Rate"].iloc[0] if "Savings Rate" in metrics_df.columns else 0
    red_flag_count = metrics_df["Red Flag Count"].iloc[0] if "Red Flag Count" in metrics_df.columns else 0
    discretionary_expenses = metrics_df["Discretionary Expenses"].iloc[0] if "Discretionary Expenses" in metrics_df.columns else 0
    income_stability = metrics_df["Income Stability"].iloc[0] if "Income Stability" in metrics_df.columns else False

    # Scoring (per requirements: 25% income stability, 25% surplus, 20% DTI, 15% expense discipline, 15% red flags)
    income_stability_score = 25 if income_stability else 10
    surplus_score = 25 if net_surplus >= 2 * proposed_emi else (15 if net_surplus >= proposed_emi else 5)
    dti_score = 20 if dti_ratio < 30 else (12 if dti_ratio < 50 else 5)
    expense_discipline_score = 15 if discretionary_expenses < 0.2 * avg_income else (8 if discretionary_expenses < 0.4 * avg_income else 2)
    red_flag_score = 15 if red_flag_count == 0 else (8 if red_flag_count <= 2 else 2)

    # Total score
    total_score = income_stability_score + surplus_score + dti_score + expense_discipline_score + red_flag_score
    risk_level = "Low" if total_score >= 80 else "Moderate" if total_score >= 60 else "High"

    # Loan purpose impact
    purpose_weights = {"Education": 1.1, "Business": 1.0, "Home Improvement": 1.2, "Luxury": 0.8, "Vacation": 0.7, "Neutral": 1.0}
    purpose_factor = purpose_weights.get(loan_purpose, 1.0)
    adjusted_score = total_score * purpose_factor
    adjusted_score = max(0, min(100, adjusted_score))

    # Decision logic
    new_dti = ((avg_emi + proposed_emi) / avg_income * 100) if avg_income > 0 else 100
    if adjusted_score >= 80 and cibil_score >= 750 and new_dti <= 40:
        action = "Approve with standard terms"
        reason = "High score, good CIBIL, and low DTI"
    elif adjusted_score >= 60 and cibil_score >= 600 and new_dti <= 50:
        action = "Approve with caution"
        reason = "Moderate score, acceptable CIBIL, and DTI"
    else:
        action = "Reject"
        reason = f"Low score ({adjusted_score:.1f}), high DTI ({new_dti:.1f}%), or low CIBIL ({cibil_score})"

    # Adjust for low CIBIL
    if cibil_score < 600 and action == "Reject":
        adjusted_emi = proposed_emi * 0.5  # Reduce EMI by 50%
        new_dti_adjusted = ((avg_emi + adjusted_emi) / avg_income * 100) if avg_income > 0 else 100
        if new_dti_adjusted <= 40:
            action = "Approve with higher interest (small loan)"
            reason = f"Low CIBIL ({cibil_score}) with adjusted EMI: ₹{adjusted_emi:,.2f}, DTI: {new_dti_adjusted:.1f}%"
            adjusted_score = min(70, adjusted_score + 5)

    return {
        "Total Score": round(adjusted_score, 1),
        "Risk Level": risk_level,
        "Action": action,
        "Reason": reason
    }

def process_folder(input_folder, output_folder):
    """Process all CSVs in input_folder and save metrics to output_folder."""
    os.makedirs(output_folder, exist_ok=True)
    all_metrics = []
    for file in os.listdir(input_folder):
        if file.endswith(".csv"):
            input_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, file.replace(".csv", "_metrics.csv"))
            print(f"[DEBUG] Processing file: {file}")
            try:
                df = pd.read_csv(input_path)
                print(f"[DEBUG] Initial columns: {list(df.columns)}")
                metrics_df = calculate_metrics(df, output_path)
                if not metrics_df.empty:
                    all_metrics.append((file, metrics_df))
                    print(f"✅ Metrics calculated for {file}, saved to {output_path}")
                else:
                    print(f"⚠️ No valid data in {file}, skipping")
            except Exception as e:
                print(f"⚠️ Failed to process {file}: {e}")
    if all_metrics:
        aggregated_dfs = [metrics for _, metrics in all_metrics]
        combined_metrics = pd.concat(aggregated_dfs, ignore_index=True)
        combined_metrics.to_csv(os.path.join(output_folder, "combined_metrics.csv"), index=False)
        print(f"✅ Combined metrics saved to {output_folder}/combined_metrics.csv")
    else:
        print("[WARNING] No metrics calculated for any files")
    return all_metrics

if __name__ == "__main__":
    # Example usage for testing
    test_df = pd.DataFrame({
        "Date": ["2025-01-01", "2025-01-02", "2025-01-03"],
        "Description": ["Salary Credit", "Rent Payment", "EMI Payment"],
        "Debit": [0, 15000, 10000],
        "Credit": [50000, 0, 0],
        "Balance": [50000, 35000, 25000],
        "Category": ["Income", "Fixed Expenses", "Fixed Expenses"]
    })
    test_output = "outputs/test_metrics.csv"
    metrics_df = calculate_metrics(test_df, test_output)
    print(metrics_df)