import pandas as pd
import os
import numpy as np
import re
from datetime import datetime
from collections import Counter
from sklearn.cluster import DBSCAN
from sklearn.impute import SimpleImputer
from dateutil.parser import parse as parse_date
import logging
logger = logging.getLogger(__name__)
from utils.preprocess_transactions import make_arrow_compatible

def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce schema for Arrow/Streamlit compatibility:
    - Ensure all expected columns exist
    - Convert money-like metrics to float64
    - Convert flags to int64/bool
    - Convert datetimes to string
    """
    df = df.copy()

    expected_cols = {
        "Average Monthly Income": 0.0,
        "Average Monthly Expenses": 0.0,
        "Average Monthly EMI": 0.0,
        "Net Surplus": 0.0,
        "DTI Ratio": 0.0,
        "Savings Rate": 0.0,
        "Red Flag Count": 0,
        "Credit Utilization": 0.0,
        "Discretionary Expenses": 0.0,
        "Average Monthly Balance": 0.0,
        "Cash Withdrawals": 0.0,
        "Number of Open Credit Accounts": 0,
        "Income Stability": False,
        "MonthStart": pd.NaT,
        "Value": 0.0,
    }

    for col, default in expected_cols.items():
        if col not in df.columns:
            df[col] = default

    # Ensure floats
    float_cols = [c for c, v in expected_cols.items() if isinstance(v, float)]
    for c in float_cols:
        df[c] = (
            pd.to_numeric(df[c], errors="coerce")
            .fillna(0.0)
            .astype("float64")
        )

    # Ensure ints
    int_cols = [c for c, v in expected_cols.items() if isinstance(v, int)]
    for c in int_cols:
        df[c] = (
            pd.to_numeric(df[c], errors="coerce")
            .fillna(0)
            .astype("int64")
        )

    # Ensure bools
    bool_cols = [c for c, v in expected_cols.items() if isinstance(v, bool)]
    for c in bool_cols:
        df[c] = df[c].astype(bool)

    # Handle datetime → string
    if "MonthStart" in df.columns:
        df["MonthStart"] = (
            pd.to_datetime(df["MonthStart"], errors="coerce")
            .dt.strftime("%Y-%m-%d")
            .fillna("Unknown")
        )

    # Force all object → string
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).replace(["nan", "NaN", "<NA>"], "Unknown")

    return df

def _ensure_cols(df: pd.DataFrame, cols_with_defaults: dict) -> pd.DataFrame:
    """Ensure required columns exist with safe defaults."""
    df = df.copy()
    for col, default in cols_with_defaults.items():
        if col not in df.columns:
            df[col] = default
    return df

def clean_currency_column(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """
    Clean currency-like columns by removing symbols/formatting and converting to float64.
    Handles ₹, $, £, commas, spaces, trailing CR/DR, and parentheses-for-negative.

    Example inputs it fixes:
      "₹ 12,345.67", "(1,234.50)", "5,000 CR", "2,000 DR", " -  7,890 "
    """
    df = df.copy()
    if columns is None:
        columns = [c for c in df.columns if c in ["Debit", "Credit", "Balance", "Amount"]]

    cr_pattern = re.compile(r"\bcr\b", flags=re.I)
    dr_pattern = re.compile(r"\bdr\b", flags=re.I)

    for col in columns:
        if col not in df.columns:
            continue
        try:
            s = df[col].astype(str)

            # strip currency symbols and spaces
            s = s.str.replace(r"[₹$£]", "", regex=True)
            # remove any non-digit/decimal/comma/paren/minus/space
            s = s.str.replace(r"[^0-9,\.\-\(\)\sA-Za-z]", "", regex=True).str.strip()

            # CR/DR handling: CR -> positive; DR -> negative (if not already negative)
            cr_mask = s.str.contains(cr_pattern, na=False)
            dr_mask = s.str.contains(dr_pattern, na=False)
            s = s.str.replace(cr_pattern, "", regex=True)
            s = s.str.replace(dr_pattern, "", regex=True)

            # remove spaces
            s = s.str.replace(r"\s+", "", regex=True)

            # parentheses negatives: (1234.56) -> -1234.56
            paren_mask = s.str.match(r"^\(.*\)$", na=False)
            s = s.str.replace(r"^\((.*)\)$", r"-\1", regex=True)

            # drop thousands commas
            s = s.str.replace(",", "", regex=False)

            # empty or invalid -> 0
            s = s.replace(["", "nan", "NaN", "None"], "0")

            out = pd.to_numeric(s, errors="coerce").fillna(0.0)

            # apply DR negative if needed and not already negative
            out = np.where(dr_mask & (out >= 0), -out, out)
            # (CR is positive by convention; nothing to do, but keep branch for clarity)

            df[col] = out.astype("float64")
        except Exception as e:
            logger.exception(f"[ERROR] Failed to clean column {col}: {e}")
            df[col] = df.get(col, 0)
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype("float64")
    return df

def infer_date(date_str, file_name="unknown") -> pd.Timestamp:
    """Attempt to infer date from string, handling common and noisy formats. Falls back to 2025-01-01."""
    try:
        if pd.isna(date_str) or not str(date_str).strip():
            raise ValueError("Empty date")

        s = str(date_str).strip()

        # OCR noise fixes
        s = re.sub(r"[Oo]", "0", s)
        s = re.sub(r"[IiLl]", "1", s)
        s = re.sub(r"[^0-9A-Za-z\s/\-:\.]", "", s)

        # fix impossible months (e.g., 31/13/2025)
        try:
            parts = re.split(r"[-/:\s\.]", s)
            if len(parts) >= 3 and parts[1].isdigit() and int(parts[1]) > 12:
                parts[1] = "12"
                s = "/".join(parts[:3])
        except Exception:
            pass

        # explicit formats first
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
            parsed = pd.to_datetime(s, format=fmt, errors="coerce")
            if pd.notna(parsed):
                return parsed

        # fuzzy parse
        parsed = parse_date(s, fuzzy=True, dayfirst=True)
        return pd.to_datetime(parsed)
    except Exception:
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/invalid_dates.log", "a", encoding="utf-8") as f:
            f.write(f"File: {file_name}, Failed to parse date: {date_str}, using default 2025-01-01\n")
        return pd.to_datetime("2025-01-01")

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize common column names."""
    df = df.copy()
    mapping = {
        "debit": "Debit", "withdrawal": "Debit",
        "credit": "Credit", "deposit": "Credit",
        "balance": "Balance",
        "date": "Date", "transaction date": "Date",
        "description": "Description", "particulars": "Description",
        "category": "Category",
        "amount": "Amount",
        "type": "Type",
        "recurring_tag": "Recurring_Tag",
        "description_clean": "Description_clean"
    }
    df.columns = [mapping.get(c.strip().lower(), c) for c in df.columns]
    return df

# ----------------- Transaction Analysis -----------------
def detect_anomalies(series: pd.Series) -> pd.Series:
    """Detect outliers using IQR. Returns boolean mask aligned to series index."""
    if series is None or series.empty or not np.issubdtype(series.dtype, np.number):
        return pd.Series([False] * (0 if series is None else len(series)), index=(None if series is None else series.index), dtype=bool)
    try:
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return (series < lower) | (series > upper)
    except Exception as e:
        logger.warning(f"[WARN] Anomaly detection failed: {e}")
        return pd.Series([False] * len(series), index=series.index, dtype=bool)

def check_salary_stability(df: pd.DataFrame) -> bool:
    """Check if income (credits tagged as Income) is consistent. Uses CV < 0.3."""
    if "Category" not in df.columns or "Credit" not in df.columns:
        return False
    s = pd.to_numeric(df.loc[df["Category"] == "Income", "Credit"], errors="coerce").dropna()
    if s.size < 2 or s.mean() <= 0:
        return False
    try:
        return (s.std() / (s.mean() + 1e-9)) < 0.3
    except Exception as e:
        logger.warning(f"[WARN] Salary stability check failed: {e}")
        return False

def identify_high_value_transactions(df: pd.DataFrame, amount_col="Debit", threshold=None) -> pd.Series:
    """High-value detector using IQR rule by default."""
    if amount_col not in df.columns:
        logger.warning(f"[WARN] Column {amount_col} not found for high-value detection")
        return pd.Series([False] * len(df), index=df.index)

    amounts = pd.to_numeric(df[amount_col], errors="coerce").dropna()
    if amounts.empty:
        logger.warning(f"[WARN] No valid {amount_col} data to detect high-value transactions")
        return pd.Series([False] * len(df), index=df.index)

    if threshold is None:
        q1, q3 = amounts.quantile(0.25), amounts.quantile(0.75)
        iqr = q3 - q1
        threshold = q3 + 1.5 * iqr

    mask = pd.to_numeric(df[amount_col], errors="coerce") > threshold
    logger.info(f"[DEBUG] Identified {mask.sum()} high-value transactions with threshold {threshold}")
    return mask.fillna(False)

def recurring_transactions(group: pd.DataFrame, keywords, amount_col="Debit") -> pd.Series:
    """Detect recurring transactions via light clustering + keyword rules."""
    if group is None or group.empty:
        return pd.Series([False] * (0 if group is None else len(group)), index=(None if group is None else group.index))

    cols_needed = {"Description": "", amount_col: 0.0, "Date": ""}
    group = _ensure_cols(group, cols_with_defaults=cols_needed)

    # safe views
    desc_amount = group[["Description", amount_col]].copy()
    desc_amount[amount_col] = pd.to_numeric(desc_amount[amount_col], errors="coerce")
    desc_amount = desc_amount.dropna(subset=["Description", amount_col])

    if desc_amount.empty:
        return pd.Series([False] * len(group), index=group.index)

    imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    amounts = imputer.fit_transform(desc_amount[[amount_col]]).ravel()

    dates = group["Date"].apply(infer_date)
    day_norm = dates.loc[desc_amount.index].dt.day.fillna(15) / 31.0

    # features: normalized amount and day-of-month
    std = np.std(amounts) + 1e-6
    feats = np.column_stack([amounts / std, day_norm])

    try:
        labels = DBSCAN(eps=0.5, min_samples=2).fit(feats).labels_
    except Exception as e:
        logger.warning(f"[WARN] DBSCAN failed: {e}")
        labels = np.full(len(desc_amount), -1)

    desc_counts = Counter(desc_amount["Description"])
    rec_mask = pd.Series(False, index=group.index, dtype=bool)

    da_index_list = list(desc_amount.index)
    for idx in group.index:
        desc = group.at[idx, "Description"]
        if pd.isna(desc):
            continue
        # keyword match
        kw_match = any(kw in str(desc).lower() for kw in keywords)
        # cluster or repeat detection
        da_matches = desc_amount.index[desc_amount["Description"] == desc]
        clustered = False
        if not da_matches.empty:
            first_idx = da_matches[0]
            pos = da_index_list.index(first_idx)
            clustered = (0 <= pos < len(labels)) and (labels[pos] != -1)
        rec_mask.at[idx] = kw_match and (desc_counts[desc] > 1 or clustered)

    return rec_mask

# Main Metrics
def calculate_metrics(df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """
    Calculate monthly financial metrics from (possibly imperfect) transactions.
    Produces Arrow/Streamlit-friendly dtypes.
    """
    try:
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to calculate_metrics")
            return pd.DataFrame()

        # 1) Standardize and ensure required columns exist
        df = standardize_columns(df)
        df = _ensure_cols(df, {
            "Date": "",
            "Debit": 0.0,
            "Credit": 0.0,
            "Balance": 0.0,
            "Amount": 0.0,
            "Type": "",
            "Recurring_Tag": "No",
            "Category": "Uncategorized",
            "Description": "",
            "Description_clean": ""
        })

        # 2) Clean/Coerce money-like fields -> numeric
        df = clean_currency_column(df, ["Debit", "Credit", "Balance", "Amount"])
        for c in ["Debit", "Credit", "Balance", "Amount"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

        # 3) Derive Amount when missing (Credit - Debit)
        if ("Amount" not in df.columns) or (pd.to_numeric(df["Amount"], errors="coerce").fillna(0).eq(0).all()):
            df["Amount"] = df["Credit"] - df["Debit"]

        # 4) Dates & month keys
        df["Date"] = df["Date"].apply(infer_date)  # must return pandas.Timestamp or NaT
        df["MonthStart"] = df["Date"].dt.to_period("M").dt.to_timestamp()  # timestamp[ns]
        df["YearMonth"] = df["Date"].dt.to_period("M").astype(str)         # "YYYY-MM" string

        # 5) Type inference when missing
        type_missing = df["Type"].astype(str).str.len().eq(0)
        df.loc[type_missing & (df["Credit"] > 0), "Type"] = "Income"
        df.loc[type_missing & (df["Debit"] > 0), "Type"] = "Expense"
        df["Type"] = df["Type"].astype(str).str.strip().str.title()

        metrics_rows = []
        for ym, grp in df.groupby("YearMonth", sort=True):
            # ensure numeric within the group (defensive)
            g_credit = pd.to_numeric(grp["Credit"], errors="coerce").fillna(0.0)
            g_debit  = pd.to_numeric(grp["Debit"],  errors="coerce").fillna(0.0)
            g_bal    = pd.to_numeric(grp["Balance"],errors="coerce")
            g_amt    = pd.to_numeric(grp["Amount"], errors="coerce").fillna(0.0)

            income   = g_credit.where(grp["Type"].eq("Income")).sum(min_count=1) or 0.0
            expenses = g_debit.where(grp["Type"].eq("Expense")).sum(min_count=1) or 0.0

            # Recurring EMI: tagged rows (case-insensitive "yes")
            is_rec = grp["Recurring_Tag"].astype(str).str.lower().eq("yes")
            emi = g_debit.where(is_rec).sum(min_count=1) or 0.0

            net_surplus = float(income - expenses)
            dti_ratio   = float((emi / income) * 100) if income > 1e-9 else 0.0
            savings_rate = float((net_surplus / income) * 100) if income > 1e-9 else 0.0

            # Discretionary (by Category), average balance
            discretionary = g_debit.where(grp["Category"].astype(str).str.lower().eq("discretionary")).sum(min_count=1) or 0.0
            avg_balance   = float(g_bal.mean()) if not pd.isna(g_bal.mean()) else 0.0

            # ATM/Cash withdrawals — use non-capturing group to avoid warnings
            desc_col = "Description_clean" if "Description_clean" in grp.columns else "Description"
            cash_withdrawals = pd.to_numeric(
                grp.loc[grp[desc_col].astype(str).str.contains(r"(?:atm|cash)", case=False, na=False), "Debit"],
                errors="coerce"
            ).sum(min_count=1) or 0.0

            credit_accounts = grp.loc[grp["Category"].astype(str).str.lower().eq("credit payment"), "Description"].astype(str).nunique()
            income_stability = check_salary_stability(grp)

            metrics_rows.append({
                "Month": ym,  # "YYYY-MM"
                "Average Monthly Income": float(income),
                "Average Monthly Expenses": float(expenses),
                "Average Monthly EMI": float(emi),
                "Net Surplus": net_surplus,
                "DTI Ratio": dti_ratio,
                "Savings Rate": savings_rate,
                "Red Flag Count": 0,
                "Credit Utilization": 0.0,
                "Discretionary Expenses": float(discretionary),
                "Average Monthly Balance": avg_balance,
                "Cash Withdrawals": float(cash_withdrawals),
                "Number of Open Credit Accounts": int(credit_accounts),
                "Income Stability": bool(income_stability),
                "MonthStart": grp["MonthStart"].min()
            })

        if not metrics_rows:
            logger.warning("No monthly groups available; returning empty metrics.")
            return pd.DataFrame()

        # 6) Build frame and lock dtypes for Arrow
        metrics_df = pd.DataFrame(metrics_rows)

        float_cols = [
            "Average Monthly Income", "Average Monthly Expenses", "Average Monthly EMI",
            "Net Surplus", "DTI Ratio", "Savings Rate", "Credit Utilization",
            "Discretionary Expenses", "Average Monthly Balance", "Cash Withdrawals"
        ]
        for c in float_cols:
            metrics_df[c] = pd.to_numeric(metrics_df[c], errors="coerce").fillna(0.0).astype("float64")

        int_cols = ["Red Flag Count", "Number of Open Credit Accounts"]
        for c in int_cols:
            metrics_df[c] = pd.to_numeric(metrics_df[c], errors="coerce").fillna(0).astype("int64")

        metrics_df["Income Stability"] = metrics_df["Income Stability"].astype(bool)
        metrics_df["Month"] = metrics_df["Month"].astype(str)
        metrics_df["MonthStart"] = pd.to_datetime(metrics_df["MonthStart"], errors="coerce")

        metrics_df = metrics_df.sort_values("Month").reset_index(drop=True)
        metrics_df = make_arrow_compatible(metrics_df)

        logger.info(f"[DEBUG] aggregated_df types:\n{metrics_df.dtypes}")
    
        # ✅ You can do the same for Savings Rate
        if "Savings Rate" not in metrics_df.columns and {"Average Monthly Income","Average Monthly Expenses"}.issubset(metrics_df.columns):
            inc = pd.to_numeric(metrics_df["Average Monthly Income"], errors="coerce").fillna(0.0)
            exp = pd.to_numeric(metrics_df["Average Monthly Expenses"], errors="coerce").fillna(0.0)
            metrics_df

        # 7) Save CSV (no index column sneaking in)
        if os.path.dirname(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        metrics_df.to_csv(output_path, index=False)
        logger.info(f"✅ Metrics saved to {output_path}")
        return metrics_df

    except Exception as e:
        logger.exception(f"Error in calculate_metrics: {e}")
        return pd.DataFrame()


# Rule-based categorization (for fallback)
category_rules = {
    "Income": [
        "salary", "bonus", "freelance", "investment",
        r"neft.*cr", r"imps.*cr", r"upi.*cr",
        r"\[Salary\]", r"\[Bonus\]", r"\[Freelance\]"
    ],
    "Fixed Expenses": [
        "rent", "emi", "insurance", "electricity", "phone", "bill", "loan",
        r"paytm.*bill", r"\[Rent\]", r"\[Electricity\]"
    ],
    "Discretionary Expenses": [
        "shopping", "dining", "entertainment", "travel", "amazon", "zomato", "swiggy", "flipkart",
        r"\[Shopping\]", r"\[Dining\]"
    ],
    "Savings": [
        "fd", "rd", "mutual fund", "deposit", "sip", "ppf", "nps",
        r"\[FD\]", r"\[RD\]"
    ],
    "Red Flags": [
        "overdraft", "bounce", "insufficient funds",
        r"cash.*(?:\d{4,})",        # ✅ non-capturing group
        r"upi.*(?:\d{5,}).*cash"    # ✅ non-capturing group
    ]
}


def rule_based_categorize(description):
    """Apply rule-based categorization."""
    if not isinstance(description, str):
        return "Uncategorized"
    description = description.lower()
    for category, patterns in category_rules.items():
        for pattern in patterns:
            if re.search(pattern, description):
                return category
    return "Uncategorized"

# Savings Metrics
def calculate_savings_metrics(df: pd.DataFrame):
    """Savings rate and emergency fund months with safe divisions."""
    df = _ensure_cols(df, {"Category": "Uncategorized", "Credit": 0.0, "Debit": 0.0})
    credits = pd.to_numeric(df["Credit"], errors="coerce").fillna(0.0)
    debits = pd.to_numeric(df["Debit"], errors="coerce").fillna(0.0)

    total_savings_in = pd.to_numeric(df.loc[df["Category"].str.lower() == "savings", "Credit"], errors="coerce").sum()
    total_income = pd.to_numeric(df.loc[df["Category"].str.lower() == "income", "Credit"], errors="coerce").sum()
    savings_rate = (total_savings_in / total_income * 100.0) if total_income > 1e-9 else 0.0

    fixed_expenses_mean = pd.to_numeric(df.loc[df["Category"].str.lower() == "fixed expenses", "Debit"], errors="coerce").mean()
    fixed_expenses_mean = fixed_expenses_mean if pd.notna(fixed_expenses_mean) and fixed_expenses_mean > 1e-9 else 0.0
    emergency_fund_months = (total_savings_in / fixed_expenses_mean) if fixed_expenses_mean > 0 else 0.0
    return float(savings_rate), float(emergency_fund_months)

# Batch Processing
def process_folder(input_folder: str, output_folder: str):
    """Process all CSVs in input_folder and save metrics to output_folder."""
    os.makedirs(output_folder, exist_ok=True)
    all_metrics = []

    for file in os.listdir(input_folder):
        if not file.lower().endswith(".csv"):
            continue

        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file[:-4] + "_metrics.csv")
        logger.info(f"[DEBUG] Processing file: {file}")
        try:
            df = pd.read_csv(input_path)
            df = standardize_columns(df)
            df = clean_currency_column(df, ["Debit", "Credit", "Balance", "Amount"])
            metrics_df = calculate_metrics(df, output_path)
            if not metrics_df.empty:
                all_metrics.append((file, metrics_df))
                logger.info(f"✅ Metrics calculated for {file}, saved to {output_path}")
            else:
                logger.warning(f"⚠️ No valid data in {file}, skipping")
        except Exception as e:
            logger.exception(f"⚠️ Failed to process {file}: {e}")

    if all_metrics:
        combined = pd.concat([m for _, m in all_metrics], axis=0, ignore_index=False)
        combined.to_csv(os.path.join(output_folder, "combined_metrics.csv"))
        logger.info(f"✅ Combined metrics saved to {os.path.join(output_folder, 'combined_metrics.csv')}")
    else:
        logger.warning("[WARNING] No metrics calculated for any files")

    return all_metrics