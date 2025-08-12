# utils/visualizer.py
import os
import re
import io
import json
import base64
import pickle
import argparse
from datetime import datetime
from pathlib import Path
import plotly.express as px

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Try to import project utils; if not available, fallback to simple helpers
try:
    from utils.text_parser import parse_bank_statement_text
except Exception:
    parse_bank_statement_text = None

try:
    from utils.categorize_transactions import categorize_transactions
except Exception:
    categorize_transactions = None

try:
    from utils.calculate_metrics import calculate_metrics
except Exception:
    calculate_metrics = None

try:
    from utils.score_bank_statements import score_and_decide
except Exception:
    score_and_decide = None

# PDF extraction
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

# ----------------------------
# Helpers
# ----------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return Path(path)

def standardize_columns(df):
    mapping = {
        'debit': 'Debit', 'withdrawal': 'Debit', 'dr': 'Debit',
        'credit': 'Credit', 'deposit': 'Credit', 'cr': 'Credit',
        'balance': 'Balance', 'bal': 'Balance',
        'date': 'Date', 'transaction date': 'Date',
        'description': 'Description', 'narration': 'Description', 'particulars': 'Description',
        'amount': 'Amount', 'type': 'Type', 'category': 'Category'
    }
    df.columns = [mapping.get(c.lower().strip(), c) for c in df.columns]
    return df

# --------------- Improved fallback parser ---------------
def basic_parse_text_to_df(text):
    """
    Improved fallback parser: extracts Date, Description, Debit, Credit, Balance where possible.
    Designed to handle many common bank-statement text patterns.
    """
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    records = []

    # Patterns for many date formats
    date_regex = r'(?P<date>\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|\w{3,9}\s+\d{1,2},?\s*\d{4}|\w{3,9}\s+\d{4})'
    # amount pattern captures potential ₹, commas, decimals, optional leading minus
    amt_regex = r'(?P<amt>-?₹?\s?\d{1,3}(?:[,\d{3}])*(?:\.\d{1,2})?)'
    # Combined pattern to attempt capture of typical statement line forms
    combined_re = re.compile(date_regex + r'.{0,60}?' + amt_regex + r'.{0,60}?' + amt_regex, re.IGNORECASE)

    for i, line in enumerate(lines):
        # first try combined pattern
        m = combined_re.search(line)
        if m:
            date_str = m.group('date')
            amts = re.findall(amt_regex, line)
            # choose likely debit/credit by presence of '-' or positions
            debit = 0.0
            credit = 0.0
            if amts:
                # normalize amounts
                normalized = [re.sub(r'[₹,\s]', '', a).replace('−', '-').replace('—', '-') for a in amts]
                # Heuristic: negative sign implies debit
                for a_raw, a_norm in zip(amts, normalized):
                    if '-' in a_raw or a_norm.startswith('-'):
                        try:
                            debit = abs(float(a_norm))
                        except:
                            pass
                    else:
                        try:
                            credit = float(a_norm)
                        except:
                            pass
            desc = line.replace(m.group('date'), '').strip()
            records.append({"Date": date_str, "Description": desc, "Debit": debit, "Credit": credit, "Balance": np.nan})
            continue

        # fallback: search for any date on line
        date_search = re.search(date_regex, line, re.IGNORECASE)
        if date_search:
            date_str = date_search.group('date')
            # check neighbors for amount tokens
            window = " ".join(lines[max(0, i-1):i+3])
            amts = re.findall(amt_regex, window)
            debit = 0.0
            credit = 0.0
            if amts:
                normalized = [re.sub(r'[₹,\s]', '', a).replace('−', '-').replace('—', '-') for a in amts]
                for a_raw, a_norm in zip(amts, normalized):
                    if '-' in a_raw or a_norm.startswith('-'):
                        try:
                            debit = abs(float(a_norm))
                        except:
                            pass
                    else:
                        try:
                            credit = float(a_norm)
                        except:
                            pass
            desc = re.sub(date_regex, '', line).strip()
            records.append({"Date": date_str, "Description": desc, "Debit": debit, "Credit": credit, "Balance": np.nan})
            continue

    df = pd.DataFrame(records)
    # normalize numeric columns
    for col in ["Debit", "Credit", "Balance"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        else:
            df[col] = 0.0
    # parse dates robustly
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce', dayfirst=True)
    df = df.dropna(subset=["Date"]).reset_index(drop=True)
    return df[["Date", "Description", "Debit", "Credit", "Balance"]]

def parse_pdf_to_df(pdf_path):
    """Extract text from PDF using PyMuPDF and parse to DataFrame (use project's parser if available)."""
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) not installed; install with `pip install pymupdf`.")
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    # Use project's parser if available (preferred)
    if parse_bank_statement_text:
        try:
            df = parse_bank_statement_text(text)
            return standardize_columns(df)
        except Exception:
            pass
    # fallback improved parser
    df = basic_parse_text_to_df(text)
    return standardize_columns(df)

# --------------- Categorization ---------------
def simple_categorize(df):
    df = df.copy()
    def cat(desc):
        s = str(desc).lower()
        if any(k in s for k in ["salary", "credit", "bonus", "payroll"]):
            return "Income"
        if any(k in s for k in ["rent","emi","loan","mortgage","insurance","bill","electricity","phone","utility"]):
            return "Fixed Expenses"
        if any(k in s for k in ["shopping","dining","restaurant","zomato","amazon","flipkart","movie","entertainment","travel"]):
            return "Discretionary Expenses"
        if any(k in s for k in ["fd","rd","mutual fund","sip","deposit","investment","savings"]):
            return "Savings"
        if any(k in s for k in ["overdraft","bounce","insufficient","penalty","late fee","chargeback"]):
            return "Red Flags"
        return "Other"
    df["Category"] = df["Description"].fillna("").apply(cat)
    return df

# --------------- ML model integration (auto feature selection) ---------------
def load_model(model_path):
    model_path = Path(model_path)
    if not model_path.exists():
        return None
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print("Failed to load ML model:", e)
        return None

def prepare_features_for_model(model, metrics_df):
    """
    Given a sklearn-like model and a metrics_df (single-row aggregated metrics or dict),
    build a feature vector automatically:
      - If model.feature_names_in_ exists, use it and fetch those columns from metrics_df (fill 0 if missing).
      - Else, use metrics_df.columns in sorted order and supply values in that order.
    Returns: X (1D or 2D array), feature_names_used (list)
    """
    # metrics_df may be DataFrame or dict
    if isinstance(metrics_df, dict):
        metrics = metrics_df
        metrics_df = pd.DataFrame([metrics])
    elif isinstance(metrics_df, pd.DataFrame):
        metrics = metrics_df.iloc[0].to_dict()
    else:
        raise ValueError("metrics_df must be dict or DataFrame")

    # Try to get feature names from model
    feature_names = None
    try:
        feature_names = getattr(model, "feature_names_in_", None)
    except Exception:
        feature_names = None

    # Some pipelines store named_steps or final estimator; try to introspect
    if feature_names is None:
        # If model is a pipeline with vectorizer, maybe vectorizer has feature names
        try:
            if hasattr(model, "named_steps"):
                # try to find a transformer with feature_names_in_ or get_feature_names_out
                for step in model.named_steps.values():
                    if hasattr(step, "get_feature_names_out"):
                        feature_names = step.get_feature_names_out()
                        break
        except Exception:
            feature_names = None

    # Fallback: use metrics_df columns
    if feature_names is None:
        feature_names = list(metrics_df.columns)

    feature_names = list(feature_names)
    # Build X
    X = []
    for fn in feature_names:
        # normalize column keys: allow spaces vs underscores
        if fn in metrics_df.columns:
            val = metrics_df.iloc[0][fn]
        else:
            # try variants
            alt = fn.replace("_", " ")
            if alt in metrics_df.columns:
                val = metrics_df.iloc[0][alt]
            else:
                # pick similar column by lowercase match
                lowered = {c.lower(): c for c in metrics_df.columns}
                if fn.lower() in lowered:
                    val = metrics_df.iloc[0][lowered[fn.lower()]]
                else:
                    # missing -> default 0
                    val = 0.0
        # coerce numeric
        try:
            if pd.isna(val):
                val = 0.0
            val = float(val)
        except Exception:
            # encode booleans and categories sensibly
            if isinstance(val, bool):
                val = 1.0 if val else 0.0
            else:
                # fallback map using length or presence
                try:
                    val = float(str(val))
                except:
                    val = 0.0
        X.append(val)
    X_arr = np.array(X).reshape(1, -1)
    return X_arr, feature_names

def model_predict_and_explain(model, metrics_df):
    """
    Run model.predict / predict_proba and build a small explanation dict.
    """
    if model is None:
        return None
    try:
        X, feature_names = prepare_features_for_model(model, metrics_df)
        pred = None
        prob = None
        try:
            pred = model.predict(X)[0]
        except Exception:
            # model might expect different shape; try 1D
            pred = model.predict(X.reshape(1, -1))[0]
        try:
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X)[0].max()
            elif hasattr(model, "decision_function"):
                dfun = model.decision_function(X)[0]
                prob = float(1 / (1 + np.exp(-dfun)))  # approximate sigmoid
        except Exception:
            prob = None
        return {"model_prediction": str(pred), "model_probability": float(prob) if prob is not None else None, "features_used": feature_names}
    except Exception as e:
        print("Model prediction failed:", e)
        return None

# --------------- Visualization helpers ---------------

def save_fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64

def fill_months_and_plot_timeseries(df, date_col, value_col, out_png, fill_method="zero", title=None, ylabel=None):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    if df.empty:
        return None, None
    monthly = df.groupby(df[date_col].dt.to_period("M"))[value_col].sum()
    full_range = pd.period_range(monthly.index.min(), monthly.index.max(), freq="M")
    monthly = monthly.reindex(full_range).astype(float)
    if fill_method == "zero":
        monthly = monthly.fillna(0.0)
    else:
        monthly = monthly.interpolate().fillna(0.0)
    if isinstance(monthly.index, pd.PeriodIndex):
        x = monthly.index.to_timestamp()
    else:
        x = monthly.index.to_pydatetime()
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(x, monthly.values, marker='o', linewidth=2)
    ax.bar(x, monthly.values, alpha=0.25)
    ax.set_xticks(x)
    ax.set_xticklabels([dt.strftime("%b %Y") for dt in x], rotation=45)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_title(title or f"Monthly {value_col}")
    ax.set_ylabel(ylabel or value_col)
    fig.tight_layout()
    fig.savefig(out_png)
    b64 = None
    try:
        b64 = save_fig_to_base64(fig)
    except:
        pass
    plt.close(fig)
    return out_png, b64

def plot_income_vs_expense_from_df(df, out_png):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    df = df.dropna(subset=["Date"])
    monthly = df.set_index("Date").groupby(pd.Grouper(freq="ME")).agg(Income=("Credit","sum"), Expenses=("Debit","sum"))
    if monthly.empty:
        return None, None
    if isinstance(monthly.index, pd.PeriodIndex):
        x = monthly.index.to_timestamp()
    else:
        x = monthly.index.to_pydatetime()
    fig, ax = plt.subplots(figsize=(12,5))
    ax.bar(x - pd.Timedelta(days=6), monthly["Income"], width=12, label="Income")
    ax.bar(x + pd.Timedelta(days=6), monthly["Expenses"], width=12, label="Expenses")
    ax.set_xticks(x)
    ax.set_xticklabels([d.strftime("%b %Y") for d in x], rotation=45)
    ax.set_title("Income vs Expenses (Monthly)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_png)
    b64 = save_fig_to_base64(fig)
    plt.close(fig)
    return out_png, b64

def plot_category_pie(df, out_png):
    if "Category" not in df.columns:
        return None, None
    counts = df["Category"].value_counts()
    if counts.empty:
        return None, None
    fig, ax = plt.subplots(figsize=(6,6))
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=140)
    ax.set_title("Spending by Category")
    fig.tight_layout()
    fig.savefig(out_png)
    b64 = save_fig_to_base64(fig)
    plt.close(fig)
    return out_png, b64

def plot_high_risk_timeline(df, out_png):
    if "Category" not in df.columns:
        return None, None
    high = df[df["Category"].str.lower() == "red flags"]
    if high.empty:
        return None, None
    high = high.copy()
    high["Date"] = pd.to_datetime(high["Date"], errors='coerce')
    high = high.dropna(subset=["Date"])
    fig, ax = plt.subplots(figsize=(12,3))
    ax.scatter(high["Date"], np.ones(len(high)), c='red', marker='x')
    for _, r in high.iterrows():
        ax.annotate(f"{str(r['Description'])[:30]} ({r.get('Debit', r.get('Credit', ''))})", (r["Date"], 1),
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    ax.set_yticks([])
    ax.set_title("High-Risk Transactions Timeline (Red Flags)")
    fig.tight_layout()
    fig.savefig(out_png)
    b64 = save_fig_to_base64(fig)
    plt.close(fig)
    return out_png, b64

def plot_cibil_gauge(score, out_png):
    fig, ax = plt.subplots(figsize=(6,3))
    ax.axis('off')
    score = max(300, min(900, int(score)))
    ax.barh([0], [600], color="#eee")
    color = "#4caf50" if score>=700 else "#ff9800" if score>=600 else "#f44336"
    ax.barh([0], [score-300], color=color)
    ax.text(0.5, 0.1, f"CIBIL Score: {score}", ha='center', va='center', transform=ax.transAxes, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_png)
    b64 = save_fig_to_base64(fig)
    plt.close(fig)
    return out_png, b64

def plot_income_trend_plotly(metrics_df):
    """Interactive line chart of monthly income."""
    try:
        df = metrics_df.copy()
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        elif "Month" in df.columns:
            df["Date"] = pd.to_datetime(df["Month"], errors='coerce')
        else:
            return None
        if "Credit" not in df.columns:
            return None
        monthly_income = df.groupby(pd.Grouper(key="Date", freq="M"))["Credit"].sum().reset_index()
        fig = px.line(monthly_income, x="Date", y="Credit", markers=True,
                      title="Monthly Income Trend", labels={"Credit": "Income (₹)"})
        fig.update_traces(line=dict(width=3))
        return fig
    except Exception:
        return None

def plot_surplus_trend_plotly(metrics_df):
    """Interactive line chart of monthly surplus (income - expenses)."""
    try:
        df = metrics_df.copy()
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        else:
            return None
        if not {"Credit", "Debit"}.issubset(df.columns):
            return None
        df["Surplus"] = df["Credit"] - df["Debit"]
        monthly_surplus = df.groupby(pd.Grouper(key="Date", freq="M"))["Surplus"].sum().reset_index()
        fig = px.line(monthly_surplus, x="Date", y="Surplus", markers=True,
                      title="Monthly Surplus Trend", labels={"Surplus": "Surplus (₹)"})
        fig.update_traces(line=dict(width=3))
        return fig
    except Exception:
        return None

def plot_income_vs_expenses_plotly(metrics_df):
    """Interactive grouped bar chart: monthly income vs expenses."""
    try:
        df = metrics_df.copy()
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        else:
            return None
        if not {"Credit", "Debit"}.issubset(df.columns):
            return None
        monthly = df.groupby(pd.Grouper(key="Date", freq="M")).agg(
            Income=("Credit", "sum"),
            Expenses=("Debit", "sum")
        ).reset_index()
        monthly_long = monthly.melt(id_vars="Date", value_vars=["Income", "Expenses"],
                                    var_name="Type", value_name="Amount")
        fig = px.bar(monthly_long, x="Date", y="Amount", color="Type", barmode="group",
                     title="Monthly Income vs Expenses", labels={"Amount": "Amount (₹)"})
        return fig
    except Exception:
        return None

def plot_cumulative_savings_plotly(metrics_df):
    """Interactive line chart of cumulative savings over time."""
    try:
        df = metrics_df.copy()
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        else:
            return None
        if not {"Credit", "Debit"}.issubset(df.columns):
            return None
        df["Savings"] = df["Credit"] - df["Debit"]
        df = df.sort_values("Date")
        df["Cumulative Savings"] = df["Savings"].cumsum()
        fig = px.line(df, x="Date", y="Cumulative Savings", markers=True,
                      title="Cumulative Savings Over Time", labels={"Cumulative Savings": "Savings (₹)"})
        fig.update_traces(line=dict(width=3))
        return fig
    except Exception:
        return None

def plot_category_breakdown_plotly(metrics_df):
    """Interactive pie chart of expense category distribution."""
    try:
        if "Category" not in metrics_df.columns:
            return None
        cat_counts = metrics_df["Category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        fig = px.pie(cat_counts, values="Count", names="Category",
                     title="Spending by Category", hole=0.4)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        return fig
    except Exception:
        return None

# --------------- HTML report ---------------
HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Loan Analysis Report - {name}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    h1 {{ color: #2c3e50; }}
    .section {{ margin-bottom: 28px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background: #f4f6f7; }}
    .img {{ max-width: 100%; height: auto; }}
    .decision {{ padding: 12px; border-radius: 6px; }}
    .approve {{ background: #e8f8f5; border: 1px solid #2ecc71; }}
    .reject {{ background: #fdecea; border: 1px solid #e74c3c; }}
  </style>
</head>
<body>
  <h1>Loan Analysis Report</h1>
  <p>Generated: {ts}</p>

  <div class="section">
    <h2>Input</h2>
    <p><strong>Source file:</strong> {input_file}</p>
    <p><strong>CIBIL Score:</strong> {cibil}</p>
    <p><strong>Proposed EMI:</strong> {emi}</p>
  </div>

  <div class="section">
    <h2>Decision</h2>
    <div class="decision {decision_class}">
      <p><strong>Action:</strong> {action}</p>
      <p><strong>Risk Level:</strong> {risk}</p>
      <p><strong>Reason:</strong> {reason}</p>
      <p><strong>Heuristic Score:</strong> {score}</p>
      {ml_block}
    </div>
  </div>

  <div class="section">
    <h2>Key Metrics</h2>
    {metrics_table}
  </div>

  <div class="section">
    <h2>Sample Transactions (first 20)</h2>
    {transactions_table}
  </div>

  <div class="section">
    <h2>Visualizations</h2>
    {visuals}
  </div>
</body>
</html>
"""

def df_to_html_table(df, max_rows=50):
    return df.head(max_rows).to_html(classes="table", index=False, float_format="{:,.2f}".format)

def build_html_report(out_path, context):
    html = HTML_TEMPLATE.format(**context)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path

# --------------- CSV export ---------------
def export_raw_csv(df, out_csv):
    df.to_csv(out_csv, index=False)
    return out_csv

# --------------- Main pipeline ---------------
def analyze_file(input_path, cibil_score=720, proposed_emi=8000, fill_method="zero", out_dir="outputs"):
    input_path = Path(input_path)
    out_dir = ensure_dir(out_dir)
    plots_dir = ensure_dir(out_dir / "plots")
    metrics_dir = ensure_dir(out_dir / "metrics")
    raw_dir = ensure_dir(out_dir / "raw_transactions")
    categorized_dir = ensure_dir(out_dir / "categorized")
    decisions_dir = ensure_dir(out_dir / "loan_decisions")
    html_dir = ensure_dir(out_dir / "reports")

    # 1) Extract
    print("[1/8] Extracting transactions...")
    df = None
    if input_path.suffix.lower() == ".pdf":
        try:
            df = parse_pdf_to_df(str(input_path))
        except Exception as e:
            print("PDF parse error:", e)
            # fallback -> empty
            df = pd.DataFrame()
    elif input_path.suffix.lower() in [".csv", ".txt", ".xls", ".xlsx"]:
        try:
            if input_path.suffix.lower() in [".xls", ".xlsx"]:
                df = pd.read_excel(str(input_path))
            else:
                df = pd.read_csv(str(input_path))
            df = standardize_columns(df)
        except Exception as e:
            print("File load error:", e)
            df = pd.DataFrame()
    else:
        raise ValueError("Unsupported file type; supported: pdf, csv, xlsx, txt")

    if df is None or df.empty:
        print("No transactions extracted — exiting.")
        return None

    # Export raw CSV
    raw_csv_path = raw_dir / f"{input_path.stem}_raw.csv"
    export_raw_csv(df, raw_csv_path)
    print(f"Exported raw CSV: {raw_csv_path}")

    # Ensure columns and types
    df = standardize_columns(df)
    for col in ["Debit", "Credit", "Balance"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[₹,]', '', regex=True), errors='coerce').fillna(0.0)
        else:
            df[col] = 0.0
    if "Date" not in df.columns:
        print("No Date column found after parsing. Exiting.")
        return None
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce', dayfirst=True)
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # 2) Categorize
    print("[2/8] Categorizing transactions...")
    categorized_csv_path = categorized_dir / f"{input_path.stem}_categorized.csv"
    if categorize_transactions:
        try:
            tmp = out_dir / f"{input_path.stem}_tmp_for_cat.csv"
            df.to_csv(tmp, index=False)
            categorized_df = categorize_transactions(str(tmp), str(categorized_csv_path))
            if categorized_df is None or categorized_df.empty:
                print("categorize_transactions returned empty, falling back.")
                categorized_df = simple_categorize(df)
                categorized_df.to_csv(categorized_csv_path, index=False)
        except Exception as e:
            print("categorize util error:", e)
            categorized_df = simple_categorize(df)
            categorized_df.to_csv(categorized_csv_path, index=False)
    else:
        categorized_df = simple_categorize(df)
        categorized_df.to_csv(categorized_csv_path, index=False)

    print(f"Categorized CSV: {categorized_csv_path}")

    # 3) Metrics
    print("[3/8] Calculating metrics...")
    metrics_file = metrics_dir / f"{input_path.stem}_metrics.csv"
    if calculate_metrics:
        try:
            out_metrics = calculate_metrics(categorized_df, str(metrics_file))
            if isinstance(out_metrics, pd.DataFrame):
                aggregated_df = out_metrics
            elif isinstance(out_metrics, dict):
                aggregated_df = pd.DataFrame([out_metrics])
            else:
                aggregated_df = pd.DataFrame([out_metrics])
            aggregated_df.to_csv(metrics_file, index=False)
        except Exception as e:
            print("calculate_metrics util error:", e)
            aggregated_df = None
    else:
        aggregated_df = None

    # fallback aggregation if aggregated_df not available
    if aggregated_df is None or aggregated_df.empty:
        monthly = categorized_df.set_index("Date").groupby(pd.Grouper(freq="M")).agg(
            Income=("Credit","sum"), Expenses=("Debit","sum"))
        avg_income = monthly["Income"].mean() if not monthly.empty else 0.0
        avg_expenses = monthly["Expenses"].mean() if not monthly.empty else 0.0
        avg_emi = categorized_df[categorized_df["Description"].str.lower().str.contains("emi")]["Debit"].sum()
        net_surplus = avg_income - avg_expenses
        aggregated_df = pd.DataFrame([{
            "Average Monthly Income": avg_income,
            "Average Monthly Expenses": avg_expenses,
            "Average Monthly EMI": avg_emi,
            "Net Surplus": net_surplus,
            "DTI Ratio": 0.0,
            "Savings Rate": 0.0,
            "Red Flag Count": int((categorized_df["Category"]=="Red Flags").sum()),
            "Discretionary Expenses": categorized_df[categorized_df["Category"]=="Discretionary Expenses"]["Debit"].sum()
        }])
        aggregated_df.to_csv(metrics_file, index=False)

    print(f"Saved metrics: {metrics_file}")

    # 4) Decision (heuristic)
    print("[4/8] Heuristic scoring & decisioning...")
    if score_and_decide:
        try:
            heuristic_decision = score_and_decide(metrics_df=aggregated_df, cibil_score=cibil_score,
                                                  loan_purpose="Neutral", proposed_emi=proposed_emi)
        except Exception as e:
            print("score_and_decide util error:", e)
            heuristic_decision = None
    else:
        # fallback simple heuristic
        row = aggregated_df.iloc[0]
        avg_income = float(row.get("Average Monthly Income", 0.0))
        avg_exp = float(row.get("Average Monthly Expenses", 0.0))
        net_surplus = float(row.get("Net Surplus", avg_income - avg_exp))
        dti = float(row.get("DTI Ratio", 0.0))
        score = 50.0
        if cibil_score >= 750:
            score += 20
        if net_surplus >= 2 * proposed_emi:
            score += 20
        if dti < 40:
            score += 10
        action = "Approve with standard terms" if score >= 80 else ("Approve with caution" if score >= 60 else "Reject")
        heuristic_decision = {"Total Score": round(score,1), "Risk Level": "Low" if score>=80 else "Moderate" if score>=60 else "High",
                              "Action": action, "Reason": f"Heuristic score {score:.1f}, CIBIL {cibil_score}, Net Surplus {net_surplus:.2f}"}

    print("Heuristic decision:", heuristic_decision)

    # 5) ML model prediction (auto feature selection)
    print("[5/8] ML model integration (if model exists)...")
    model_path = Path("models/loan_approval_model.pkl")
    model = load_model(model_path)
    ml_result = None
    if model is not None:
        try:
            ml_result = model_predict_and_explain(model, aggregated_df)
            print("ML result:", ml_result)
        except Exception as e:
            print("ML model prediction failed:", e)
            ml_result = None
    else:
        print("No ML model found at", model_path)

    # 6) Visualizations
    print("[6/8] Generating visualizations...")
    categorized_df["Amount"] = categorized_df["Credit"].fillna(0.0) - categorized_df["Debit"].fillna(0.0)

    # monthly net flow
    monthly_plot = plots_dir / f"{input_path.stem}_monthly_netflow.png"
    monthly_png, monthly_b64 = fill_months_and_plot_timeseries(categorized_df, "Date", "Amount", str(monthly_plot),
                                                               fill_method=fill_method, title="Monthly Net Flow (Credit - Debit)", ylabel="Amount (₹)")

    # income vs expenses
    income_exp_png = plots_dir / f"{input_path.stem}_income_vs_expenses.png"
    income_exp_path, income_exp_b64 = plot_income_vs_expense_from_df(categorized_df, str(income_exp_png))

    # category pie
    cat_pie_png = plots_dir / f"{input_path.stem}_category_pie.png"
    cat_pie_path, cat_pie_b64 = plot_category_pie(categorized_df, str(cat_pie_png))

    # high risk timeline
    high_risk_png = plots_dir / f"{input_path.stem}_high_risk_timeline.png"
    high_risk_path, high_risk_b64 = plot_high_risk_timeline(categorized_df, str(high_risk_png))

    # cibil gauge
    cibil_png = plots_dir / f"{input_path.stem}_cibil_gauge.png"
    cibil_path, cibil_b64 = plot_cibil_gauge(cibil_score, str(cibil_png))

    print("Saved plots to", plots_dir)

    # 7) HTML report
    print("[7/8] Building HTML report...")
    visuals_html = ""
    def img_tag(b64, title):
        if not b64:
            return ""
        return f'<div><h4>{title}</h4><img class="img" src="data:image/png;base64,{b64}"/></div>'

    visuals_html += img_tag(monthly_b64, "Monthly Net Flow")
    visuals_html += img_tag(income_exp_b64, "Income vs Expenses")
    visuals_html += img_tag(cat_pie_b64, "Spending Breakdown by Category")
    visuals_html += img_tag(high_risk_b64, "High-Risk Transactions Timeline")
    visuals_html += img_tag(cibil_b64, "CIBIL Score Gauge")

    decision_action = heuristic_decision.get("Action") if heuristic_decision else "N/A"
    decision_reason = heuristic_decision.get("Reason") if heuristic_decision else ""
    decision_score = heuristic_decision.get("Total Score") if heuristic_decision else 0
    decision_risk = heuristic_decision.get("Risk Level") if heuristic_decision else "N/A"
    decision_class = "approve" if "approve" in str(decision_action).lower() else "reject"

    ml_block = ""
    if ml_result is not None:
        ml_block = f"<p><strong>ML Model Prediction:</strong> {ml_result.get('model_prediction')} (prob {ml_result.get('model_probability')})</p>"
        ml_block += f"<p><strong>Features used (sample):</strong> {ml_result.get('features_used')[:10]}</p>"

    metrics_table = aggregated_df.round(2).to_html(index=False)
    transactions_table = df.head(20).to_html(index=False, float_format="{:,.2f}".format)

    context = {
        "name": input_path.stem,
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_file": str(input_path),
        "cibil": cibil_score,
        "emi": proposed_emi,
        "action": decision_action,
        "risk": decision_risk,
        "reason": decision_reason,
        "score": decision_score,
        "decision_class": decision_class,
        "metrics_table": metrics_table,
        "transactions_table": transactions_table,
        "visuals": visuals_html,
        "ml_block": ml_block
    }

    html_path = html_dir / f"{input_path.stem}_report.html"
    build_html_report(html_path, context)
    print("Saved HTML report:", html_path)

    # 8) Save decision JSON and log
    print("[8/8] Saving decision JSON and log...")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record = {
        "timestamp": timestamp,
        "file": str(input_path),
        "cibil": cibil_score,
        "proposed_emi": proposed_emi,
        "heuristic_action": heuristic_decision.get("Action"),
        "heuristic_score": heuristic_decision.get("Total Score"),
        "heuristic_reason": heuristic_decision.get("Reason"),
        "ml_prediction": ml_result.get("model_prediction") if ml_result else None,
        "ml_probability": ml_result.get("model_probability") if ml_result else None
    }
    decision_json = decisions_dir / f"{input_path.stem}_decision.json"
    with open(decision_json, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    log_csv = decisions_dir / "decisions_log.csv"
    pd.DataFrame([record]).to_csv(log_csv, mode='a', header=not log_csv.exists(), index=False)

    print("Decision JSON saved:", decision_json)
    print("Decision log appended:", log_csv)

    out = {
        "raw_csv": str(raw_csv_path),
        "categorized_csv": str(categorized_csv_path),
        "metrics_csv": str(metrics_file),
        "plots": [str(p) for p in plots_dir.glob(f"{input_path.stem}*")],
        "html_report": str(html_path),
        "decision_json": str(decision_json),
    }
    return out

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Powerful bank-statement analyzer and loan decisioner.")
    parser.add_argument("--file", "-f", required=False, default="/mnt/data/Taylor_Fernandes_moderate.pdf",
                        help="Path to statement file (pdf/csv/xlsx/txt)")
    parser.add_argument("--cibil", "-c", type=int, default=720, help="CIBIL score (300-900)")
    parser.add_argument("--emi", "-e", type=float, default=8000.0, help="Proposed EMI amount")
    parser.add_argument("--fill", "-m", choices=["zero", "interpolate"], default="zero",
                        help="How to fill missing months in time-series")
    parser.add_argument("--out", "-o", default="outputs", help="Output directory")
    args = parser.parse_args()

    print("Starting analysis...")
    res = analyze_file(args.file, cibil_score=args.cibil, proposed_emi=args.emi, fill_method=args.fill, out_dir=args.out)
    if res is None:
        print("Analysis failed or no data extracted.")
    else:
        print("Analysis complete. Outputs:")
        for k, v in res.items():
            print(f" - {k}: {v}")
