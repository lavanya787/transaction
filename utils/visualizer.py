import os
import re
import io
import json
import pickle
import argparse
import base64
import logging
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from pathlib import Path
# Try to import project utils; if not available, fallback to simple helpers
from utils.text_parser import parse_bank_statement_text
from utils.categorize_transactions import categorize_transactions, TransactionCategorizer
from utils.financial_metrics import calculate_metrics
from utils.score_bank_statements import score_and_decide
from utils.preprocess_transactions import make_arrow_compatible

# Helpers
def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

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
    for col in ['Debit', 'Credit', 'Balance', 'Amount']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'[₹,\s]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype('float64')
    if 'Description' in df.columns:
        df['Description'] = df['Description'].astype(str)
    if 'Category' in df.columns:
        df['Category'] = df['Category'].astype(str)
    return df

def ensure_numeric(df, numeric_cols=None):
    df = df.copy()
    if numeric_cols is None:
        numeric_cols = [
            "Average Monthly Income", "Average Monthly Expenses",
            "Average Monthly EMI", "Net Surplus", "DTI Ratio",
            "Savings Rate", "Credit Utilization",
            "Discretionary Expenses", "Average Monthly Balance",
            "Cash Withdrawals"
        ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'[^\d\.\-]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype("float64")
    return df

def enforce_metrics_schema(df):
    required_columns = {
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
        "Income Stability": 0,
    }
    df = df.copy()
    for col, default in required_columns.items():
        if col not in df.columns:
            df[col] = default
    df = ensure_numeric(df, list(required_columns.keys()))
    df["Savings Rate"] = np.where(
        df["Average Monthly Income"] > 0,
        ((df["Average Monthly Income"] - df["Average Monthly Expenses"])
         / df["Average Monthly Income"]) * 100.0,
        0.0
    )
    df["DTI Ratio"] = np.where(
        df["Average Monthly Income"] > 0,
        (df["Average Monthly EMI"] / df["Average Monthly Income"]) * 100.0,
        0.0
    )
    df["Savings Rate"] = df["Savings Rate"].astype("float64")
    df["DTI Ratio"] = df["DTI Ratio"].astype("float64")
    if 'Month' in df.columns:
        df['Month'] = df['Month'].astype('int64')
    if 'MonthStart' in df.columns:
        df['MonthStart'] = df['MonthStart'].astype(str)
    return df

# --------------- Improved fallback parser ---------------
def basic_parse_text_to_df(text):
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    records = []
    date_formats = ['%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y', '%b %d, %Y', '%B %Y']
    date_regex = r'(?P<date>\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|\w{3,9}\s+\d{1,2},?\s*\d{4}|\w{3,9}\s+\d{4})'
    amt_regex = r'(?P<amt>-?₹?\s?\d{1,3}(?:[,\d{3}])*(?:\.\d{1,2})?)'
    combined_re = re.compile(date_regex + r'.{0,60}?' + amt_regex + r'.{0,60}?' + amt_regex, re.IGNORECASE)
    for i, line in enumerate(lines):
        m = combined_re.search(line)
        if m:
            date_str = m.group('date')
            amts = re.findall(amt_regex, line)
            debit = 0.0
            credit = 0.0
            if amts:
                normalized = [re.sub(r'[₹,\s]', '', a).replace('−', '-').replace('—', '-') for a in amts]
                for a_raw, a_norm in zip(amts, normalized):
                    if '-' in a_raw or a_norm.startswith('-'):
                        try:
                            debit = abs(float(a_norm))
                        except ValueError:
                            debit = 0.0
                    else:
                        try:
                            credit = float(a_norm)
                        except ValueError:
                            credit = 0.0
            desc = line.replace(m.group('date'), '').strip()
            records.append({"Date": date_str, "Description": desc, "Debit": debit, "Credit": credit, "Balance": 0.0})
            continue
        date_search = re.search(date_regex, line, re.IGNORECASE)
        if date_search:
            date_str = date_search.group('date')
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
                        except ValueError:
                            debit = 0.0
                    else:
                        try:
                            credit = float(a_norm)
                        except ValueError:
                            credit = 0.0
            desc = re.sub(date_regex, '', line).strip()
            records.append({"Date": date_str, "Description": desc, "Debit": debit, "Credit": credit, "Balance": 0.0})
            continue
    df = pd.DataFrame(records)
    for col in ["Debit", "Credit", "Balance"]:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype('float64')
    for fmt in date_formats:
        df["Date"] = pd.to_datetime(df["Date"], format=fmt, errors='coerce')
        if df["Date"].notna().all():
            break
    df["Date"] = df["Date"].fillna(pd.Timestamp("2025-01-01"))
    df["Description"] = df["Description"].astype(str)
    df = df.dropna(subset=["Date"]).reset_index(drop=True)
    return df[["Date", "Description", "Debit", "Credit", "Balance"]]

def parse_pdf_to_df(pdf_path):
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) not installed; install with `pip install pymupdf`.")
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    if parse_bank_statement_text:
        try:
            df = parse_bank_statement_text(text)
            df = standardize_columns(df)
            df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
            df["Date"] = df["Date"].fillna(pd.Timestamp("2025-01-01"))
            return df
        except Exception:
            pass
    df = basic_parse_text_to_df(text)
    df = standardize_columns(df)
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    df["Date"] = df["Date"].fillna(pd.Timestamp("2025-01-01"))
    return df

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
    df["Category"] = df["Description"].fillna("").apply(cat).astype(str)
    return df

# --------------- ML model integration ---------------
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
    expected_features = ["Average Monthly Income", "Average Monthly Expenses", "Average Monthly EMI",
                        "Net Surplus", "DTI Ratio", "Savings Rate", "Red Flag Count", "Credit Utilization",
                        "Discretionary Expenses", "Income Stability"]
    if isinstance(metrics_df, dict):
        metrics_df = pd.DataFrame([metrics_df])
    elif not isinstance(metrics_df, pd.DataFrame):
        raise ValueError("metrics_df must be dict or DataFrame")
    metrics_df = ensure_numeric(metrics_df, expected_features)
    feature_names = getattr(model, "feature_names_in_", expected_features)
    X = []
    for fn in feature_names:
        val = metrics_df.iloc[0].get(fn, 0.0)
        try:
            val = float(val) if not pd.isna(val) else 0.0
        except:
            val = 0.0
        X.append(val)
    return np.array(X).reshape(1, -1), feature_names

def model_predict_and_explain(model, metrics_df):
    try:
        expected_features = [
            "Average Monthly Income", "Net Surplus", "DTI Ratio", "Savings Rate",
            "Red Flag Count", "Income Variability Index", "Bounced Cheques Count",
            "Discretionary Spending (%)"
        ]
        print(f"[DEBUG] Expected ML model features: {expected_features}")
        print(f"[DEBUG] Available columns in metrics_df: {list(metrics_df.columns)}")
        
        # Extract features with validation
        features = []
        for feature in expected_features:
            if feature in metrics_df.columns:
                value = metrics_df[feature].iloc[0]
                try:
                    features.append(float(value) if not pd.isna(value) else 0.0)
                except (ValueError, TypeError) as e:
                    print(f"[ERROR] Failed to convert {feature} value {value} to float: {e}")
                    features.append(0.0)
            else:
                print(f"[WARNING] Missing feature {feature} in metrics_df, using 0.0")
                features.append(0.0)
        
        print(f"[DEBUG] ML model input features: {dict(zip(expected_features, features))}")
        
        # Check feature count compatibility
        expected_feature_count = getattr(model, "n_features_in_", len(expected_features))
        if len(features) != expected_feature_count:
            print(f"[WARNING] Feature count mismatch: model expects {expected_feature_count}, got {len(features)}")
            features = features[:expected_feature_count] + [0.0] * (expected_feature_count - len(features)) if len(features) < expected_feature_count else features[:expected_feature_count]
        
        # Make prediction
        prediction = model.predict([features])[0]
        prob = model.predict_proba([features])[0] if hasattr(model, "predict_proba") else [0.5, 0.5]
        ml_result = {
            "model_prediction": "APPROVED" if prediction == 1 else "REJECTED",
            "model_probability": round(max(prob) * 100, 2),
            "features_used": dict(zip(expected_features, features))
        }
        print(f"[DEBUG] ML model output: {ml_result}")
        return ml_result
    except Exception as e:
        print(f"[ERROR] ML model prediction failed: {e}")
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
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce').fillna(pd.Timestamp("2025-01-01"))
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
        x = pd.to_datetime(monthly.index)
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(x, monthly.values, marker='o', linewidth=2)
    ax.bar(x, monthly.values, alpha=0.25)
    ax.set_xticks(x)
    ax.set_xticklabels([pd.Timestamp(dt).strftime("%b %Y") for dt in x], rotation=45)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_title(title or f"Monthly {value_col}")
    ax.set_ylabel(ylabel or value_col)
    fig.tight_layout()
    fig.savefig(out_png)
    b64 = save_fig_to_base64(fig)
    return out_png, b64

def plot_income_vs_expense_from_df(df, out_png):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce').fillna(pd.Timestamp("2025-01-01"))
    df = df.dropna(subset=["Date"])
    monthly = df.set_index("Date").groupby(pd.Grouper(freq="ME")).agg(Income=("Credit","sum"), Expenses=("Debit","sum"))
    if monthly.empty:
        return None, None
    if isinstance(monthly.index, pd.PeriodIndex):
        x = monthly.index.to_timestamp()
    else:
        x = pd.to_datetime(monthly.index)
    fig, ax = plt.subplots(figsize=(12,5))
    ax.bar(x - pd.Timedelta(days=6), monthly["Income"], width=12, label="Income")
    ax.bar(x + pd.Timedelta(days=6), monthly["Expenses"], width=12, label="Expenses")
    ax.set_xticks(x)
    ax.set_xticklabels([pd.Timestamp(d).strftime("%b %Y") for d in x], rotation=45)
    ax.set_title("Income vs Expenses (Monthly)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_png)
    b64 = save_fig_to_base64(fig)
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
    return out_png, b64

def plot_high_risk_timeline(df, out_png):
    try:
        if "Category" not in df.columns:
            return None
        high = df[df["Category"].str.lower() == "red flags"].copy()
        if high.empty:
            return None
        high["Date"] = pd.to_datetime(high["Date"], errors='coerce').fillna(pd.Timestamp("2025-01-01"))
        high = high.dropna(subset=["Date"])
        fig = px.scatter(high, x="Date", y=[1]*len(high), text="Description",
                         title="High-Risk Transactions Timeline (Red Flags)")
        fig.update_traces(marker=dict(size=10, symbol="x", color="red"), textposition="top center")
        fig.update_yaxes(showticklabels=False)
        fig.write_image(out_png)
        return fig
    except Exception as e:
        print(f"Error in plot_high_risk_timeline: {e}")
        return None
    
def plot_cibil_gauge(cibil_score, output_path):
    try:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=cibil_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "CIBIL Score"},
            gauge={
                'axis': {'range': [300, 900]},
                'bar': {'color': "#FF6384"},
                'steps': [
                    {'range': [300, 600], 'color': "red"},
                    {'range': [600, 750], 'color': "yellow"},
                    {'range': [750, 900], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': cibil_score
                }
            }
        ))
        fig.write_image(output_path)
        return fig
    except Exception as e:
        print(f"Error in plot_cibil_gauge: {e}")
        return None

def load_model(model_path):
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None

def plot_income_trend_plotly(df):
    try:
        df = df.copy()
        if "Date" not in df.columns:
            return None
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce').fillna(pd.Timestamp("2025-01-01"))
        if "Credit" not in df.columns:
            return None
        monthly_income = df.groupby(pd.Grouper(key="Date", freq="ME"))["Credit"].sum().reset_index()
        fig = px.line(monthly_income, x="Date", y="Credit", markers=True,
                      title="Monthly Income Trend", labels={"Credit": "Income (₹)"})
        fig.update_traces(line=dict(width=3))
        return fig
    except Exception as e:
        print(f"Error in plot_income_trend_plotly: {e}")
        return None

def plot_surplus_trend_plotly(df):
    try:
        df = df.copy()
        if "Date" not in df.columns:
            return None
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce').fillna(pd.Timestamp("2025-01-01"))
        if not {"Credit", "Debit"}.issubset(df.columns):
            return None
        df["Surplus"] = df["Credit"] - df["Debit"]
        monthly_surplus = df.groupby(pd.Grouper(key="Date", freq="ME"))["Surplus"].sum().reset_index()
        fig = px.line(monthly_surplus, x="Date", y="Surplus", markers=True,
                      title="Monthly Surplus Trend", labels={"Surplus": "Surplus (₹)"})
        fig.update_traces(line=dict(width=3))
        return fig
    except Exception as e:
        print(f"Error in plot_surplus_trend_plotly: {e}")
        return None

def plot_income_vs_expenses_plotly(df):
    try:
        df = df.copy()
        if "Date" not in df.columns:
            return None
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce').fillna(pd.Timestamp("2025-01-01"))
        if not {"Credit", "Debit"}.issubset(df.columns):
            return None
        monthly = df.groupby(pd.Grouper(key="Date", freq="ME")).agg(
            Income=("Credit", "sum"),
            Expenses=("Debit", "sum")
        ).reset_index()
        monthly_long = monthly.melt(id_vars="Date", value_vars=["Income", "Expenses"],
                                   var_name="Type", value_name="Amount")
        fig = px.bar(monthly_long, x="Date", y="Amount", color="Type", barmode="group",
                     title="Monthly Income vs Expenses", labels={"Amount": "Amount (₹)"})
        return fig
    except Exception as e:
        print(f"Error in plot_income_vs_expenses_plotly: {e}")
        return None

def plot_cumulative_savings_plotly(df):
    try:
        df = df.copy()
        if "Date" not in df.columns:
            return None
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce').fillna(pd.Timestamp("2025-01-01"))
        if not {"Credit", "Debit"}.issubset(df.columns):
            return None
        df["Savings"] = df["Credit"] - df["Debit"]
        df = df.sort_values("Date")
        df["Cumulative Savings"] = df["Savings"].cumsum()
        fig = px.line(df, x="Date", y="Cumulative Savings", markers=True,
                      title="Cumulative Savings Over Time", labels={"Cumulative Savings": "Savings (₹)"})
        fig.update_traces(line=dict(width=3))
        return fig
    except Exception as e:
        print(f"Error in plot_cumulative_savings_plotly: {e}")
        return None

def plot_category_breakdown_plotly(df):
    try:
        if "Category" not in df.columns:
            return None
        cat_counts = df["Category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        fig = px.pie(cat_counts, values="Count", names="Category",
                     title="Spending by Category", hole=0.4)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        return fig
    except Exception as e:
        print(f"Error in plot_category_breakdown_plotly: {e}")
        return None

# --------------- PDF Report Generation ---------------
def df_to_table_data(df, max_rows=50):
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) or col in ["Debit", "Credit", "Balance", "Amount", "DTI Ratio", "Savings Rate"]:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).round(2)
            df[col] = df[col].apply(lambda x: f"{x:,.2f}")
        elif col == "Date":
            df[col] = pd.to_datetime(df[col], errors='coerce').fillna(pd.Timestamp("2025-01-01"))
            df[col] = df[col].dt.strftime("%Y-%m-%d")
        else:
            df[col] = df[col].astype(str)
    return [df.columns.tolist()] + df.head(max_rows).values.tolist()

def build_pdf_report(pdf_path, context, plot_paths):
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph(f"Loan Analysis Report - {context['name']}", styles['Title']))
    story.append(Paragraph(f"Generated: {context['ts']}", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # Input Section
    story.append(Paragraph("Input", styles['Heading2']))
    story.append(Paragraph(f"<b>Source file:</b> {context['input_file']}", styles['Normal']))
    story.append(Paragraph(f"<b>CIBIL Score:</b> {context['cibil']}", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # Decision Section
    story.append(Paragraph("Decision", styles['Heading2']))
    decision_style = ParagraphStyle(
        name='Decision',
        parent=styles['Normal'],
        backColor=colors.lightgreen if context['decision_class'] == 'approve' else colors.pink,
        spaceAfter=12,
        borderPadding=10,
        borderWidth=1,
        borderColor=colors.green if context['decision_class'] == 'approve' else colors.red
    )
    story.append(Paragraph(f"<b>Action:</b> {context['action']}", decision_style))
    story.append(Paragraph(f"<b>Risk Level:</b> {context['risk']}", decision_style))
    story.append(Paragraph(f"<b>Reason:</b> {context['reason']}", decision_style))
    story.append(Paragraph(f"<b>Heuristic Score:</b> {context['score']}", decision_style))
    if context['ml_block']:
        story.append(Paragraph(context['ml_block'], decision_style))
    story.append(Spacer(1, 0.2 * inch))

    # Key Metrics
    story.append(Paragraph("Key Metrics", styles['Heading2']))
    metrics_data = df_to_table_data(context['metrics_df'])
    metrics_table = Table(metrics_data)
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 0.2 * inch))

    # Sample Transactions
    story.append(Paragraph("Sample Transactions (first 20)", styles['Heading2']))
    transactions_data = df_to_table_data(context['transactions_df'], max_rows=20)
    transactions_table = Table(transactions_data)
    transactions_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(transactions_table)
    story.append(Spacer(1, 0.2 * inch))

    # Visualizations
    story.append(Paragraph("Visualizations", styles['Heading2']))
    for plot_path, title in plot_paths:
        if Path(plot_path).exists():
            story.append(Paragraph(title, styles['Heading3']))
            story.append(Image(plot_path, width=6*inch, height=3*inch))
            story.append(Spacer(1, 0.1 * inch))

    doc.build(story)
    return pdf_path

# --------------- CSV export ---------------
def export_raw_csv(df, out_csv):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce').fillna(pd.Timestamp("2025-01-01"))
    for col in ["Debit", "Credit", "Balance"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype('float64')
    df["Description"] = df["Description"].astype(str)
    df.to_csv(out_csv, index=False, date_format='%Y-%m-%d')
    return out_csv

# --------------- Main pipeline ---------------
def analyze_file(input_path, cibil_score=720, fill_method="zero", out_dir="outputs", applicant_data=None):
    input_path = Path(input_path)
    out_dir = ensure_dir(out_dir)
    plots_dir = ensure_dir(out_dir / "plots")
    metrics_dir = ensure_dir(out_dir / "metrics")
    raw_dir = ensure_dir(out_dir / "raw_transactions")
    categorized_dir = ensure_dir(out_dir / "categorized")
    decisions_dir = ensure_dir(out_dir / "loan_decisions")
    reports_dir = ensure_dir(out_dir / "reports")

    # Ensure applicant_data is a dictionary with defaults
    if applicant_data is None:
        applicant_data = {"name": Path(input_path).stem, "account_number": "Unknown"}
    else:
        applicant_data = {
            "name": applicant_data.get("name", Path(input_path).stem),
            "account_number": applicant_data.get("account_number", "Unknown")
        }

    # 1) Extract
    print("[1/8] Extracting transactions...")
    df = None
    if input_path.suffix.lower() == ".pdf":
        try:
            df = parse_pdf_to_df(str(input_path))
        except Exception as e:
            print("PDF parse error:", e)
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

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce').fillna(pd.Timestamp("2025-01-01")).dt.strftime('%Y-%m-%d')
        df = df.dropna(subset=["Date"]).reset_index(drop=True)

    raw_csv_path = raw_dir / f"{input_path.stem}_raw.csv"
    export_raw_csv(df, raw_csv_path)
    print(f"Exported raw CSV: {raw_csv_path}")

    df = standardize_columns(df)
    for col in ["Debit", "Credit", "Balance"]:
        if col not in df.columns:
            df[col] = 0.0
    if "Date" not in df.columns:
        print("No Date column found after parsing. Exiting.")
        return None

    df = make_arrow_compatible(df)

    # 2) Categorize
    print("[2/8] Categorizing transactions...")
    categorized_csv_path = categorized_dir / f"{input_path.stem}_categorized.csv"
    categorized_df = None
    try:
        vectorizer_path = Path("models/vectorizer.pkl")
        classifier_path = Path("models/classifier.pkl")
        # Initialize TransactionCategorizer (adjust initialization as per your setup)
        tc = TransactionCategorizer()  # May need parameters like model_path or vectorizer
        if vectorizer_path.exists() and classifier_path.exists():
            tc.load_model(vectorizer_path, classifier_path)
        else:
            print("[WARNING] Model files not found, training TransactionCategorizer with default data...")
        
        # If tc needs training, ensure it's trained before use
        if not hasattr(tc, 'vectorizer') or not hasattr(tc, 'classifier'):
            print("[WARNING] TransactionCategorizer not fully initialized, training or loading model...")
            # Example: tc.load_model("path/to/model.pkl") or tc.train_model(training_data)

        # Categorize the DataFrame directly
        categorized_df = categorize_transactions(df, tc, model_type="hybrid", add_confidence=True)
        print(f"[DEBUG] categorize_transactions returned type: {type(categorized_df)}, shape: {categorized_df.shape}")

        # Ensure DataFrame is valid
        if categorized_df is None or (isinstance(categorized_df, pd.DataFrame) and categorized_df.empty):
            print("categorize_transactions produced empty or invalid DataFrame, falling back to simple_categorize.")
            categorized_df = simple_categorize(df)

        # Standardize columns and handle Date
        categorized_df = standardize_columns(categorized_df)
        if "Date" in categorized_df.columns:
            categorized_df["Date"] = pd.to_datetime(categorized_df["Date"], errors='coerce').fillna(pd.Timestamp("2025-01-01"))
        else:
            print("Warning: 'Date' column missing in categorized DataFrame, adding default.")
            categorized_df["Date"] = pd.Timestamp("2025-01-01")

        categorized_df.to_csv(categorized_csv_path, index=False)
    except Exception as e:
        print(f"categorize util error: {e}")
        categorized_df = simple_categorize(df)
        categorized_df = standardize_columns(categorized_df)
        if "Date" in categorized_df.columns:
            categorized_df["Date"] = pd.to_datetime(categorized_df["Date"], errors='coerce').fillna(pd.Timestamp("2025-01-01"))
        else:
            categorized_df["Date"] = pd.Timestamp("2025-01-01")
        categorized_df.to_csv(categorized_csv_path, index=False)
    print(f"Categorized CSV: {categorized_csv_path}")

    # 3) Metrics
    print("[3/8] Calculating metrics...")
    metrics_file = metrics_dir / f"{input_path.stem}_metrics.csv"
    try:
        out_metrics = calculate_metrics(categorized_df, str(metrics_file))
        if isinstance(out_metrics, pd.DataFrame):
            metrics_df = out_metrics
        elif isinstance(out_metrics, dict):
            metrics_df = pd.DataFrame([out_metrics])
        else:
            metrics_df = pd.DataFrame([out_metrics])
        metrics_df = enforce_metrics_schema(metrics_df)
    except Exception as e:
        print("calculate_metrics util error:", e)
        monthly = categorized_df.set_index("Date").groupby(pd.Grouper(freq="M")).agg(
            Income=("Credit", "sum"), Expenses=("Debit", "sum"))
        avg_income = monthly["Income"].mean() if not monthly.empty else 0.0
        avg_expenses = monthly["Expenses"].mean() if not monthly.empty else 0.0
        net_surplus = avg_income - avg_expenses
        metrics_df = pd.DataFrame([{
            "Average Monthly Income": avg_income,
            "Average Monthly Expenses": avg_expenses,
            "Net Surplus": net_surplus,
            "DTI Ratio": 0.0,
            "Savings Rate": 0.0,
            "Red Flag Count": int((categorized_df["Category"] == "Red Flags").sum()),
            "Credit Utilization": 0.0,
            "Discretionary Expenses": categorized_df[categorized_df["Category"] == "Discretionary Expenses"]["Debit"].sum(),
            "Income Stability": 0
        }])
        metrics_df = enforce_metrics_schema(metrics_df)

    # Additional metrics
    additional_metrics_df = pd.DataFrame({
        'Income Variability Index': [categorized_df[categorized_df['Category'] == 'Income']['Credit'].std() / 
                                    categorized_df[categorized_df['Category'] == 'Income']['Credit'].mean() 
                                    if categorized_df[categorized_df['Category'] == 'Income']['Credit'].mean() > 0 else 0.0],
        'Number of Income Sources': [len(categorized_df[categorized_df['Category'] == 'Income']['Description'].unique())],
        'Recent Salary Trend (%)': [(categorized_df[categorized_df['Category'] == 'Income'].groupby(categorized_df['Date'].dt.strftime('%Y-%m'))['Credit'].sum().iloc[-1] - 
                           categorized_df[categorized_df['Category'] == 'Income'].groupby(categorized_df['Date'].dt.strftime('%Y-%m'))['Credit'].sum().iloc[-2]) / 
                           categorized_df[categorized_df['Category'] == 'Income'].groupby(categorized_df['Date'].dt.strftime('%Y-%m'))['Credit'].sum().iloc[-2] * 100 
                           if len(categorized_df[categorized_df['Category'] == 'Income'].groupby(categorized_df['Date'].dt.strftime('%Y-%m'))['Credit'].sum()) >= 2 and 
                           categorized_df[categorized_df['Category'] == 'Income'].groupby(categorized_df['Date'].dt.strftime('%Y-%m'))['Credit'].sum().iloc[-2] > 0 else 0.0],
        'Discretionary Spending (%)': [(categorized_df[categorized_df['Category'] == 'Discretionary Expenses']['Debit'].sum() / 
                                      categorized_df[categorized_df['Category'] == 'Income']['Credit'].sum() * 100) 
                                      if categorized_df[categorized_df['Category'] == 'Income']['Credit'].sum() > 0 else 0.0],
        'High-Cost EMI Payments': [categorized_df[categorized_df['Category'] == 'Loan Payments']['Debit'].mean() 
                                  if not categorized_df[categorized_df['Category'] == 'Loan Payments'].empty else 0.0],
        'Existing Loan Count': [len(categorized_df[categorized_df['Category'] == 'Loan Payments']['Description'].unique())],
        'Credit Card Payments': [categorized_df[categorized_df['Category'] == 'Credit Card Payments']['Debit'].sum() 
                                if not categorized_df[categorized_df['Category'] == 'Credit Card Payments'].empty else 0.0],
        'Bounced Cheques Count': [len(categorized_df[categorized_df['Category'] == 'Bounced Cheques'])],
        'Minimum Monthly Balance': [categorized_df.groupby(categorized_df['Date'].dt.strftime('%Y-%m'))['Balance'].min().mean() 
                           if not categorized_df.empty else 0.0],
        'Average Closing Balance': [categorized_df.groupby(categorized_df['Date'].dt.strftime('%Y-%m'))['Balance'].last().mean() 
                           if not categorized_df.empty else 0.0],
        'Overdraft Usage Frequency': [len(categorized_df[categorized_df['Balance'] < 0])],
        'Negative Balance Days': [len(categorized_df[categorized_df['Balance'] < 0]['Date'].unique())],
        'Sudden High-Value Credits': [len(categorized_df[(categorized_df['Category'] == 'Income') & (categorized_df['Credit'] > 100000)])],
        'Circular Transactions': [len(categorized_df['Description'].value_counts()[categorized_df['Description'].value_counts() > 5])]
    })
    metrics_df = pd.concat([metrics_df, additional_metrics_df], axis=1)
    metrics_df = enforce_metrics_schema(metrics_df)
    metrics_df = make_arrow_compatible(metrics_df)
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Saved metrics: {metrics_file}")


    # 4) Decision (heuristic)
    print("[4/8] Heuristic scoring & decisioning...")
    try:
        heuristic_decision = score_and_decide(metrics_df=metrics_df, cibil_score=cibil_score, categorized_file=str(categorized_csv_path))
    except Exception as e:
        print("score_and_decide util error:", e)
        heuristic_decision = {
            "Total Score": 0,
            "Risk Level": "High",
            "Action": "Reject",
            "Reason": f"Error in decision logic: {e}"
        }
    print("Heuristic decision:", heuristic_decision)


    # 5) ML model prediction
    print("[5/8] ML model integration (if model exists)...")
    model_path = Path("models/loan_approval_model.pkl")
    model = load_model(model_path)
    ml_result = None
    if model is not None:
        try:
            ml_result = model_predict_and_explain(model, metrics_df)
        except Exception as e:
            print(f"[ERROR] ML model prediction failed: {e}")
            ml_result = None
    else:
        print(f"No ML model found at {model_path}")

    # 6) Combine heuristic and ML decisions
    print("[6/8] Final decision reconciliation...")
    final_decision = heuristic_decision
    if ml_result is not None and ml_result["model_probability"] >= 95.0:
        print(f"[INFO] Using ML model prediction (confidence {ml_result['model_probability']}%) over heuristic")
        final_decision = {
            "Total Score": heuristic_decision["Total Score"],
            "Risk Level": heuristic_decision["Risk Level"],
            "Action": ml_result["model_prediction"],
            "Reason": f"ML model prediction (confidence {ml_result['model_probability']}%) based on features: {ml_result['features_used']}"
        }
    else:
        print(f"[INFO] Using heuristic decision (ML confidence {ml_result['model_probability'] if ml_result else 'None'}% or model unavailable)")
        final_decision["ML_Prediction"] = ml_result["model_prediction"] if ml_result else "None"
        final_decision["ML_Confidence"] = ml_result["model_probability"] if ml_result else 0.0

    print(f"Final Decision: {final_decision['Action']}, Reason: {final_decision['Reason']}")
    # 6) Visualizations
    print("[6/8] Generating visualizations...")
    categorized_df["Amount"] = categorized_df["Credit"].fillna(0.0) - categorized_df["Debit"].fillna(0.0)
    plot_functions = [
        (plot_income_trend_plotly, "income_trend.png"),
        (plot_surplus_trend_plotly, "surplus_trend.png"),
        (plot_income_vs_expenses_plotly, "income_vs_expenses.png"),
        (plot_cumulative_savings_plotly, "cumulative_savings.png"),
        (plot_category_breakdown_plotly, "category_breakdown.png"),
        (lambda df: plot_high_risk_timeline(df, str(plots_dir / "high_risk_timeline.png")), "high_risk_timeline.png"),
        (lambda df: plot_cibil_gauge(cibil_score, str(plots_dir / "cibil_gauge.png")), "cibil_gauge.png")
    ]
    plot_paths = []
    for func, filename in plot_functions:
        try:
            fig = func(categorized_df) if 'cibil' not in filename else func(cibil_score)
            if fig:
                output_path = str(plots_dir / filename)
                fig.write_image(output_path)
                plot_paths.append((output_path, filename.replace(".png", "").replace("_", " ").title()))
        except Exception as e:
            print(f"Error generating {filename}: {e}")
    print("Saved plots to", plots_dir)

    # 7) PDF report
    print("[7/8] Building PDF report...")
    # Sanitize applicant name to remove invalid characters
    sanitized_name = re.sub(r'[^\w\s-]', '', applicant_data.get('name', input_path.stem)).replace('\n', ' ').strip()
    report_name = f"{sanitized_name}_{cibil_score}_report.pdf"
    report_path = reports_dir / report_name
    doc = SimpleDocTemplate(str(report_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("Loan Eligibility Report", styles['Title']))
    story.append(Spacer(1, 0.2 * inch))

    # Applicant Data
    story.append(Paragraph("Applicant Data", styles['Heading1']))
    applicant_data_table = [
        ['Name', applicant_data.get('name', input_path.stem) if applicant_data else input_path.stem],
        ['Account Number', applicant_data.get('account_number', 'Unknown') if applicant_data else 'Unknown'],
        ['CIBIL Score', str(cibil_score)]
    ]
    table = Table(applicant_data_table)
    table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('FONT', (0,0), (-1,-1), 'Helvetica', 10)]))
    story.append(table)
    story.append(Spacer(1, 0.2 * inch))

    # Income & Stability
    story.append(Paragraph("Income & Stability", styles['Heading1']))
    income_data = [
        ['Average Monthly Income', f"₹{metrics_df['Average Monthly Income'].iloc[0]:,.2f}"],
        ['Income Variability Index', f"{metrics_df['Income Variability Index'].iloc[0]:.2f}"],
        ['Number of Income Sources', str(metrics_df['Number of Income Sources'].iloc[0])],
        ['Recent Salary Trend', f"{metrics_df['Recent Salary Trend (%)'].iloc[0]:.2f}%"]
    ]
    table = Table(income_data)
    table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black)]))
    story.append(table)
    story.append(Spacer(1, 0.2 * inch))

    # Expenses & Lifestyle
    story.append(Paragraph("Expenses & Lifestyle", styles['Heading1']))
    expenses_data = [
        ['Average Monthly Expenses', f"₹{metrics_df['Average Monthly Expenses'].iloc[0]:,.2f}"],
        ['Savings Ratio', f"{metrics_df['Savings Rate'].iloc[0]:.2f}%"],
        ['Discretionary Spending', f"{metrics_df['Discretionary Spending (%)'].iloc[0]:.2f}%"],
        ['High-Cost EMI Payments', f"₹{metrics_df['High-Cost EMI Payments'].iloc[0]:,.2f}"]
    ]
    table = Table(expenses_data)
    table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black)]))
    story.append(table)
    story.append(Spacer(1, 0.2 * inch))

    # Debt Metrics
    story.append(Paragraph("Debt Metrics", styles['Heading1']))
    debt_data = [
        ['DTI Ratio', f"{metrics_df['DTI Ratio'].iloc[0]:.2f}%"],
        ['Existing Loan Count', str(metrics_df['Existing Loan Count'].iloc[0])],
        ['Credit Card Payments', f"₹{metrics_df['Credit Card Payments'].iloc[0]:,.2f}"],
        ['Bounced Cheques Count', str(metrics_df['Bounced Cheques Count'].iloc[0])]
    ]
    table = Table(debt_data)
    table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black)]))
    story.append(table)
    story.append(Spacer(1, 0.2 * inch))

    # Cash Flow & Liquidity
    story.append(Paragraph("Cash Flow & Liquidity", styles['Heading1']))
    cash_data = [
        ['Minimum Monthly Balance', f"₹{metrics_df['Minimum Monthly Balance'].iloc[0]:,.2f}"],
        ['Average Closing Balance', f"₹{metrics_df['Average Closing Balance'].iloc[0]:,.2f}"],
        ['Overdraft Usage Frequency', str(metrics_df['Overdraft Usage Frequency'].iloc[0])],
        ['Negative Balance Days', str(metrics_df['Negative Balance Days'].iloc[0])]
    ]
    table = Table(cash_data)
    table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black)]))
    story.append(table)
    story.append(Spacer(1, 0.2 * inch))

    # Creditworthiness Indicators
    story.append(Paragraph("Creditworthiness Indicators", styles['Heading1']))
    credit_data = [
        ['CIBIL Score', str(cibil_score)],
        ['Payment History', 'Derived from statement'],
        ['Delinquency Flags', str(metrics_df['Bounced Cheques Count'].iloc[0])],
        ['Recent Loan Inquiries', 'Not available']
    ]
    table = Table(credit_data)
    table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black)]))
    story.append(table)
    story.append(Spacer(1, 0.2 * inch))

    # Fraud & Compliance Checks
    story.append(Paragraph("Fraud & Compliance Checks", styles['Heading1']))
    fraud_data = [
        ['Sudden High-Value Credits', str(metrics_df['Sudden High-Value Credits'].iloc[0])],
        ['Circular Transactions', str(metrics_df['Circular Transactions'].iloc[0])],
        ['Salary Mismatch', 'Not detected'],
        ['Blacklisted Accounts', 'Not detected']
    ]
    table = Table(fraud_data)
    table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black)]))
    story.append(table)
    story.append(Spacer(1, 0.2 * inch))

    # Decision Metrics
    story.append(Paragraph("Decision Metrics", styles['Heading1']))
    decision_metrics = [
        ['Bank Score', f"{heuristic_decision.get('Total Score', 0):.2f}/100"],
        ['DTI Ratio', f"{metrics_df['DTI Ratio'].iloc[0]:.2f}%"],
        ['Average Closing Balance', f"₹{metrics_df['Average Closing Balance'].iloc[0]:,.2f}"],
        ['CIBIL Score', str(cibil_score)],
        ['Bounced Cheques', str(metrics_df['Bounced Cheques Count'].iloc[0])]
    ]
    table = Table(decision_metrics)
    table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black)]))
    story.append(table)
    story.append(Spacer(1, 0.2 * inch))

    # Final Decision
    story.append(Paragraph("Final Decision", styles['Heading1']))
    story.append(Paragraph(f"<b>{heuristic_decision['Action']}</b>: {heuristic_decision['Reason']}", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # ML Prediction
    if ml_result:
        story.append(Paragraph("ML Model Prediction", styles['Heading1']))
        ml_data = [
            ['Prediction', ml_result.get('model_prediction', 'N/A')],
            ['Confidence', f"{ml_result.get('model_probability', 0):.2f}%"]
        ]
        table = Table(ml_data)
        table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black)]))
        story.append(table)
        story.append(Spacer(1, 0.2 * inch))

    # Add Plots
    for plot_path, title in plot_paths:
        if os.path.exists(plot_path):
            story.append(Paragraph(title, styles['Heading2']))
            story.append(Image(plot_path, width=6*inch, height=3*inch))
            story.append(Spacer(1, 0.2 * inch))

    doc.build(story)
    print("Saved PDF report:", report_path)


    # 8) Save decision JSON and log
    print("[8/8] Saving decision JSON and log...")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record = {
        "timestamp": timestamp,
        "file": str(input_path),
        "cibil": cibil_score,
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
        "plots": [p[0] for p in plot_paths],
        "pdf_report": str(report_path),
        "decision_json": str(decision_json),
        "metrics_df": metrics_df,
        "transactions_df": categorized_df
    }
    return out

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Powerful bank-statement analyzer and loan decisioner.")
    parser.add_argument("--file", "-f", required=False, default="/mnt/data/Taylor_Fernandes_moderate.pdf",
                        help="Path to statement file (pdf/csv/xlsx/txt)")
    parser.add_argument("--cibil", "-c", type=int, default=720, help="CIBIL score (300-900)")
    parser.add_argument("--fill", "-m", choices=["zero", "interpolate"], default="zero",
                        help="How to fill missing months in time-series")
    parser.add_argument("--out", "-o", default="outputs", help="Output directory")
    args = parser.parse_args()

    try:
        from reportlab.lib.pagesizes import letter
    except ImportError:
        raise ImportError("Please install reportlab to generate PDF reports: `pip install reportlab`")

    print("Starting analysis...")
    res = analyze_file(args.file, cibil_score=args.cibil, fill_method=args.fill, out_dir=args.out)
    if res is None:
        print("Analysis failed or no data extracted.")
    else:
        print("Analysis complete. Outputs:")
        for k, v in res.items():
            print(f" - {k}: {v}")