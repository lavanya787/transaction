import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import joblib
import plotly.express as px
import chardet
import fitz  # PyMuPDF
import docx2txt
import os
from datetime import datetime
import spacy
import re
from cryptography.fernet import Fernet
from pathlib import Path
import tempfile
import logging
import base64
import pyarrow as pa

# Import utility functions (assuming these are in the utils folder)
from utils.financial_metrics import calculate_metrics, identify_high_value_transactions, recurring_transactions, clean_currency_column, enforce_schema
from utils.score_bank_statements import score_and_decide
from utils.visualizer import (
    analyze_file,
    plot_income_trend_plotly,
    plot_surplus_trend_plotly,
    plot_income_vs_expenses_plotly,
    plot_cumulative_savings_plotly,
    plot_category_breakdown_plotly
)
from utils.text_parser import parse_bank_statement_text
from utils.extract_transactions import extract_df_from_scanned_pdf, extract_transactions
from utils.categorize_transactions import TransactionCategorizer, categorize_transactions
from utils.preprocess_transactions import preprocess_transactions, make_arrow_compatible

# Setup logging
logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define base directory dynamically
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
CLEANED_CSV_DIR = BASE_DIR / "cleaned_csvs"
CATEGORIZED_CSV_DIR = BASE_DIR / "categorized_csvs"
OUTPUT_DIR = BASE_DIR / "outputs"
TEMP_DIR = BASE_DIR / "temp"

# Ensure directories exist
for directory in [MODEL_DIR, CLEANED_CSV_DIR, CATEGORIZED_CSV_DIR, OUTPUT_DIR, TEMP_DIR]:
    directory.mkdir(exist_ok=True)

# Load pre-trained models
try:
    vectorizer = joblib.load(MODEL_DIR / "vectorizer.pkl")
    classifier = joblib.load(MODEL_DIR / "classifier.pkl")
except Exception as e:
    logger.error(f"Failed to load pre-trained models: {e}")
    st.error(f"❌ Failed to load pre-trained models: {e}")
    vectorizer, classifier = None, None

# Cache model loading with explicit file checking
def load_models():
    """
    Load PaddleX models for OCR processing.
    
    Returns:
        dict: Dictionary of model names to model objects
    """
    try:
        # Placeholder for actual PaddleX model loading
        # Replace with your actual model loading logic
        models = {
            "PP-LCNet_x1_0_doc_ori": None,
            "UVDoc": None,
            "PP-LCNet_x1_0_textline_ori": None,
            "PP-OCRv5_server_det": None,
            "PP-OCRv5_server_rec": None
        }
        logger.info("Loaded PaddleX models: %s", list(models.keys()))
        return models
    except Exception as e:
        logger.error("Failed to load PaddleX models: %s", e)
        return None

# Load spaCy for categorization
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except Exception as e:
    logger.error(f"Failed to load spaCy model: {e}")
    nlp = None

# --- Category Rules with Indian Banking Context ---
category_rules = {
    "Income": ["salary", "bonus", "freelance", "investment", "neft.*cr", "imps.*cr", "upi.*cr", r"\[Salary\]", r"\[Bonus\]", r"\[Freelance\]"],
    "Fixed Expenses": ["rent", "emi", "insurance", "electricity", "phone", "bill", "loan", "paytm.*bill", r"\[Rent\]", r"\[Electricity\]"],
    "Discretionary Expenses": ["shopping", "dining", "entertainment", "travel", "amazon", "zomato", "swiggy", "flipkart", r"\[Shopping\]", r"\[Dining\]"],
    "Savings": ["fd", "rd", "mutual fund", "deposit", "sip", "ppf", "nps", r"\[FD\]", r"\[RD\]"],
    "Red Flags": ["overdraft", "bounce", "insufficient funds", r"cash.*\d{4,}", r"upi.*\d{5,}.*cash"],
    "Credit Card Payments": ["credit card", "cc payment", "visa", "mastercard"],
    "Bounced Cheques": ["bounce", "insufficient funds", "cheque return"],
    "Loan Payments": ["emi", "loan repayment"]
}


# --- Rule-Based Categorization ---
def rule_based_categorize(description):
    # Ensure description is a string and handle NaN/None
    description = str(description) if pd.notna(description) else "Unknown"
    if nlp:
        doc = nlp(description.lower())
        for category, patterns in category_rules.items():
            for pattern in patterns:
                if re.search(pattern, description.lower()):
                    return category
    return "Other"

# --- Standardize Column Names ---
def standardize_columns(df):
    if not isinstance(df, pd.DataFrame):
        logger.error("standardize_columns received non-DataFrame input")
        raise ValueError("Input to standardize_columns must be a pandas DataFrame")
    df.columns = df.columns.astype(str)
    column_mapping = {
        'debit': 'Debit', 'DEBIT': 'Debit', 'withdrawal': 'Debit',
        'credit': 'Credit', 'CREDIT': 'Credit', 'deposit': 'Credit',
        'balance': 'Balance', 'BALANCE': 'Balance',
        'date': 'Date', 'DATE': 'Date', 'transaction date': 'Date',
        'description': 'Description', 'DESCRIPTION': 'Description', 'particulars': 'Description', 'narration': 'Description',
        'category': 'Category', 'CATEGORY': 'Category'
    }
    df.columns = [column_mapping.get(col.lower(), col) for col in df.columns]
    # Ensure Arrow-compatible types
    for col in df.columns:
        if col in ['Debit', 'Credit', 'Balance']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype('float64')
        elif col == 'Date':
            df[col] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%Y-%m-%d')
        elif col == 'Description':
            df[col] = df[col].apply(lambda x: str(x) if pd.notna(x) else 'Unknown')
        else:
            df[col] = df[col].apply(lambda x: str(x) if pd.notna(x) else 'Unknown')
    logger.info(f"Standardized DataFrame columns: {df.dtypes}")
    logger.info(f"[DEBUG] Standardized Description sample: {df['Description'].head().tolist()}")
    return df

def extract_applicant_info(text):
    """
    Extract applicant name and account number from text using regex patterns.
    Returns: (applicant_name, account_number)
    """
    name = "Unknown"
    account_number = "Unknown"
    
    # Ensure text is a string
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
        logger.warning(f"Non-string text input to extract_applicant_info: {type(text)}")
    
    # Regex patterns for name and account number
    name_patterns = [
        r"Account Holder\s*:\s*([A-Za-z\s]+)",
        r"Customer Name\s*:\s*([A-Za-z\s]+)",
        r"Name\s*:\s*([A-Za-z\s]+)"
    ]
    account_patterns = [
        r"A/C No\s*:\s*(\d{9,16})",
        r"Account Number\s*:\s*(\d{9,16})",
        r"Acc\s*No\s*:\s*(\d{9,16})"
    ]
    
    # Extract name
    for pattern in name_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            break
    
    # Extract account number
    for pattern in account_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            account_number = match.group(1).strip()
            break
    
    logger.info(f"Extracted Applicant Name: {name}, Account Number: {account_number}")
    return name, account_number

# Additional metrics calculation
def calculate_additional_metrics(categorized_df):
    metrics = {}
    
    # Income & Stability
    income_df = categorized_df[categorized_df['Category'] == 'Income']
    income_df['Date'] = pd.to_datetime(income_df['Date'], errors='coerce')
    income_df['YearMonth'] = income_df['Date'].dt.to_period('M').astype(str)
    
    # Average Monthly Income (last 6 months)
    recent_six_months = income_df['YearMonth'].unique()[-6:]
    monthly_income = income_df[income_df['YearMonth'].isin(recent_six_months)].groupby('YearMonth')['Credit'].sum()
    metrics['Average Monthly Income'] = monthly_income.mean() if not monthly_income.empty else 0.0
    
    # Income Variability Index (std dev / mean)
    metrics['Income Variability Index'] = monthly_income.std() / monthly_income.mean() if monthly_income.mean() > 0 else 0.0
    
    # Number of Income Sources
    metrics['Number of Income Sources'] = len(income_df['Description'].unique())
    
    # Recent Salary Trend
    if len(monthly_income) >= 2:
        trend = (monthly_income.iloc[-1] - monthly_income.iloc[-2]) / monthly_income.iloc[-2] * 100 if monthly_income.iloc[-2] > 0 else 0.0
        metrics['Recent Salary Trend (%)'] = trend
    else:
        metrics['Recent Salary Trend (%)'] = 0.0
    
    # Expenses & Lifestyle
    expenses_df = categorized_df[categorized_df['Category'].isin(['Fixed Expenses', 'Discretionary Expenses', 'Loan Payments'])]
    monthly_expenses = expenses_df.groupby('YearMonth')['Debit'].sum()
    metrics['Average Monthly Expenses'] = monthly_expenses.mean() if not monthly_expenses.empty else 0.0
    
    # Savings Ratio
    metrics['Savings Ratio (%)'] = ((metrics['Average Monthly Income'] - metrics['Average Monthly Expenses']) / 
                                   metrics['Average Monthly Income'] * 100) if metrics['Average Monthly Income'] > 0 else 0.0
    
    # Discretionary Spending %
    discretionary_df = categorized_df[categorized_df['Category'] == 'Discretionary Expenses']
    monthly_discretionary = discretionary_df.groupby('YearMonth')['Debit'].sum()
    metrics['Discretionary Spending (%)'] = (monthly_discretionary.mean() / metrics['Average Monthly Income'] * 100) if metrics['Average Monthly Income'] > 0 else 0.0
    
    # High-Cost EMI
    loan_df = categorized_df[categorized_df['Category'] == 'Loan Payments']
    monthly_emi = loan_df.groupby('YearMonth')['Debit'].sum()
    metrics['High-Cost EMI Payments'] = monthly_emi.mean() if not monthly_emi.empty else 0.0
    
    # Debt Metrics
    metrics['DTI Ratio (%)'] = (metrics['High-Cost EMI Payments'] / metrics['Average Monthly Income'] * 100) if metrics['Average Monthly Income'] > 0 else 0.0
    metrics['Existing Loan Count'] = len(loan_df['Description'].unique())
    
    # Credit Card Payments
    cc_df = categorized_df[categorized_df['Category'] == 'Credit Card Payments']
    metrics['Credit Card Payments'] = cc_df['Debit'].sum() if not cc_df.empty else 0.0
    
    # Bounced Cheques
    bounce_df = categorized_df[categorized_df['Category'] == 'Bounced Cheques']
    metrics['Bounced Cheques Count'] = len(bounce_df)
    
    # Cash Flow & Liquidity
    metrics['Minimum Monthly Balance'] = categorized_df.groupby('YearMonth')['Balance'].min().mean() if not categorized_df.empty else 0.0
    metrics['Average Closing Balance'] = categorized_df.groupby('YearMonth')['Balance'].last().mean() if not categorized_df.empty else 0.0
    metrics['Overdraft Usage Frequency'] = len(categorized_df[categorized_df['Balance'] < 0])
    metrics['Negative Balance Days'] = len(categorized_df[categorized_df['Balance'] < 0]['Date'].unique())
    
    # Fraud & Compliance
    high_value_credits = categorized_df[(categorized_df['Category'] == 'Income') & (categorized_df['Credit'] > 100000)]
    metrics['Sudden High-Value Credits'] = len(high_value_credits)
    
    # Circular Transactions (simplified: repeated transfers between same entities)
    desc_counts = categorized_df['Description'].value_counts()
    metrics['Circular Transactions'] = len(desc_counts[desc_counts > 5])
    
    return pd.DataFrame([metrics])

# --- Page Setup ---
st.set_page_config(page_title="📊 Loan Scoring Dashboard", layout="wide")
st.title("🏦 AI-Powered Loan Eligibility Analyzer")
st.markdown("Boost lending decisions with smarter transaction scoring and real-time risk assessment per RBI guidelines.")

# --- IMPROVED SIDEBAR ORGANIZATION ---
st.sidebar.markdown("# 📋 Loan Application")
st.sidebar.markdown("---")

# === LOAN DETAILS ===
with st.sidebar.container():
    st.markdown("## 💰 Loan Requirements")
    
    cibil_score = st.number_input(#slider
    "Enter your Credit Score",
    min_value=300,
    max_value=900,
    value=720,   # default
    step=1,
    help="Your current credit score (300-900)"
)
    
# ✅ Auto-validate the entered value
if cibil_score < 300:
    st.warning("Credit score too low. Resetting to minimum 300.")
    credit_score = 300
elif cibil_score > 900:
    st.warning("Credit score too high. Resetting to maximum 900.")
    credit_score = 900
st.write(f"Your entered cibil score is: {cibil_score}")  
st.sidebar.markdown("---")

# === DOCUMENT UPLOADS ===
with st.sidebar.container():
    st.markdown("## 📄 Document Uploads")
    
    # Bank Statement Upload (Most Important)
    st.markdown("### 🏦 Bank Statement*")
    uploaded_file = st.file_uploader("Upload Bank Statement", 
                                    type=["csv", "pdf", "txt", "doc", "docx"],
                                    help="Last 6 months bank statement (required)")
    
    if uploaded_file is not None:
        input_path = os.path.join("uploads", uploaded_file.name)
        os.makedirs("uploads", exist_ok=True)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"✅ File uploaded: {uploaded_file.name}")
        file_size = len(uploaded_file.getvalue()) / (1024*1024)  # Size in MB
        st.caption(f"📁 File size: {file_size:.1f} MB")
    
    # Show upload status
    uploaded_docs = sum([1 for doc in [uploaded_file] if doc is not None])
    st.caption(f"📊 Documents uploaded: {uploaded_docs}")

# === FOOTER ===
st.markdown("---")
st.markdown("### ℹ️ Need Help?")

with st.expander("Supported Formats"):
    st.markdown("""
    **📊 Bank Statements:**
    - CSV, PDF, TXT, DOC, DOCX  
    - Must contain: Date, Description, Debit, Credit, Balance  

    **📄 Documents:**
    - PDF, JPG, PNG, JPEG  
    - Max size: 200MB per file  

    **🎯 Training Data:**
    - CSV format with Description, Category columns  
    """)

with st.expander("💡 Tips & Guidelines"):
    st.markdown("""
    **🎯 For Best Results:**
    - Upload 6+ months bank statements  
    - Ensure CIBIL score is current  
    - Provide accurate income details  

    **📊 CIBIL Score Guide:**
    - 750-900: Excellent (High approval chances)  
    - 650-749: Good (Moderate rates)  
    - 550-649: Fair (Higher rates)  
    - Below 550: Poor (Manual review)  

    **💰 EMI Guidelines:**
    - Keep total EMI < 40% of income  
    - Include existing loan EMIs  
    - Consider future financial commitments  
    """)

# --- Encryption Setup ---
key = Fernet.generate_key()
cipher_suite = Fernet(key)

# Cache file processing
@st.cache_data
def process_file(file_content, file_type, file_name, models=None):
    """
    Process uploaded file and return DataFrame and extracted text (if applicable).
    
    Args:
        file_content: Bytes content of the uploaded file
        file_type: File extension (csv, pdf, txt, doc, docx)
        file_name: Name of the file
        models: PaddleX models for OCR (optional)
    
    Returns:
        tuple: (DataFrame, error message, extracted text)
    """
    df = None
    error = None
    extracted_text = None
    applicant_name = "Unknown"
    account_number = "Unknown"
    
    try:
        if file_type == "csv":
            encoding = chardet.detect(file_content)["encoding"] or "latin-1"
            text = file_content.decode(encoding, errors="replace")
            applicant_name, account_number = extract_applicant_info(text)
            df = pd.read_csv(io.StringIO(text))
            df = standardize_columns(df)
            required_cols = {"Date", "Description", "Debit", "Credit", "Balance"}
            if not required_cols.issubset(df.columns):
                error = f"CSV missing columns: {required_cols - set(df.columns)}"
                logger.error(f"CSV missing columns in {file_name}: {error}")
                return None, error, None, applicant_name, account_number
            if df.empty or df.isnull().all().all():
                error = "CSV is empty or contains only null values"
                logger.error(f"Empty or invalid CSV {file_name}")
                return None, error, None, applicant_name, account_number
            for col in ["Debit", "Credit", "Balance"]:
                df[col] = clean_currency_column(pd.DataFrame({col: df[col]}))[col]
            logger.info(f"Successfully processed CSV {file_name}")
            logger.info(f"[DEBUG] CSV DataFrame types: {df.dtypes}")
            logger.info(f"[DEBUG] CSV Description sample: {df['Description'].head().tolist()}")

        elif file_type == "pdf":
            if not file_content:
                error = "Uploaded PDF is empty"
                logger.error(f"Empty PDF uploaded: {file_name}")
                return None, error, None, applicant_name, account_number
            with fitz.open(stream=file_content, filetype="pdf") as doc:
                if doc.page_count == 0:
                    error = "PDF has no pages"
                    logger.error(f"PDF has no pages: {file_name}")
                    return None, error, None, applicant_name, account_number
                extracted_text = "\n".join([page.get_text() for page in doc])
                applicant_name, account_number = extract_applicant_info(extracted_text)
                if extracted_text.strip():
                    logger.info(f"Extracted text from PDF {file_name}: {extracted_text[:1000]}")
                    df = parse_bank_statement_text(extracted_text)
                    df = standardize_columns(df)
                else:
                    logger.info(f"No text extracted from PDF {file_name}, switching to OCR mode")
                    if models:
                        df = extract_df_from_scanned_pdf(file_content, models=models)
                        df = standardize_columns(df)
                        extracted_text = "\n".join([page.get_text() for page in fitz.open(stream=file_content, filetype="pdf")])
                        applicant_name, account_number = extract_applicant_info(extracted_text)
                    else:
                        error = "No text extracted and no OCR models provided"
                        logger.error(error)
                        return None, error, extracted_text, applicant_name, account_number
            for col in ["Debit", "Credit", "Balance"]:
                if df is not None and col in df.columns:
                    df[col] = clean_currency_column(pd.DataFrame({col: df[col]}))[col]
            logger.info(f"Successfully processed PDF {file_name}")
            logger.info(f"[DEBUG] PDF DataFrame types: {df.dtypes if df is not None else 'None'}")
            logger.info(f"[DEBUG] PDF Description sample: {df['Description'].head().tolist() if df is not None else 'None'}")

        elif file_type == "txt":
            encoding = chardet.detect(file_content)["encoding"] or "latin-1"
            extracted_text = file_content.decode(encoding, errors="replace")
            applicant_name, account_number = extract_applicant_info(extracted_text)
            logger.info(f"Extracted text from TXT {file_name}: {extracted_text[:1000]}")
            df = parse_bank_statement_text(extracted_text)
            df = standardize_columns(df)
            for col in ["Debit", "Credit", "Balance"]:
                if df is not None and col in df.columns:
                    df[col] = clean_currency_column(pd.DataFrame({col: df[col]}))[col]
            logger.info(f"Successfully processed TXT {file_name}")
            logger.info(f"[DEBUG] TXT DataFrame types: {df.dtypes if df is not None else 'None'}")
            logger.info(f"[DEBUG] TXT Description sample: {df['Description'].head().tolist() if df is not None else 'None'}")

        elif file_type in ["doc", "docx"]:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                tmp.write(file_content)
                extracted_text = docx2txt.process(tmp.name)
            applicant_name, account_number = extract_applicant_info(extracted_text)
            logger.info(f"Extracted text from DOC {file_name}: {extracted_text[:1000]}")
            df = parse_bank_statement_text(extracted_text)
            df = standardize_columns(df)
            for col in ["Debit", "Credit", "Balance"]:
                if df is not None and col in df.columns:
                    df[col] = clean_currency_column(pd.DataFrame({col: df[col]}))[col]
            logger.info(f"Successfully processed DOC {file_name}")
            logger.info(f"[DEBUG] DOC DataFrame types: {df.dtypes if df is not None else 'None'}")
            logger.info(f"[DEBUG] DOC Description sample: {df['Description'].head().tolist() if df is not None else 'None'}")

        if df is not None and not df.empty:
            required_cols = {"Date", "Description", "Debit", "Credit", "Balance"}
            if not required_cols.issubset(df.columns):
                error = f"Parsed DataFrame missing columns: {required_cols - set(df.columns)}"
                logger.error(f"Parsed DataFrame missing columns in {file_name}: {error}")
                return None, error, extracted_text, applicant_name, account_number
            for col in ["Debit", "Credit", "Balance"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype("float64")
            df["Date"] = pd.to_datetime(df[col], errors="coerce").dt.strftime('%Y-%m-%d')
            df["Description"] = df["Description"].apply(lambda x: str(x) if pd.notna(x) else 'Unknown')
            df = df.dropna(subset=["Date"]).reset_index(drop=True)
            if df.empty:
                error = "All rows dropped due to invalid dates"
                logger.error(f"{error} in {file_name}")
                return None, error, extracted_text, applicant_name, account_number
            logger.info(f"Validated DataFrame for {file_name}: {df.dtypes}")
            logger.info(f"[DEBUG] Validated Description sample: {df['Description'].head().tolist()}")
        else:
            error = "No valid transactions extracted from file"
            logger.error(f"No valid transactions extracted from {file_name}")
            return None, error, extracted_text, applicant_name, account_number

        return df, None, extracted_text, applicant_name, account_number

    except Exception as e:
        error = f"Error processing file: {e}"
        logger.error(f"Error processing {file_name}: {e}")
        return None, error, extracted_text, applicant_name, account_number

# Main processing
if uploaded_file is not None:
    file_type = uploaded_file.name.split(".")[-1].lower()
    file_name = uploaded_file.name.replace(f".{file_type}", ".csv")
    file_content = uploaded_file.read()
    
    # Show processing status
    st.header("📊 Loan Analysis in Progress")
    
    # Create progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Extract raw transactions
    with st.spinner("Extracting transactions and applicant details... ⏳"):
        df, error, extracted_text, applicant_name, account_number = process_file(file_content, file_type, file_name, models=load_models())
        progress_bar.progress(20)

    # Display extracted applicant details
    st.subheader("🧑‍💼 Extracted Applicant Details")
    if applicant_name != "Unknown" and account_number != "Unknown":
        st.write(f"**Applicant Name**: {applicant_name}")
        st.write(f"**Account Number**: {account_number}")
    else:
        st.warning("⚠️ Could not extract applicant name or account number from the bank statement. Using defaults.")
        st.write(f"**Applicant Name**: {applicant_name}")
        st.write(f"**Account Number**: {account_number}")

    # Display extracted text for PDF/TXT/DOC
    if extracted_text and file_type in ["pdf", "txt", "doc", "docx"]:
        st.info(f"📄 Extracted Text from {file_type.upper()} (Preview)")
        st.text_area(f"{file_type.upper()} Extract", extracted_text[:1000], key=f"{file_type}_extract_preview")
        if file_type == "pdf" and not extracted_text.strip():
            st.info("📷 OCR Mode Activated for Scanned PDF")
    
    # Display debug info
    if df is not None and not df.empty:
        st.write("[DEBUG] Extracted DataFrame types:", df.dtypes)
        st.write("[DEBUG] Extracted Description sample:", df["Description"].head().tolist())
        try:
            st.dataframe(make_arrow_compatible(df.head(5)), use_container_width=True)
        except Exception as e:
            logger.error(f"Arrow serialization failed for extracted DataFrame: {e}")
            st.warning(f"⚠️ Failed to display DataFrame: {e}. Converting to strings.")
            df = df.apply(lambda x: x.astype(str) if x.dtype == "object" else x)
            st.dataframe(df.head(5), use_container_width=True)    
    if error:
        st.error(f"❌ {error}")
        with open(OUTPUT_DIR / "extraction_errors.log", "a") as f:
            f.write(f"{datetime.now()}: {error} in {file_name}\n")
    elif df is not None and not df.empty:
        st.success("✅ Transactions extracted successfully")
        progress_bar.progress(40)
        
        def needs_encryption(desc):
            pattern = r"[A-Z0-9]+/[A-Za-z]+/\d+/[a-z]+@ok[a-z]+/UPI/\d+/UPI/BRANCH.*"
            return not bool(re.match(pattern, str(desc)))
        
        # Encrypt descriptions
        df["Description"] = df["Description"].apply(
            lambda x: cipher_suite.encrypt(str(x).encode()).decode() if pd.notna(x) and needs_encryption(x) else str(x)
        )
        df.to_csv(CLEANED_CSV_DIR / file_name, index=False)
        logger.info(f"[DEBUG] After encryption Description sample: {df['Description'].head().tolist()}")

        # Step 2: Preprocess transactions
        with st.spinner("Preprocessing transactions... 🛠️"):
            try:
                df = preprocess_transactions(df)
                if not isinstance(df, pd.DataFrame):
                    logger.error("preprocess_transactions returned non-DataFrame")
                    raise ValueError("preprocess_transactions must return a pandas DataFrame")
                df["Description"] = df["Description"].apply(lambda x: str(x) if pd.notna(x) else 'Unknown')
                logger.info(f"[DEBUG] After preprocess_transactions Description sample: {df['Description'].head().tolist()}")
                logger.info(f"[DEBUG] After preprocess_transactions types: {df.dtypes}")
            except Exception as e:
                logger.error(f"Preprocessing failed: {e}")
                st.error(f"❌ Preprocessing failed: {e}")
                df["Description"] = df["Description"].apply(lambda x: str(x) if pd.notna(x) else 'Unknown')
            progress_bar.progress(50)

        st.subheader("📄 Cleaned Transactions")
        try:
            st.dataframe(make_arrow_compatible(df.head(10)), use_container_width=True)
        except Exception as e:
            logger.error(f"Arrow serialization failed for cleaned transactions: {e}")
            st.warning(f"⚠️ Failed to display cleaned transactions: {e}. Converting to strings.")
            df = df.apply(lambda x: x.astype(str) if x.dtype == "object" else x)
            st.dataframe(df.head(10), use_container_width=True)
    
        # Decrypt only if actually encrypted
        def try_decrypt(val):
            if pd.isna(val) or not isinstance(val, str):
                logger.warning(f"Non-string value in try_decrypt: {val} (type: {type(val)})")
                return str(val) if pd.notna(val) else "Unknown"
            try:
                if needs_encryption(val):
                    return val
                decrypted = cipher_suite.decrypt(val.encode()).decode()
                logger.info(f"Decrypted value: {decrypted[:50]}...")
                return decrypted
            except Exception as e:
                logger.warning(f"Decryption failed for value: {val[:50]}..., error: {e}")
                return str(val)

        df["Description"] = df["Description"].apply(try_decrypt)
        df["Description"] = df["Description"].apply(lambda x: str(x) if pd.notna(x) else 'Unknown')
        logger.info(f"[DEBUG] After decryption Description sample: {df['Description'].head().tolist()}")
        logger.info(f"[DEBUG] After decryption types: {df.dtypes}")

        try:
            st.dataframe(make_arrow_compatible(df.head(10)), use_container_width=True)
        except Exception as e:
            logger.error(f"Arrow serialization failed after decryption: {e}")
            st.warning(f"⚠️ Failed to display decrypted transactions: {e}. Converting to strings.")
            df = df.apply(lambda x: x.astype(str) if x.dtype == "object" else x)
            st.dataframe(df.head(10), use_container_width=True)
            
        #Identify special cases
        high_value = identify_high_value_transactions(df)
        recurring = recurring_transactions(df, category_rules["Fixed Expenses"] + category_rules["Discretionary Expenses"])
        with st.expander("💰 High Value Transactions"):
            try:
                st.dataframe(make_arrow_compatible(df[high_value][["Date", "Description", "Debit", "Credit"]]))
            except Exception as e:
                logger.error(f"Arrow serialization failed for high value transactions: {e}")
                st.dataframe(df[high_value][["Date", "Description", "Debit", "Credit"]])        
        with st.expander("🔄 Recurring Transactions"):
            try:
                st.dataframe(make_arrow_compatible(df[recurring][["Date", "Description", "Debit", "Credit"]]))
            except Exception as e:
                logger.error(f"Arrow serialization failed for recurring transactions: {e}")
                st.dataframe(df[recurring][["Date", "Description", "Debit", "Credit"]])

        # Categorize transactions
        with st.spinner("🔍 Categorizing transactions using ML & rules..."):
            categorized_file = CATEGORIZED_CSV_DIR / file_name
            # Initialize TransactionCategorizer with vectorizer and classifier
            tc = TransactionCategorizer(vectorizer=vectorizer, classifier=classifier)

            # Categorize transactions using the DataFrame
            try:
                if not isinstance(df, pd.DataFrame):
                    raise ValueError("Input to categorize_transactions must be a pandas DataFrame")
                categorized_df = categorize_transactions(
                    df=df,
                    tc=tc,
                    model_type="hybrid",
                    add_confidence=True
                )
                if not isinstance(categorized_df, pd.DataFrame):
                    logger.error("categorize_transactions returned non-DataFrame")
                    raise ValueError("categorize_transactions must return a pandas DataFrame")
                categorized_df["Description"] = categorized_df["Description"].apply(lambda x: str(x) if pd.notna(x) else 'Unknown')
                categorized_df["Category"] = categorized_df["Category"].apply(lambda x: str(x) if pd.notna(x) else 'Unknown')
                logger.info(f"[DEBUG] After categorize_transactions Description sample: {categorized_df['Description'].head().tolist()}")
                logger.info(f"[DEBUG] After categorize_transactions Category sample: {categorized_df['Category'].head().tolist()}")
                logger.info(f"[DEBUG] After categorize_transactions types: {categorized_df.dtypes}")
            except Exception as e:
                logger.error(f"Categorization failed: {e}. Falling back to rule-based.")
                st.warning(f"⚠️ Categorization failed: {e}. Using rule-based categorization.")
                categorized_df = df.copy()
                categorized_df["Description"] = categorized_df["Description"].apply(lambda x: str(x) if pd.notna(x) else 'Unknown')
                categorized_df["Category"] = categorized_df["Description"].apply(rule_based_categorize)
                categorized_df["Category"] = categorized_df["Category"].apply(lambda x: str(x) if pd.notna(x) else 'Unknown')
                categorized_df["Confidence"] = 1.0
                logger.info(f"[DEBUG] After rule-based categorization Description sample: {categorized_df['Description'].head().tolist()}")
                logger.info(f"[DEBUG] After rule-based categorization Category sample: {categorized_df['Category'].head().tolist()}")
                logger.info(f"[DEBUG] After rule-based categorization types: {categorized_df.dtypes}")


            # Clean numeric columns in categorized_df
            for col in ["Debit", "Credit", "Balance", "Amount", "Confidence"]:
                if col in categorized_df.columns:
                    categorized_df = clean_currency_column(categorized_df, col)
            categorized_df["Description"] = categorized_df["Description"].apply(lambda x: str(x) if pd.notna(x) else 'Unknown')
            categorized_df["Category"] = categorized_df["Category"].apply(lambda x: str(x) if pd.notna(x) else 'Unknown')
            categorized_df = make_arrow_compatible(categorized_df)
            st.write("[DEBUG] categorized_df types:", categorized_df.dtypes)
            st.write("[DEBUG] categorized_df sample:", categorized_df.head())
            categorized_df.to_csv(categorized_file, index=False)
            progress_bar.progress(60)
            
        if categorized_df.empty:
            st.error("❌ Failed to categorize transactions")
        else:
            # Automated Category Review
            st.subheader("🔧 Review Low-Confidence Transaction Categories (Optional)")
            low_confidence_file = OUTPUT_DIR / "low_confidence.csv"
            if os.path.exists(low_confidence_file):
                low_confidence_df = pd.read_csv(low_confidence_file)
                low_confidence_df["Description"] = low_confidence_df["Description"].apply(lambda x: str(x) if pd.notna(x) else 'Unknown')
                low_confidence_df["Category"] = low_confidence_df["Category"].apply(lambda x: str(x) if pd.notna(x) else 'Unknown')
                logger.info(f"[DEBUG] low_confidence_df Description sample: {low_confidence_df['Description'].head().tolist()}")
                logger.info(f"[DEBUG] low_confidence_df Category sample: {low_confidence_df['Category'].head().tolist()}")
                logger.info(f"[DEBUG] low_confidence_df types: {low_confidence_df.dtypes}")
                if not low_confidence_df.empty:
                    st.write(f"Found {len(low_confidence_df)} low-confidence transactions")
                    try:
                        edited_df = st.data_editor(
                            make_arrow_compatible(low_confidence_df),
                            column_config={
                                "Description": st.column_config.TextColumn("Description", disabled=True),
                                "Category": st.column_config.SelectboxColumn(
                                    "Category",
                                    options=list(category_rules.keys()) + ["Uncategorized"],
                                    default="Uncategorized"
                                ),
                                "Confidence": st.column_config.NumberColumn("Confidence", disabled=True)
                            },
                            num_rows="dynamic",
                            key="category_editor"
                        )
                    except Exception as e:
                        logger.error(f"Arrow serialization failed for low_confidence_df: {e}")
                        st.warning(f"⚠️ Failed to display low-confidence transactions: {e}. Converting to strings.")
                        low_confidence_df = low_confidence_df.apply(lambda x: x.astype(str) if x.dtype == "object" else x)
                        edited_df = st.data_editor(
                            low_confidence_df,
                            column_config={
                                "Description": st.column_config.TextColumn("Description", disabled=True),
                                "Category": st.column_config.SelectboxColumn(
                                    "Category",
                                    options=list(category_rules.keys()) + ["Uncategorized"],
                                    default="Uncategorized"
                                ),
                                "Confidence": st.column_config.NumberColumn("Confidence", disabled=True)
                            },
                            num_rows="dynamic",
                            key="category_editor_fallback"
                        )
                    if st.button("Save Category Corrections", key="save_corrections_button"):
                        override_file = OUTPUT_DIR / "overrides.csv"
                        new_overrides = edited_df[["Description", "Category"]][edited_df["Category"] != "Uncategorized"]
                        new_overrides["Description"] = new_overrides["Description"].apply(lambda x: str(x) if pd.notna(x) else 'Unknown')
                        new_overrides["Category"] = new_overrides["Category"].apply(lambda x: str(x) if pd.notna(x) else 'Unknown')
                        if not new_overrides.empty:
                            if os.path.exists(override_file):
                                existing_overrides = pd.read_csv(override_file)
                                existing_overrides["Description"] = existing_overrides["Description"].apply(lambda x: str(x) if pd.notna(x) else 'Unknown')
                                existing_overrides["Category"] = existing_overrides["Category"].apply(lambda x: str(x) if pd.notna(x) else 'Unknown')
                                combined_overrides = pd.concat([existing_overrides, new_overrides]).drop_duplicates(
                                    subset="Description", keep="last"
                                )
                            else:
                                combined_overrides = new_overrides
                            combined_overrides.to_csv(override_file, index=False)
                            st.success(f"✅ Saved {len(new_overrides)} category corrections to {override_file}")
                            for _, row in new_overrides.iterrows():
                                mask = categorized_df["Description"] == row["Description"]
                                categorized_df.loc[mask, "Category"] = row["Category"]
                                categorized_df.loc[mask, "Confidence"] = 1.0
                            categorized_df["Description"] = categorized_df["Description"].apply(lambda x: str(x) if pd.notna(x) else 'Unknown')
                            categorized_df["Category"] = categorized_df["Category"].apply(lambda x: str(x) if pd.notna(x) else 'Unknown')
                            categorized_df.to_csv(categorized_file, index=False)
                            st.write("Updated categorized transactions with corrections")
                else:
                    st.write("No transactions require review")
            else:
                st.write("No transactions require review")

            # Decrypt categorized_df safely
            categorized_df["Description"] = categorized_df["Description"].apply(try_decrypt)
            categorized_df["Description"] = categorized_df["Description"].apply(lambda x: str(x) if pd.notna(x) else 'Unknown')
            categorized_df["Category"] = categorized_df["Category"].apply(lambda x: str(x) if pd.notna(x) else 'Unknown')
            logger.info(f"[DEBUG] After final decryption Description sample: {categorized_df['Description'].head().tolist()}")
            logger.info(f"[DEBUG] After final decryption Category sample: {categorized_df['Category'].head().tolist()}")
            logger.info(f"[DEBUG] After final decryption types: {categorized_df.dtypes}")

            # Transaction Category Pie Chart
            st.subheader("📊 Transaction Category Distribution") 
            category_counts = categorized_df["Category"].value_counts().reset_index() 
            category_counts.columns = ["Category", "Count"] 
            fig = px.pie( category_counts, 
                            values="Count", 
                            names="Category", 
                            title="Transaction Category Distribution", 
                            color_discrete_sequence=["#36A2EB", "#FF6384", "#FFCE56", "#4BC0C0", "#FF5555", "#9966FF"] ) 
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate Metrics using categorized_df
            with st.spinner("📊 Analyzing bank patterns..."):
                metrics_file = OUTPUT_DIR / "metrics" / file_name.replace(".csv", "_metrics.csv")
                metrics_file.parent.mkdir(exist_ok=True)
                metrics_df = calculate_metrics(categorized_df, str(metrics_file))
                metrics_df = enforce_schema(metrics_df)
                metrics_df = make_arrow_compatible(metrics_df) # 🔑 Enforce Arrow-safe schema & numeric conversions
                if metrics_df.empty:
                    st.error("No valid metrics calculated. Please check input data.")
                else:
                    # Ensure numeric columns
                    numeric_cols = [
                        "Average Monthly Income", "Average Monthly Expenses", "Net Surplus",
                        "DTI Ratio", "Discretionary Expenses", "Savings Rate", "Cumulative Savings",
                        "Average Monthly EMI", "Credit Utilization", "Average Monthly Balance",
                        "Cash Withdrawals", "Income Variability Index", "Recent Salary Trend (%)",
                        "Discretionary Spending (%)", "High-Cost EMI Payments", "Credit Card Payments",
                        "Minimum Monthly Balance", "Average Closing Balance", "Overdraft Usage Frequency",
                        "Negative Balance Days", "Sudden High-Value Credits", "Circular Transactions"
                    ]
                    for col in numeric_cols:
                        if col in metrics_df.columns:
                            metrics_df[col] = pd.to_numeric(metrics_df[col], errors="coerce").fillna(0.0).astype(float)

                    # 🔑 Enforce DTI Ratio if missing or corrupted
                    if "DTI Ratio" not in metrics_df.columns and {"Average Monthly EMI", "Average Monthly Income"}.issubset(metrics_df.columns):
                        emi = pd.to_numeric(metrics_df["Average Monthly EMI"], errors="coerce").fillna(0.0)
                        inc = pd.to_numeric(metrics_df["Average Monthly Income"], errors="coerce").fillna(0.0)
                        metrics_df["DTI Ratio"] = np.where(inc > 0, (emi / inc) * 100.0, 0.0).astype(float)

                    logger.info(f"metrics_df types after schema enforcement: {metrics_df.dtypes}")
                    st.write("[DEBUG] metrics_df after enforcement:", metrics_df.head())
                    st.success("✅ Metrics calculated successfully")
                    progress_bar.progress(70)

            # Step 5: Run visualizer.py pipeline
            with st.spinner("Generating full analysis and PDF report..."):
                temp_path = TEMP_DIR / file_name
                temp_path.parent.mkdir(exist_ok=True)
                categorized_df.to_csv(temp_path, index=False)
                try:
                    logger.info(f"Calling analyze_file with applicant_name: {applicant_name}, account_number: {account_number}")
                    applicant_data = {
                        "name": applicant_name if applicant_name else "Unknown",
                        "account_number": account_number if account_number else "Unknown"}
                    results = analyze_file(
                        input_path=str(temp_path),
                        cibil_score=int(cibil_score),
                        fill_method="interpolate",
                        out_dir=str(OUTPUT_DIR),
                        applicant_data=applicant_data
                    )
                    progress_bar.progress(90)
                except Exception as e:
                    st.error(f"❌ Analysis failed: {e}")
                    logger.error(f"Analysis failed: {e}")
                    results = None

            # Step 6: Display results
            if results:
                st.header("Loan Analysis Results")
                metrics_df = results.get("metrics_df", pd.DataFrame())
                monthly_income = float(metrics_df["Average Monthly Income"].iloc[0]) if not metrics_df.empty else 0.0
                monthly_expenses = float(metrics_df["Average Monthly Expenses"].iloc[0]) if not metrics_df.empty and "Average Monthly Expenses" in metrics_df.columns else 0.0
                net_surplus = float(metrics_df["Net Surplus"].iloc[0]) if not metrics_df.empty and "Net Surplus" in metrics_df.columns else monthly_income - monthly_expenses
                dti_ratio = float(metrics_df["DTI Ratio"].iloc[0]) if not metrics_df.empty and "DTI Ratio" in metrics_df.columns else 0.0
                red_flag_count = int(metrics_df["Red Flag Count"].iloc[0]) if not metrics_df.empty and "Red Flag Count" in metrics_df.columns else 0
                discretionary_spending = float(metrics_df["Discretionary Spending (%)"].iloc[0]) if "Discretionary Spending (%)" in metrics_df.columns else 0.0
                avg_monthly_balance = float(metrics_df["Average Closing Balance"].iloc[0]) if "Average Closing Balance" in metrics_df.columns else 0.0
                cash_withdrawals = float(metrics_df["Cash Withdrawals"].iloc[0]) if "Cash Withdrawals" in metrics_df.columns else 0.0
                existing_loans = int(metrics_df["Existing Loan Count"].iloc[0]) if "Existing Loan Count" in metrics_df.columns else 0
                bounced_cheques = int(metrics_df["Bounced Cheques Count"].iloc[0]) if "Bounced Cheques Count" in metrics_df.columns else 0
                overdraft_frequency = int(metrics_df["Overdraft Usage Frequency"].iloc[0]) if "Overdraft Usage Frequency" in metrics_df.columns else 0
                negative_balance_days = int(metrics_df["Negative Balance Days"].iloc[0]) if "Negative Balance Days" in metrics_df.columns else 0
                high_value_credits = int(metrics_df["Sudden High-Value Credits"].iloc[0]) if "Sudden High-Value Credits" in metrics_df.columns else 0
                circular_transactions = int(metrics_df["Circular Transactions"].iloc[0]) if "Circular Transactions" in metrics_df.columns else 0
                income_variability = float(metrics_df["Income Variability Index"].iloc[0]) if "Income Variability Index" in metrics_df.columns else 0.0
                salary_trend = float(metrics_df["Recent Salary Trend (%)"].iloc[0]) if "Recent Salary Trend (%)" in metrics_df.columns else 0.0
                savings_ratio = float(metrics_df["Savings Ratio (%)"].iloc[0]) if "Savings Ratio (%)" in metrics_df.columns else 0.0
                emi_payments = float(metrics_df["High-Cost EMI Payments"].iloc[0]) if "High-Cost EMI Payments" in metrics_df.columns else 0.0

                # Decision
                try:
                    decision = score_and_decide(
                        metrics_df=metrics_df,
                        cibil_score=cibil_score,
                        categorized_file=str(categorized_file)
                    )
                except Exception as e:
                    st.error(f"Error in decision logic: {e}")
                    decision = {"Total Score": 0, "Risk Level": "High", "Action": "Reject", "Reason": f"Error: {e}"}
                bank_score = float(decision["Total Score"])

                st.subheader("Analysis")
                st.write(f"- **Net Surplus**: ₹{monthly_income:,.2f} - ₹{monthly_expenses:,.2f} = ₹{net_surplus:,.2f}")
                st.write(f"- **Red Flags**: {red_flag_count} (including {bounced_cheques} bounced cheques)")
                st.write(f"- **Bank Statement Score**: {bank_score:.1f}/100")

                st.subheader("Decision")
                reason = decision.get("Reason", "Based on DTI, CIBIL, and transaction history")
                st.write(f"**{decision['Action']}**: {reason}")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Bank Score", f"{bank_score}/100")
                with col2:
                    st.metric("Risk Level", decision['Risk Level'])
                with col3:
                    st.metric("Action", decision['Action'])

                # ML Prediction
                st.subheader("ML Model Decision")
                prediction = None
                prob = [0, 0]
                try:
                    with open(MODEL_DIR / "loan_approval_model.pkl", "rb") as f:
                        ml_model = pickle.load(f)
                    input_features = [
                        float(monthly_income),
                        float(net_surplus),
                        float(dti_ratio),
                        float(savings_ratio),
                        float(red_flag_count),
                        float(cibil_score),
                        float(salary_trend),
                        float(bounced_cheques),
                        float(discretionary_spending),
                        float(metrics_df.get("Cumulative Savings", pd.Series([0])).iloc[0])
                    ]
                    expected_features = ml_model.n_features_in_
                    if len(input_features) > expected_features:
                        input_features = input_features[:expected_features]
                    elif len(input_features) < expected_features:
                        input_features += [0] * (expected_features - len(input_features))
                    prediction = ml_model.predict([input_features])[0]
                    prob = ml_model.predict_proba([input_features])[0]
                    if prediction == 1:
                        st.success(f"✅ APPROVED ({round(prob[1] * 100, 2)}% confidence)")
                    else:
                        st.error(f"❌ REJECTED ({round(prob[0] * 100, 2)}% confidence)")
                except Exception as e:
                    st.warning(f"⚠️ ML Prediction Failed: {e}")

                # Visual Insights
                st.subheader("📈 Visual Insights")
                if metrics_df is not None and not metrics_df.empty:
                    # Ensure column types before visualizations
                    categorized_df["Description"] = categorized_df["Description"].apply(lambda x: str(x) if pd.notna(x) else 'Unknown')
                    categorized_df["Category"] = categorized_df["Category"].apply(lambda x: str(x) if pd.notna(x) else 'Unknown')
                    categorized_df["Date"] = pd.to_datetime(categorized_df["Date"], errors='coerce').dt.strftime('%Y-%m-%d')
                    categorized_df["Date"] = categorized_df["Date"].apply(lambda x: str(x) if pd.notna(x) else 'Unknown')
                    logger.info(f"[DEBUG] Before visualizations types: {categorized_df.dtypes}")
                    logger.info(f"[DEBUG] Before visualizations Description sample: {categorized_df['Description'].head().tolist()}")
                    logger.info(f"[DEBUG] Before visualizations Category sample: {categorized_df['Category'].head().tolist()}")
                    logger.info(f"[DEBUG] Before visualizations Date sample: {categorized_df['Date'].head().tolist()}")
                    try:
                        # Row 1: Income trend + Surplus trend
                        col1, col2 = st.columns(2)
                        with col1:
                            fig_income = plot_income_trend_plotly(categorized_df)
                            if fig_income:
                                st.plotly_chart(fig_income, use_container_width=True)
                            else:
                                st.info("No income data available.")
                        with col2:
                            fig_surplus = plot_surplus_trend_plotly(categorized_df)
                            if fig_surplus:
                                st.plotly_chart(fig_surplus, use_container_width=True)
                            else:
                                st.info("No surplus data available.")

                        # Row 2: Income vs Expenses
                        st.markdown("### 💰 Income vs Expenses")
                        fig_expenses = plot_income_vs_expenses_plotly(categorized_df)
                        if fig_expenses:
                            st.plotly_chart(fig_expenses, use_container_width=True)

                        # Row 3: Cumulative savings
                        st.markdown("### 📊 Cumulative Savings Over Time")
                        fig_savings = plot_cumulative_savings_plotly(categorized_df)
                        if fig_savings:
                            st.plotly_chart(fig_savings, use_container_width=True)

                        # Row 4: Category breakdown
                        st.markdown("### 📂 Expense Category Breakdown")
                        fig_category = plot_category_breakdown_plotly(categorized_df)
                        if fig_category:
                            st.plotly_chart(fig_category, use_container_width=True)
                        else:
                            st.warning("⚠️ No category data available for visualization.")

                        # Row 5: DTI Ratio
                        st.markdown("### 📊 Debt-to-Income Ratio")
                        if "DTI Ratio" in metrics_df.columns:
                            dti_value = float(metrics_df["DTI Ratio"].iloc[0]) if not metrics_df["DTI Ratio"].empty else 0.0
                            if dti_value >= 0:
                                fig_dti = px.bar(
                                    x=["DTI Ratio"],
                                    y=[dti_value],
                                    title="Debt-to-Income Ratio (Acceptable ≤40%)",
                                    labels={"x": "Metric", "y": "Percentage (%)"},
                                    color_discrete_sequence=["#FF6384"]
                                )
                                st.plotly_chart(fig_dti, use_container_width=True)
                            else:
                                st.warning(f"⚠️ Invalid DTI Ratio value: {dti_value}")
                    except Exception as e:
                        st.warning(f"Could not generate visualizations: {e}")

                    # Download PDF Report
                    pdf_path = results.get("pdf_report")
                    if pdf_path and Path(pdf_path).exists():
                        with open(pdf_path, "rb") as f:
                            st.download_button(
                                label="📥 Download PDF Report",
                                data=f,
                                file_name=Path(pdf_path).name,
                                mime="application/pdf"
                            )
                    else:
                        st.warning("⚠️ PDF report not found.")

                else:
                    st.warning("⚠️ No metrics data available for visualization.")

                # Save to Log
                output_file = OUTPUT_DIR / "loan_scores_log.csv"
                summary_data = {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "File": file_name.replace(".csv", "_metrics.csv"),
                    "CIBIL Score": cibil_score,
                    "Bank Score": bank_score,
                    "Risk Level": decision["Risk Level"],
                    "Action": decision["Action"],
                    "Average Monthly Income": monthly_income,
                    "Average Monthly Expenses": monthly_expenses,
                    "Net Surplus": net_surplus,
                    "DTI Ratio": dti_ratio,
                    "Red Flag Count": red_flag_count,
                    "Average Closing Balance": avg_monthly_balance,
                    "Cash Withdrawals": cash_withdrawals,
                    "Existing Loan Count": existing_loans,
                    "Bounced Cheques Count": bounced_cheques,
                    "Overdraft Usage Frequency": overdraft_frequency,
                    "Negative Balance Days": negative_balance_days,
                    "Sudden High-Value Credits": high_value_credits,
                    "Circular Transactions": circular_transactions,
                    "Income Variability Index": income_variability,
                    "Recent Salary Trend (%)": salary_trend,
                    "Savings Ratio (%)": savings_ratio,
                    "High-Cost EMI Payments": emi_payments,
                    "ML Decision": "APPROVED" if prediction == 1 else "REJECTED",
                    "Confidence": round(max(prob) * 100, 2)
                }
                df_output = pd.DataFrame([summary_data])
                if output_file.exists():
                    df_output.to_csv(output_file, mode='a', header=False, index=False)
                else:
                    df_output.to_csv(output_file, index=False)

                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
    else:
        st.error("❌ No valid transactions extracted")
else:
    st.error("❌ No file uploaded")