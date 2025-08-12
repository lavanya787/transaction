import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import plotly.express as px
import chardet
import fitz  # PyMuPDF
import docx2txt
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import spacy
import re
from cryptography.fernet import Fernet  # For encryption
from utils.calculate_metrics import calculate_metrics
from utils.score_bank_statements import score_and_decide
from utils.check_cibil import check_cibil_risk
from utils.visualizer import ( 
    analyze_file,
    plot_income_trend_plotly,
    plot_surplus_trend_plotly,
    plot_income_vs_expenses_plotly,
    plot_cumulative_savings_plotly,
    plot_category_breakdown_plotly
)
from pathlib import Path
import base64
from utils.text_parser import parse_bank_statement_text
from utils.ocr_extract import extract_df_from_scanned_pdf
from utils.categorize_transactions import categorize_transactions

# Currency cleaning helper ---
def clean_currency_column(df, col):
    if col in df.columns:
        df[col] = (
            df[col].replace('[₹,]', '', regex=True)
                   .replace('', '0')
                   .astype(float)
        )
    return df

# --- Load spaCy for Rule-Based Categorization ---
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    st.warning(f"⚠️ spaCy model not available: {e}. Using basic rule-based categorization.")
    nlp = None

# --- Category Rules with Indian Banking Context ---
category_rules = {
    "Income": ["salary", "bonus", "freelance", "investment", "neft.*cr", "imps.*cr", "upi.*cr", r"\[Salary\]", r"\[Bonus\]", r"\[Freelance\]"],
    "Fixed Expenses": ["rent", "emi", "insurance", "electricity", "phone", "bill", "loan", "paytm.*bill", r"\[Rent\]", r"\[Electricity\]"],
    "Discretionary Expenses": ["shopping", "dining", "entertainment", "travel", "amazon", "zomato", "swiggy", "flipkart", r"\[Shopping\]", r"\[Dining\]"],
    "Savings": ["fd", "rd", "mutual fund", "deposit", "sip", "ppf", "nps", r"\[FD\]", r"\[RD\]"],
    "Red Flags": ["overdraft", "bounce", "insufficient funds", r"cash.*\d{4,}", r"upi.*\d{5,}.*cash"]
}

# --- Rule-Based Categorization ---
def rule_based_categorize(description):
    if nlp:
        doc = nlp(description.lower())
        for category, patterns in category_rules.items():
            for pattern in patterns:
                if re.search(pattern, description.lower()):
                    return category
    return "Other"
# --- Standardize Column Names ---
def standardize_columns(df):
    column_mapping = {
        'debit': 'Debit', 'DEBIT': 'Debit', 'withdrawal': 'Debit',
        'credit': 'Credit', 'CREDIT': 'Credit', 'deposit': 'Credit',
        'balance': 'Balance', 'BALANCE': 'Balance',
        'date': 'Date', 'DATE': 'Date', 'transaction date': 'Date',
        'description': 'Description', 'DESCRIPTION': 'Description', 'particulars': 'Description', 'narration': 'Description',
        'category': 'Category', 'CATEGORY': 'Category'
    }
    df.columns = [column_mapping.get(col.lower(), col) for col in df.columns]
    return df

# --- Train Categorization Model ---
def train_categorization_model(uploaded_file=None):
    if uploaded_file:
        try:
            raw = uploaded_file.read()
            encoding = chardet.detect(raw)["encoding"] or "latin-1"  # Fallback to latin-1
            data = pd.read_csv(io.StringIO(raw.decode(encoding, errors="replace")))
            if not {"Description", "Category"}.issubset(data.columns):
                st.error("Training file must have 'Description' and 'Category' columns")
                return None, None
        except Exception as e:
            st.error(f"Error loading training file: {e}")
            return None, None
    else:
        data = pd.DataFrame({
            "Description": [
                "Salary Credit NEFT", "Rent Payment UPI", "EMI HDFC", "Shopping Amazon",
                "Utility Bill Paytm", "Loan Repayment SBI", "Bonus IMPS", "FD ICICI",
                "Overdraft Fee", "Zomato Order"
            ],
            "Category": [
                "Income", "Fixed Expenses", "Fixed Expenses", "Discretionary Expenses",
                "Fixed Expenses", "Fixed Expenses", "Income", "Savings",
                "Red Flags", "Discretionary Expenses"
            ]
        })

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data["Description"])
    model = RandomForestClassifier(random_state=42)
    model.fit(X, data["Category"])

    os.makedirs(r"C:\Users\lavan\OneDrive\Desktop\bank_transaction_tool\models", exist_ok=True)
    model_path = r"C:\Users\lavan\OneDrive\Desktop\bank_transaction_tool\models\category_classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "vectorizer": vectorizer,
            "classifier": model,
            "label_encoder": None
        }, f, protocol=4)
    st.success(f"✅ Categorization model trained and saved to {model_path}")
    return vectorizer, model

# --- Page Setup ---
st.set_page_config(page_title="📊 Loan Scoring Dashboard", layout="wide")
st.title("🏦 AI-Powered Loan Eligibility Analyzer")
st.markdown("Boost lending decisions with smarter transaction scoring and real-time risk assessment per RBI guidelines.")

# --- Sidebar Inputs ---
st.sidebar.header("🔍 Input Parameters")
uploaded_file = st.sidebar.file_uploader("📤 Upload Bank Statement", type=["csv", "pdf", "txt", "doc", "docx"])
cibil_score = st.sidebar.slider("🧾 CIBIL Score", min_value=300, max_value=900, step=10, value=720)
proposed_emi = st.sidebar.number_input("💰 Proposed EMI (₹)", min_value=0, value=8000)
loan_purpose = st.sidebar.selectbox("🏦 Loan Purpose", ["Education", "Business", "Home Improvement", "Luxury", "Vacation", "Neutral"])
co_applicant = st.checkbox("Co-applicant Available")
file_name = uploaded_file.name if uploaded_file else None

st.sidebar.markdown("---")
st.sidebar.caption("📁 Supports CSV, PDF, TXT, DOC formats for statements; CSV for training (Description, Category)")

# --- Encryption Setup ---
key = Fernet.generate_key()
cipher_suite = Fernet(key)

# --- Retrain Model ---
if uploaded_file and st.sidebar.button("Retrain Categorization Model"):
    vectorizer, classifier = train_categorization_model(uploaded_file)
else:
    try:
        with open(r"C:\Users\lavan\OneDrive\Desktop\bank_transaction_tool\models\category_classifier.pkl", "rb") as f:
            model_data = pickle.load(f)
        vectorizer = model_data["vectorizer"]
        classifier = model_data["classifier"]
        st.sidebar.success("✅ Loaded existing categorization model")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Failed to load model: {e}. Using rule-based categorization.")
        vectorizer, classifier = None, None

# --- File Parsing with Data Quality Validation ---
df = None
text = ""
metrics_df = None

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1].lower()
    file_name = uploaded_file.name.replace(f".{file_type}", ".csv")

    # Validate standardized formats and completeness
    if file_type not in ["csv", "pdf", "txt", "doc", "docx"]:
        st.error("❌ Unsupported file format. Use CSV, PDF, TXT, DOC, or DOCX.")
    else:
        try:
            if file_type == "csv":
                raw = uploaded_file.read()
                encoding = chardet.detect(raw)["encoding"] or "latin-1"
                df = pd.read_csv(io.StringIO(raw.decode(encoding, errors= "replace")))
                df = standardize_columns(df)
                required_cols = {"Date", "Description", "Debit", "Credit", "Balance"}
                if not required_cols.issubset(df.columns):
                    st.error(f"❌ CSV missing required columns: {required_cols - set(df.columns)}")
                    with open(r"C:\Users\lavan\OneDrive\Desktop\bank_transaction_tool\outputs\extraction_errors.log", "a") as f:
                        f.write(f"{datetime.now()}: CSV missing columns in {file_name}: {required_cols - set(df.columns)}\n")
                    df = None
                elif df.empty or df.isnull().all().all():
                    st.error("❌ CSV is empty or contains only null values")
                    with open(r"C:\Users\lavan\OneDrive\Desktop\bank_transaction_tool\outputs\extraction_errors.log", "a") as f:
                        f.write(f"{datetime.now()}: Empty or invalid CSV {file_name}\n")
                    df = None
                else:
                    # FIX: Clean currency fields here
                    for col in ["Debit", "Credit", "Balance", "Value"]:
                        df = clean_currency_column(df, col)
                    st.success("✅ CSV Loaded Successfully")
            elif file_type == "pdf":
                uploaded_data = uploaded_file.read()
                if not uploaded_data:
                    st.error("❌ Uploaded PDF is empty")
                    with open(r"C:\Users\lavan\OneDrive\Desktop\bank_transaction_tool\outputs\extraction_errors.log", "a") as f:
                        f.write(f"{datetime.now()}: Empty PDF uploaded: {file_name}\n")
                    df = None

                else:
                    try:
                        with fitz.open(stream=uploaded_data, filetype="pdf") as doc:
                            if doc.page_count == 0:
                                st.error("❌ PDF has no pages")
                                with open(r"C:\Users\lavan\OneDrive\Desktop\bank_transaction_tool\outputs\extraction_errors.log", "a") as f:
                                    f.write(f"{datetime.now()}: PDF has no pages: {file_name}\n")
                                df = None
                            else:
                                text = "\n".join([page.get_text() for page in doc])
                                if text.strip():
                                    st.info("📄 Extracted Text Preview")
                                    st.text_area("PDF Extract", text[:1000])
                                    df = parse_bank_statement_text(text)
                                    df = standardize_columns(df)
                                else:
                                    st.warning("📷 OCR Mode Activated for Scanned PDF")
                                    df = extract_df_from_scanned_pdf(uploaded_data)
                                df = standardize_columns(df)
                                # FIX: Clean currency columns
                                for col in ["Debit", "Credit", "Balance", "Value"]:
                                    df = clean_currency_column(df, col)
                                    
                    except Exception as e:
                        st.error(f"❌ Error processing PDF: {e}")
                        with open(r"C:\Users\lavan\OneDrive\Desktop\bank_transaction_tool\outputs\extraction_errors.log", "a") as f:
                            f.write(f"{datetime.now()}: Error processing PDF {file_name}: {e}\n")
                        df = None
            elif file_type == "txt":
                raw = uploaded_file.read()
                encoding = chardet.detect(raw)["encoding"]
                text = raw.decode(encoding)
                st.text_area("TXT Extract", text[:1000])
                df = parse_bank_statement_text(text)
                df = standardize_columns(df)
            elif file_type in ["doc", "docx"]:
                with open(r"C:\Users\lavan\OneDrive\Desktop\bank_transaction_tool\temp.docx", "wb") as f:
                    f.write(uploaded_file.read())
                text = docx2txt.process(r"C:\Users\lavan\OneDrive\Desktop\bank_transaction_tool\temp.docx")
                st.text_area("DOC Extract", text[:1000])
                df = parse_bank_statement_text(text)
                df = standardize_columns(df)

            if df is not None and not df.empty:
                required_cols = {"Date", "Description", "Debit", "Credit", "Balance"}
                if not required_cols.issubset(df.columns):
                    st.error(f"❌ Parsed DataFrame missing required columns: {required_cols - set(df.columns)}")
                    with open(r"C:\Users\lavan\OneDrive\Desktop\bank_transaction_tool\outputs\extraction_errors.log", "a") as f:
                        f.write(f"{datetime.now()}: Missing columns in {file_name}: {required_cols - set(df.columns)}\n")
                    df = None
                else:
                    os.makedirs(r"C:\Users\lavan\OneDrive\Desktop\bank_transaction_tool\cleaned_csvs", exist_ok=True)
                    df["Description"] = df["Description"].apply(lambda x: cipher_suite.encrypt(str(x).encode()).decode() if pd.notna(x) else "")
                    df.to_csv(f"C:/Users/lavan/OneDrive/Desktop/bank_transaction_tool/cleaned_csvs/{file_name}", index=False)
                    st.success("✅ Transactions Parsed and Encrypted")
                    print(f"[DEBUG] Parsed DataFrame columns: {df.columns.tolist()}")
            else:
                st.error("❌ No valid transactions extracted")
                with open(r"C:\Users\lavan\OneDrive\Desktop\bank_transaction_tool\outputs\extraction_errors.log", "a") as f:
                    f.write(f"{datetime.now()}: Failed to extract transactions from {file_name}\n")

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
            with open(r"C:\Users\lavan\OneDrive\Desktop\bank_transaction_tool\outputs\extraction_errors.log", "a") as f:
                f.write(f"{datetime.now()}: Error processing {file_name}: {e}\n")

# --- Main Processing ---
if df is not None and not df.empty:
    st.subheader("📄 Parsed Transactions")

    # Decrypt only if actually encrypted
    def try_decrypt(val):
        if pd.isna(val) or not isinstance(val, str):
            return val
        try:
            return cipher_suite.decrypt(val.encode()).decode()
        except Exception:
            return val  # already plain text

    df["Description"] = df["Description"].apply(try_decrypt)
    st.dataframe(df.head(10), use_container_width=True)

    # Categorize transactions
    with st.spinner("🔍 Categorizing transactions using ML & rules..."):
        os.makedirs(r"C:\Users\lavan\OneDrive\Desktop\bank_transaction_tool\categorized_csvs", exist_ok=True)
        categorized_file = f"C:/Users/lavan/OneDrive/Desktop/bank_transaction_tool/categorized_csvs/{file_name}"
        categorized_df = categorize_transactions(
            f"C:/Users/lavan/OneDrive/Desktop/bank_transaction_tool/cleaned_csvs/{file_name}",
            categorized_file,
            vectorizer=vectorizer,
            classifier=classifier
        )

    if categorized_df.empty:
        st.error("❌ Failed to categorize transactions")
    else:
        # Automated Category Review
        st.subheader("🔧 Review Low-Confidence Transaction Categories (Optional)")
        low_confidence_file = "outputs/low_confidence.csv"
        if os.path.exists(low_confidence_file):
            low_confidence_df = pd.read_csv(low_confidence_file)
            if not low_confidence_df.empty:
                st.write(f"Found {len(low_confidence_df)} low-confidence transactions")
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
                    key="category_editor"
                )
                if st.button("Save Category Corrections"):
                    override_file = "overrides.csv"
                    new_overrides = edited_df[["Description", "Category"]][edited_df["Category"] != "Uncategorized"]
                    if not new_overrides.empty:
                        if os.path.exists(override_file):
                            existing_overrides = pd.read_csv(override_file)
                            combined_overrides = pd.concat([existing_overrides, new_overrides]).drop_duplicates(
                                subset="Description", keep="last"
                            )
                        else:
                            combined_overrides = new_overrides
                        combined_overrides.to_csv(override_file, index=False)
                        st.success(f"✅ Saved {len(new_overrides)} category corrections to {override_file}")

                        # Apply corrections in categorized_df
                        for _, row in new_overrides.iterrows():
                            mask = categorized_df["Description"].str.lower() == row["Description"].lower()
                            categorized_df.loc[mask, "Category"] = row["Category"]
                            categorized_df.loc[mask, "Confidence"] = 1.0
                        categorized_df.to_csv(categorized_file, index=False)
                        st.write("Updated categorized transactions with corrections")
            else:
                st.write("No transactions require review")
        else:
            st.write("No transactions require review")

        # Decrypt categorized_df safely
        categorized_df["Description"] = categorized_df["Description"].apply(try_decrypt)

        # Transaction Category Pie Chart
        st.subheader("📊 Transaction Category Distribution")
        category_counts = categorized_df["Category"].value_counts().reset_index()
        category_counts.columns = ["Category", "Count"]
        fig = px.pie(
            category_counts,
            values="Count",
            names="Category",
            title="Transaction Category Distribution",
            color_discrete_sequence=["#36A2EB", "#FF6384", "#FFCE56", "#4BC0C0", "#FF5555", "#9966FF"]
        )
        st.plotly_chart(fig, use_container_width=True)

        # Load loan applications
        loan_file = "loan_applications.csv"
        try:
            loan_purposes = pd.read_csv(loan_file)
            loan_purposes = dict(zip(
                loan_purposes["File"],
                zip(loan_purposes["Loan Purpose"], pd.to_numeric(loan_purposes["Proposed EMI"], errors='coerce').fillna(0.0))
            ))
        except Exception as e:
            st.warning(f"[WARNING] Failed to load loan applications: {e}, using provided loan purpose and EMI")
            loan_purposes = {f"{file_name.replace('.csv', '_metrics.csv')}": (loan_purpose, proposed_emi)}

        # Calculate Metrics using categorized_df
        with st.spinner("📊 Analyzing bank patterns..."):
            metrics_file = f"C:/Users/lavan/OneDrive/Desktop/bank_transaction_tool/outputs/metrics/{file_name.replace('.csv', '_metrics.csv')}"
            os.makedirs(r"C:\Users\lavan\OneDrive\Desktop\bank_transaction_tool\outputs\metrics", exist_ok=True)
            metrics_df = calculate_metrics(categorized_df, metrics_file)
            if metrics_df.empty:
                st.error("No valid metrics calculated. Please check input data.")
            else:
                st.success("✅ Metrics calculated successfully")

        # Decision Logic
        try:
            decision = score_and_decide(
                metrics_df=metrics_df,
                cibil_score=cibil_score,
                loan_purpose=loan_purpose,
                proposed_emi=proposed_emi,
                categorized_file=categorized_file,
                loan_purposes=loan_purposes
            )
        except Exception as e:
            st.error(f"Error in decision logic: {e}")
            decision = {"Total Score": 0, "Risk Level": "High", "Action": "Reject", "Reason": f"Error: {e}"}

        # Special handling for low CIBIL score
        if cibil_score < 600 and decision["Action"] == "Reject":
            adjusted_emi = proposed_emi * 0.5
            avg_income = metrics_df["Average Monthly Income"].iloc[0] if not metrics_df.empty else 0
            avg_emi = metrics_df["Average Monthly EMI"].iloc[0] if not metrics_df.empty else 0
            new_dti = ((avg_emi + adjusted_emi) / avg_income * 100) if avg_income > 0 else 100
            if co_applicant and new_dti <= 40:
                decision["Action"] = "Approve with higher interest and co-applicant"
                decision["Reason"] = f"Low CIBIL ({cibil_score}) mitigated by co-applicant. Adjusted EMI: ₹{adjusted_emi:,.2f}, DTI: {new_dti:.1f}%"
                decision["Total Score"] = min(75, decision["Total Score"] + 10)
            elif new_dti <= 40:
                decision["Action"] = "Approve with higher interest (small loan)"
                decision["Reason"] = f"Low CIBIL ({cibil_score}) with adjusted EMI: ₹{adjusted_emi:,.2f}, DTI: {new_dti:.1f}%"
                decision["Total Score"] = min(70, decision["Total Score"] + 5)


    # --- Formatted Output ---
    st.header("Loan Analysis Results")
    monthly_income = metrics_df["Average Monthly Income"].iloc[0]
    monthly_expenses = metrics_df["Average Monthly Expenses"].iloc[0]
    existing_emis = metrics_df["Average Monthly EMI"].iloc[0] if "Average Monthly EMI" in metrics_df else monthly_expenses * 0.3
    net_surplus = metrics_df["Net Surplus"].iloc[0]
    dti_ratio = metrics_df["DTI Ratio"].iloc[0]
    red_flag_count = metrics_df["Red Flag Count"].iloc[0]
    discretionary_expenses = metrics_df["Discretionary Expenses"].iloc[0] if "Discretionary Expenses" in metrics_df else 0
    bank_score = decision["Total Score"]

    st.subheader("Customer Profile")
    st.table(pd.DataFrame({
        "Metric": ["CIBIL Score", "Monthly Income", "Monthly Expenses", "Existing EMIs", "Proposed EMI", "Discretionary Expenses"],
        "Value": [cibil_score, f"₹{monthly_income:,.2f}", f"₹{monthly_expenses:,.2f}", f"₹{existing_emis:,.2f}", f"₹{proposed_emi:,.2f}", f"₹{discretionary_expenses:,.2f}"]
    }))

    st.subheader("Analysis")
    st.write(f"- **Net Surplus**: ₹{monthly_income:,.2f} - ₹{monthly_expenses:,.2f} = ₹{net_surplus:,.2f}")
    st.write(f"- **DTI Ratio**: (₹{existing_emis:,.2f} + ₹{proposed_emi:,.2f}) / ₹{monthly_income:,.2f} = {dti_ratio:.1f}% (acceptable if <40%)")
    st.write(f"- **Red Flags**: {red_flag_count} (no overdrafts, bounces, or high cash withdrawals)")
    st.write(f"- **Bank Statement Score**: {bank_score:.1f}/100 (stable income, sufficient surplus, moderate DTI)")

    st.subheader("Decision")
    reason = decision.get("Reason", "Stable income, sufficient surplus, moderate DTI, no red flags" if decision["Action"] == "Approve with standard terms" else "High DTI, low surplus, or excessive red flags")
    st.write(f"**{decision['Action']}**: {reason}")

    # --- Metrics Display ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("💯 Bank Score", f"{bank_score}/100")
    with col2:
        st.metric("⚠️ Risk Level", decision['Risk Level'])
    with col3:
        st.metric("📌 Action", decision['Action'])

    # --- ML Prediction ---
    st.subheader("🤖 ML Model Decision")

    # Defaults to prevent NameError in case of failure
    prediction = None
    prob = [0, 0]  # [reject_prob, approve_prob]

    try:
        with open(r"C:\Users\lavan\OneDrive\Desktop\bank_transaction_tool\models\loan_approval_model.pkl", "rb") as f:
            ml_model = pickle.load(f)

        input_features = [
            monthly_income,
            net_surplus,
            dti_ratio,
            metrics_df["Savings Rate"].iloc[0],
            red_flag_count,
            cibil_score,
            metrics_df.get("Income Trend", pd.Series([0])).iloc[0],
            metrics_df.get("EMI Bounce Count", pd.Series([0])).iloc[0],
            metrics_df.get("Discretionary Expenses", pd.Series([0])).iloc[0],
            metrics_df.get("Cumulative Savings", pd.Series([0])).iloc[0]
        ]
        
        
        # FIX: Match feature count to model expectation
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


    # --- Visual Insights ---
    st.subheader("📈 Visual Insights")
    
    if metrics_df is not None and not metrics_df.empty:
        # Convert PeriodIndex to DatetimeIndex if needed
        if isinstance(metrics_df.index, pd.PeriodIndex):
            metrics_df.index = metrics_df.index.to_timestamp()
        elif not isinstance(metrics_df.index, pd.DatetimeIndex):
            metrics_df.index = pd.to_datetime(metrics_df.index, errors="coerce")
        

    # Save categorized file temporarily for analyzer
    temp_path = Path("temp") / f"{file_name}"
    temp_path.parent.mkdir(exist_ok=True)
    categorized_df.to_csv(temp_path, index=False)

    # Run upgraded analyzer from visualizer.py
    try:
        results = analyze_file(
            input_path=str(temp_path),
            cibil_score=cibil_score,
            proposed_emi=proposed_emi,
            fill_method="zero",  # could also be "interpolate"
            out_dir="outputs"
        )

        if metrics_df is not None and not metrics_df.empty:
            # Row 1: Income trend + Surplus trend
            col1, col2 = st.columns(2)

            with col1:
                fig_income = plot_income_trend_plotly(metrics_df)
                if fig_income:
                    st.plotly_chart(fig_income, use_container_width=True)
                else:
                    st.info("No income data available.")

            with col2:
                fig_surplus = plot_surplus_trend_plotly(metrics_df)
                if fig_surplus:
                    st.plotly_chart(fig_surplus, use_container_width=True)
                else:
                    st.info("No surplus data available.")

            # Row 2: Income vs Expenses
            st.markdown("### 💰 Income vs Expenses")
            fig_expenses = plot_income_vs_expenses_plotly(metrics_df)
            if fig_expenses:
                st.plotly_chart(fig_expenses, use_container_width=True)

            # Row 3: Cumulative savings
            st.markdown("### 📊 Cumulative Savings Over Time")
            fig_savings = plot_cumulative_savings_plotly(metrics_df)
            if fig_savings:
                st.plotly_chart(fig_savings, use_container_width=True)

            # Row 4: Category breakdown
            st.markdown("### 📂 Expense Category Breakdown")
            fig_category = plot_category_breakdown_plotly(metrics_df)
            if fig_category:
                st.plotly_chart(fig_category, use_container_width=True)
        else:
            st.warning("⚠️ No metrics data available for visualization.")


        # Allow HTML report download
        if "html_report" in results and Path(results["html_report"]).exists():
            with open(results["html_report"], "rb") as f:
                st.download_button(
                    label="📥 Download Full HTML Report",
                    data=f,
                    file_name=Path(results["html_report"]).name,
                    mime="text/html"
                )

    except Exception as e:
        st.error(f"❌ Error generating visual insights: {e}")


    # --- Save to Log ---
    output_dir = r"C:\Users\lavan\OneDrive\Desktop\bank_transaction_tool\outputs"
    os.makedirs(output_dir, exist_ok=True)
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
        "ML Decision": "APPROVED" if prediction == 1 else "REJECTED",
        "Confidence": round(max(prob) * 100, 2)
    }
    output_file = os.path.join(output_dir, "loan_scores_log.csv")
    df_output = pd.DataFrame([summary_data])
    if os.path.exists(output_file):
        df_output.to_csv(output_file, mode='a', header=False, index=False)
    else:
        df_output.to_csv(output_file, index=False)

else:
    st.error("❌ No valid transactions extracted")