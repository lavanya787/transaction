**Overview**
This project is an AI-powered tool for analyzing bank statements to evaluate loan eligibility. It combines insights from CIBIL scores and bank statement analysis to make informed lending decisions, reducing EMI defaults. The tool supports data extraction from various formats (CSV, PDF, TXT, DOC), transaction categorization using machine learning and rule-based methods, metric calculation (e.g., income stability, DTI ratio, red flags), scoring, and decision-making. It is designed to comply with RBI guidelines for credit reporting and data protection.

**Key features include:**
**Bank Statement Generation:** Generate realistic bank statement PDFs with logos, transaction data, and summaries for testing.
**Data Extraction:** Parse transactions from PDFs, CSVs, TXT, and DOC files using OCR and text parsing.
**Transaction Categorization:** Automate categorization with ML, rules, and manual overrides for accuracy.
**Metric Calculation:** Compute financial metrics like income assessment, expense patterns, surplus capacity, red flags, and savings behavior.
**Scoring and Decision Engine:** Combine CIBIL score and bank metrics to generate a composite score and loan decision.
**Streamlit Dashboard:** User-friendly interface for uploading statements, configuring parameters, and viewing results.
**Visualization:** Charts for transaction categories, income trends, and surplus trends.

The tool is built with Python, using libraries like FPDF for PDF generation, PyMuPDF for parsing, scikit-learn for ML categorization, and Streamlit for the UI.

**Features**
**PDF Generation:** Create synthetic bank statements with customizable risk profiles (low, moderate, high) and Indian banking context (e.g., UPI, NEFT).
**Multi-Format Parsing:** Support for CSV, PDF, TXT, DOC/DOCX, with OCR for scanned PDFs.

**Transaction Categorization:**
Rule-based matching for common patterns.
ML classification using TF-IDF and RandomForest.

**Financial Metrics:**
Income assessment (stability, source, average).
Expense patterns (recurring, discretionary).
Surplus and DTI ratio.
Red flags (overdrafts, bounces).
Savings behavior (rate, emergency fund).


**Loan Scoring:** Weighted scoring model (25% income stability, 25% surplus, 20% DTI, 15% expense discipline, 15% red flags).
**Decision Making:** Approve, approve with caution, or reject based on score and CIBIL thresholds, with co-applicant support.
**Visualizations:** Pie charts for categories, line plots for income/surplus trends.
**Compliance:** Data encryption, consent checkbox, RBI guideline adherence.

**Requirements**
Python 3.9+
Dependencies (install via pip install -r requirements.txt)

**Download spaCy model:** python -m spacy download en_core_web_sm


**Fonts and Logos:** Place DejaVu fonts in fonts/ and bank logos in logos/ folders.
**Models:** The category_classifier.pkl is generated during training; a default is provided.

**Installation**

**Clone the repository:**
git clone https://github.com/your-repo/bank-transaction-tool.git
cd bank-transaction-tool

**Install dependencies:**
pip install -r requirements.txt
python -m spacy download en_core_web_sm

**Run the Streamlit app:**
streamlit run app.py

**For PDF generation:**
python statement.py

Outputs PDFs in statements/ folder.

**Usage**

**Generate Bank Statements:**
Run statement.py to create synthetic PDFs for testing.
Customize names, risk profiles, and transaction categories in the script.

**Run the Streamlit App:**
Upload a bank statement file (CSV, PDF, TXT, DOC).
Input CIBIL score, proposed EMI, loan purpose, and co-applicant status.
The app parses, categorizes, calculates metrics, and provides a loan decision.
View visualizations and export results.


**Batch Processing (CLI):**
Use categorize_transactions.py to process all CSVs in a folder:
textpython categorize_transactions.py

Outputs categorized CSVs in categorized_csvs/ and metrics in outputs/metrics/.


