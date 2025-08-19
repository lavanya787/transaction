# extract_transactions.py
"""
Extract transactions from bank statement file (PDF, image, CSV, XLSX).
Only handles OCR/text parsing and returns a raw DataFrame.
"""
import pdfplumber
import re
import pandas as pd
import numpy as np
import os
import io
import cv2
import docx2txt
import chardet
from datetime import datetime
import fitz
from PIL import Image
import pytesseract
from paddleocr import PaddleOCR
import sys
from dateutil import parser as parse_date
from utils.text_parser import parse_bank_statement_text

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Normalize date format
def normalize_date(date_str):
    date_str = date_str.strip() if isinstance(date_str, str) else ""
    for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%d-%b-%Y", "%d %b %Y"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except:
            continue
    return None

# Normalize amount (remove commas, INR, etc.)
def clean_amount(val):
    try:
        val = str(val).replace(',', '').replace('INR', '').replace('₹', '').strip()
        return float(val) if val else 0.0
    except:
        return 0.0
    
def clean_ocr_date(date_str):
    date_str = str(date_str).strip()
    date_str = re.sub(r"[Oo]", "0", date_str)
    date_str = re.sub(r"[IiLl]", "1", date_str)
    date_str = re.sub(r"[^0-9a-zA-Z\s/-:]", "", date_str)
    try:
        parts = re.split(r"[-/:\s]", date_str)
        if len(parts) >= 3 and parts[1].isdigit() and int(parts[1]) > 12:
            parts[1] = "12"
            date_str = "/".join(parts[:3])
        parsed_date = parse_date(date_str, fuzzy=True, dayfirst=True)
        return parsed_date.strftime("%Y-%m-%d")
    except Exception as e:
        with open("outputs/invalid_dates.log", "a") as f:
            f.write(f"Invalid date '{date_str}' in file: {e}\n")
        return "2025-01-01"
    
MODE_KEYWORDS = {
    "NEFT": ["neft", "rtgs"],
    "IMPS": ["imps"],
    "UPI": ["upi", "gpay", "googlepay", "phonepe", "paytm"],
    "CHEQUE": ["cheque", "check", "chq"],
    "ATM": ["atm", "withdrawal"],
    "CARD": ["card", "visa", "mastercard", "rupay"],
    "WALLET": ["wallet", "mobikwik", "paytm wallet"],
    "TRANSFER": ["transfer", "to account", "from account", "credited from"],
}

def detect_mode_from_description(desc: str):
    d = desc.lower()
    for mode, kws in MODE_KEYWORDS.items():
        if any(kw in d for kw in kws):
            return mode
    return ""

def extract_text_from_image(path):
    img = Image.open(path)
    return pytesseract.image_to_string(img)
            
def extract_transactions(input_file, output_file=None, file_type="pdf"):
    """Extract and clean bank statement data."""
    try:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file {input_file} not found")
        if os.path.getsize(input_file) == 0:
            raise ValueError(f"Input file {input_file} is empty")

        # Handle file parsing
        if file_type == "pdf":
            with fitz.open(input_file) as doc:
                text = "\n".join([page.get_text() for page in doc])
                if text.strip():
                    df = parse_bank_statement_text(text)
                else:
                    df = extract_df_from_scanned_pdf(input_file)
        elif file_type == "csv":
            raw = open(input_file, "rb").read()
            encoding = chardet.detect(raw)["encoding"]
            df = pd.read_csv(io.StringIO(raw.decode(encoding)))
        elif file_type == "txt":
            raw = open(input_file, "rb").read()
            encoding = chardet.detect(raw)["encoding"]
            text = raw.decode(encoding)
            df = parse_bank_statement_text(text)
        elif file_type in ["doc", "docx"]:
            text = docx2txt.process(input_file)
            df = parse_bank_statement_text(text)
        else:
            raise ValueError("Unsupported file format")

        # Check results
        if df.empty:
            print(f"[WARNING] Empty file after processing: {input_file}")
            with open("outputs/extraction_errors.log", "a") as f:
                f.write(f"Empty file after processing: {input_file}\n")
            return pd.DataFrame()

        required_columns = ["Date", "Description", "Debit", "Credit", "Balance"]
        if not all(col in df.columns for col in required_columns):
            print(f"[ERROR] Missing required columns in {input_file}: {df.columns}")
            with open("outputs/extraction_errors.log", "a") as f:
                f.write(f"Missing columns in {input_file}: {df.columns}\n")
            return pd.DataFrame()

        # Save only if path is provided
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            df.to_csv(output_file, index=False)
            print(f"✅ Extracted and cleaned: {output_file}")

        return df

    except Exception as e:
        print(f"[ERROR] Failed to process {input_file}: {e}")
        with open("outputs/extraction_errors.log", "a") as f:
            f.write(f"Failed to process {input_file}: {e}\n")
        return pd.DataFrame()

def extract_df_from_scanned_pdf(pdf_data):
    try:
        import pytesseract
        from PIL import Image
        with fitz.open(stream=pdf_data, filetype="pdf") as doc:
            for page_num in range(doc.page_count):
                image_list = doc.get_page_images(page_num)
                if image_list:
                    xref = image_list[0][0]  # Get first image reference
                    base_image = doc.extract_image(xref)
                    image = Image.open(io.BytesIO(base_image["image"]))
                    text = pytesseract.image_to_string(image)
                    if text.strip():
                        return parse_bank_statement_text(text)
            return pd.DataFrame(columns=["Date", "Description", "Debit", "Credit"])
    except Exception as e:
        print(f"OCR Error: {e}")
        return pd.DataFrame(columns=["Date", "Description", "Debit", "Credit"])
    
def parse_text_to_transactions(text: str) -> pd.DataFrame:
    """Parse OCR text into a structured DataFrame."""
    lines = text.strip().split("\n")
    data = []
    
    # Updated regex to match OCR output from Priya_Sharma_low.pdf
    transaction_pattern = re.compile(
        r"(\d{2}-\d{2}-\d{4})\s+(.+?)\s+(?:-?INR\s+([\d,]+\.\d{2})|-)?\s+(?:INR\s+([\d,]+\.\d{2})|-)?\s+(?:INR\s+([\d,]+\.\d{2}))"
    )
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = transaction_pattern.search(line)
        if match:
            date, description, debit, credit, balance = match.groups()
            data.append({
                "Date": normalize_date(date),
                "Description": description.strip(),
                "Debit": clean_amount(debit) if debit else 0.0,
                "Credit": clean_amount(credit) if credit else 0.0,
                "Balance": clean_amount(balance)
            })
    
    df = pd.DataFrame(data)
    return df

def extract_pdf_table(pdf_path):
    """Extract table from a text-based PDF using pdfplumber, fallback to OCR."""
    all_data = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                table = page.extract_table()
                if table:
                    headers = [h.strip().lower() for h in table[0]]
                    for row in table[1:]:
                        if len(row) < 5:
                            continue
                        all_data.append({
                            'Date': normalize_date(row[0]),
                            'Description': row[1].strip() if row[1] else "",
                            'Debit': clean_amount(row[2]),
                            'Credit': clean_amount(row[3]),
                            'Balance': clean_amount(row[4])
                        })
        df = pd.DataFrame(all_data)
        if not df.empty:
            return df
    except Exception as e:
        print(f"[OCR fallback] Extracting {pdf_path} via PaddleOCR...")
    
    # Fallback to OCR
    with open(pdf_path, "rb") as f:
        pdf_data = f.read()
    return extract_df_from_scanned_pdf(pdf_data)

def extract_from_csv_or_excel(path):
    """Read and normalize CSV or Excel files."""
    try:
        if path.endswith(".csv"):
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)
        cols = [c.lower() for c in df.columns]
        df.columns = cols

        # Normalize columns
        df['Date'] = df['date'].apply(normalize_date)
        df['Description'] = df['description'].astype(str)
        df['Debit'] = df['debit'].apply(clean_amount)
        df['Credit'] = df['credit'].apply(clean_amount)
        df['Balance'] = df['balance'].apply(clean_amount)

        return df[['Date', 'Description', 'Debit', 'Credit', 'Balance']].dropna(subset=["Date", "Balance"])
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return pd.DataFrame(columns=["Date", "Description", "Debit", "Credit", "Balance"])

def process_folder(input_folder, output_folder):
    """Process all files in the input folder and save as CSVs."""
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file)
        if file.endswith(".pdf"):
            df = extract_pdf_table(file_path)
        elif file.endswith((".csv", ".xlsx")):
            df = extract_from_csv_or_excel(file_path)
        else:
            continue

        if not df.empty:
            df.dropna(subset=["Date", "Balance"], inplace=True)
            out_file = os.path.join(output_folder, file.replace(".pdf", ".csv").replace(".xlsx", ".csv"))
            df.to_csv(out_file, index=False)
            print(f"✅ Processed: {file} → {out_file}")
        else:
            print(f"⚠️ No data extracted from {file}")

if __name__ == "__main__":
    input_dir = "statements"
    output_dir = "cleaned_csvs"
    process_folder(input_dir, output_dir)