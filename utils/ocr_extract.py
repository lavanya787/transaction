import pandas as pd
import pdfplumber
import pytesseract
from PIL import Image
import numpy as np
import cv2
import os
import io
import docx2txt
import re
import chardet
from dateutil.parser import parse as parse_date
from datetime import datetime
import fitz  # PyMuPDF
from paddleocr import PaddleOCR
#from utils.text_parser import parse_bank_statement_text
import sys
import os
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
    return pd.DataFrame(columns=["Date", "Description", "Debit", "Credit", "Balance"])

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
def parse_bank_statement_text(text):
    """Parse text from TXT/DOC/PDF into a DataFrame."""
    lines = text.split("\n")
    data = []
    for line in lines:
        if line.strip():
            parts = re.split(r"\s{2,}|\t", line.strip())
            if len(parts) >= 4:
                data.append(parts[:5])
    columns = ["Date", "Description", "Debit", "Credit", "Balance"]
    df = pd.DataFrame(data, columns=columns[:len(data[0])])
    if not df.empty:
        df["Date"] = df["Date"].apply(clean_ocr_date)
        df["Debit"] = pd.to_numeric(df["Debit"], errors='coerce').fillna(0.0)
        df["Credit"] = pd.to_numeric(df["Credit"], errors='coerce').fillna(0.0)
        df["Balance"] = pd.to_numeric(df["Balance"], errors='coerce').fillna(0.0)
    return df

def extract_bank_statement(input_file, output_file, file_type="pdf"):
    """Extract and clean bank statement data."""
    try:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file {input_file} not found")
        if os.path.getsize(input_file) == 0:
            raise ValueError(f"Input file {input_file} is empty")
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
        
        if df.empty:
            print(f"[WARNING] Empty file after processing: {input_file}")
            with open("outputs/extraction_errors.log", "a") as f:
                f.write(f"Empty file after processing: {input_file}\n")
            return
        required_columns = ["Date", "Description", "Debit", "Credit", "Balance"]
        if not all(col in df.columns for col in required_columns):
            print(f"[ERROR] Missing required columns in {input_file}: {df.columns}")
            with open("outputs/extraction_errors.log", "a") as f:
                f.write(f"Missing columns in {input_file}: {df.columns}\n")
            return
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"✅ Extracted and cleaned: {output_file}")
    except Exception as e:
        print(f"[ERROR] Failed to process {input_file}: {e}")
        with open("outputs/extraction_errors.log", "a") as f:
            f.write(f"Failed to process {input_file}: {e}\n")

# Normalize amount (remove commas, INR, etc.)
def clean_amount(val):
    try:
        val = str(val).replace(',', '').replace('INR', '').replace('₹', '').strip()
        return float(val) if val else 0.0
    except:
        return 0.0

def extract_text_from_pdf(pdf_path):
    """Extract text from a text-based PDF using PyMuPDF."""
    try:
        with fitz.open(pdf_path) as doc:
            text = "\n".join([page.get_text() for page in doc])
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def extract_df_from_scanned_pdf(pdf_data):
    """Extract data from a scanned PDF using PaddleOCR."""
    try:
        # Save PDF data to a temporary file
        temp_pdf = "temp.pdf"
        with open(temp_pdf, "wb") as f:
            f.write(pdf_data)
        
        # Open PDF and process each page
        data = []
        with fitz.open(temp_pdf) as doc:
            for page in doc:
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                # Convert PIL Image to OpenCV format for PaddleOCR
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                ocr_result = ocr.ocr(img_cv, cls=True)
                
                # Extract text from OCR result
                text_lines = [line[1][0] for line in ocr_result[0] if line[1][0]]
                page_text = "\n".join(text_lines)
                df = parse_text_to_transactions(page_text)
                data.append(df)
        
        # Clean up
        if os.path.exists(temp_pdf):
            os.remove(temp_pdf)
        
        # Combine data from all pages
        if data:
            return pd.concat(data, ignore_index=True).dropna(subset=["Date", "Balance"])
        return pd.DataFrame(columns=["Date", "Description", "Debit", "Credit", "Balance"])
    except Exception as e:
        print(f"Error in OCR extraction: {e}")
        return pd.DataFrame(columns=["Date", "Description", "Debit", "Credit", "Balance"])

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
