import re
import pandas as pd
from dateutil.parser import parse as parse_date

def clean_ocr_date(date_str):
    """Clean and standardize OCR-extracted dates."""
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
    except:
        return "2025-01-01"

#using regex
def parse_bank_statement_text(text):
    """
    Parses unstructured text from bank statements into structured DataFrame
    Format: Date | Description | Debit | Credit | Balance
    """
    # Normalize spacing
    transactions = []
    lines = text.split("\n")

    # Regular expression for dates and amounts
    date_pattern = r"\d{2}-\d{2}-\d{4}"
    amount_pattern = r"(?:-INR|INR)\s*([\d,]+\.\d{2})"

    current_transaction = {}
    current_description = []
    current_date = None

    for line in lines:
            line = line.strip()
            if not line:
                continue

            # Skip headers and summaries
            if re.match(r"(ACCOUNT DETAILS|ACCOUNT SUMMARY|Summary for|Page \d+ of)", line, re.I):
                continue
            
            # Check for date
            if re.match(date_pattern, line):
                if current_transaction:
                    if current_date and current_description:
                        current_transaction["Date"] = current_date
                        current_transaction["Description"] = " ".join(current_description)
                        transactions.append(current_transaction)
                    current_transaction = {"Debit": 0.0, "Credit": 0.0, "Balance": 0.0}
                    current_description = []
                current_date = line

            # Check for amount (Debit, Credit, or Balance)
            match = re.search(amount_pattern, line)
            if match:
                amount = float(match.group(1).replace(",", ""))
                if "-INR" in line:
                    current_transaction["Debit"] = amount
                elif "INR" in line and not current_transaction.get("Credit"):
                    current_transaction["Credit"] = amount
                else:
                    current_transaction["Balance"] = amount 
            # Treat other lines as part of description
            else:
                current_description.append(line)

    # Append the last transaction
    if current_date and current_description:
        current_transaction["Date"] = current_date
        current_transaction["Description"] = " ".join(current_description)
        transactions.append(current_transaction)
    if not transactions:
        print("[WARNING] No transactions parsed from text")
        return pd.DataFrame()
    df = pd.DataFrame(transactions)
    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors="coerce")
        df = df.dropna(subset=["Date"])
        # Ensure all required columns
        for col in ["Description", "Debit", "Credit", "Balance"]:
            if col not in df.columns:
                df[col] = "" if col == "Description" else 0.0
    return df