import pandas as pd
import re

def parse_transactions(text):
    lines = text.split("\n")
    txns = []
    for line in lines:
        match = re.match(r'(\d{2}\s\w{3}\s\d{4})\s+(.*?)\s+([0-9,]+\.\d{2})\s+([0-9,]+\.\d{2})?\s+([0-9,]+\.\d{2})?', line)
        if match:
            date, desc, debit, credit, balance = match.groups()
            txns.append({
                "Date": date,
                "Description": desc.strip(),
                "Debit": float(debit.replace(",", "")) if debit else 0.0,
                "Credit": float(credit.replace(",", "")) if credit else 0.0,
                "Balance": float(balance.replace(",", "")) if balance else None
            })
    return pd.DataFrame(txns)
