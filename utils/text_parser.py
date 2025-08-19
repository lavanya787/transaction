import re
import pandas as pd

def parse_bank_statement_text(text):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    data = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Match date in dd-mm-yyyy format with more flexibility
        date_match = re.match(r'(\d{2}-\d{2}-\d{4})', line)
        if date_match:
            date = date_match.group(1)
            i += 1
            description = ""
            # Collect description until an amount or next date is found
            while i < len(lines) and not re.search(r'-?INR\s*[\d,]+\.?\d*|\d{2}-\d{2}-\d{4}', lines[i]):
                description += " " + lines[i].strip()
                i += 1
            if i < len(lines):
                # Look for debit/credit on the current or next few lines
                amount_found = False
                for j in range(i, min(i + 3, len(lines))):
                    amount_line = lines[j].strip()
                    if '-INR' in amount_line:
                        debit_match = re.search(r'-INR\s*([\d,]+\.?\d*)', amount_line)
                        debit = debit_match.group(1).replace(',', '') if debit_match else "0"
                        credit = "0"
                        amount_found = True
                        i = j + 1
                        break
                    elif 'INR' in amount_line and '-INR' not in amount_line:
                        credit_match = re.search(r'INR\s*([\d,]+\.?\d*)', amount_line)
                        credit = credit_match.group(1).replace(',', '') if credit_match else "0"
                        debit = "0"
                        amount_found = True
                        i = j + 1
                        break
                if not amount_found:
                    debit, credit = "0", "0"
                    i += 1
                # Look for balance on the next line
                balance_line = lines[i] if i < len(lines) and re.search(r'INR\s*[\d,]+\.?\d*', lines[i]) else "0"
                balance_match = re.search(r'INR\s*([\d,]+\.?\d*)', balance_line)
                balance = balance_match.group(1).replace(',', '') if balance_match else "0"
                i += 1
                # Validate and append data
                try:
                    data.append([date, description.strip(), float(debit) if debit else 0.0, float(credit) if credit else 0.0, float(balance) if balance else 0.0])
                except ValueError as e:
                    print(f"Warning: Skipping invalid row at line {i} due to {e}")
        else:
            i += 1
    if not data:
        print("Warning: No valid transactions extracted from text")
        return pd.DataFrame(columns=["Date", "Description", "Debit", "Credit", "Balance"])
    df = pd.DataFrame(data, columns=["Date", "Description", "Debit", "Credit", "Balance"])
    return df