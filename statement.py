from fpdf import FPDF
from faker import Faker
import random
import os
from datetime import datetime, timedelta
from collections import defaultdict
from fpdf.enums import XPos, YPos

fake = Faker("en_IN")
bank_logos = {
    "SBI": "logos/sbi.png",
    "HDFC": "logos/HDFC_Bank_Logo.png",
    "ICICI": "logos/ICICI_Bank_Logo.png",
    "IOB": "logos/Indian_Overseas_Bank_Logo.png",
    "AXIS": "logos/Axis_Bank_logo.png"
}

names = [
    "Arjun Mehta", "Karan Malhotra", "Raghav Iyer", "Sandeep Rathi", "Akshay Deshmukh",
    "Neeraj Sinha", "Harish Patil", "Vivek Menon", "Priya Sharma", "Ayesha Khan",
    "Meera Nambiar", "Ritu Bansal", "Sneha Joshi", "Ananya Rao", "Kavya Pillai",
    "Nandini Agarwal", "Alex Thomas", "Jordan Das", "Sam Roy", "Robin Joseph", "Taylor Fernandes"
]

risk_profiles = ["low", "moderate", "high"]
transaction_categories = {
    "Income": ["Salary", "Freelance", "Bonus"],
    "Fixed": ["Rent", "Electricity", "Phone", "Insurance"],
    "Discretionary": ["Shopping", "Dining", "Entertainment"],
    "Savings": ["FD", "RD", "Mutual Fund"]
}

def load_fonts():
    return {"DejaVu": "fonts/DejaVuSans.ttf"}

class BankStatementPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.date_range_text = ""
        self.bank_name = ""
        self.add_font("DejaVu", "", "fonts/DejaVuSans.ttf")
        self.add_font("DejaVu", "B", "fonts/DejaVuSans-Bold.ttf")
        self.add_font("DejaVu", "I", "fonts/DejaVuSans-Oblique.ttf")

    def header(self):
        # Debug logo path
        logo_path = bank_logos.get(self.bank_name, "")
        print(f"[DEBUG] Attempting to load logo for {self.bank_name}: {logo_path}")
        if logo_path and os.path.exists(logo_path):
            print(f"[DEBUG] Logo file found at {logo_path}, adding to PDF")
            try:
                self.image(logo_path, x=160, y=10, w=40)
            except Exception as e:
                print(f"[ERROR] Failed to add logo {logo_path}: {e}")
                self.set_font("DejaVu", "B", 12)
                self.cell(0, 10, self.bank_name, align="R")
        else:
            print(f"[WARNING] Logo file not found for {self.bank_name} at {logo_path}, using bank name")
            self.set_font("DejaVu", "B", 12)
            self.cell(0, 10, self.bank_name, align="R")
        if self.page_no() == 1:  # Only add full header on first page
            self.set_font("DejaVu", "B", 12)
            self.cell(0, 10, "ACCOUNT STATEMENT", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
            self.set_font("DejaVu", "", 10)
            self.cell(0, 8, self.date_range_text, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
            self.ln(4)
        self._add_watermark()

    def footer(self):
        self.set_y(-15)
        self.set_font("DejaVu", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()} of {{nb}}", align="C")

    def _add_watermark(self):
        self.set_text_color(230, 230, 230)
        self.set_font("DejaVu", "B", 50)
        self.rotate(45, x=self.w / 2, y=self.h / 2)
        self.text(self.w / 3, self.h / 2, "CONFIDENTIAL")
        self.rotate(0)
        self.set_text_color(0, 0, 0)

    def add_account_details(self, name, acc_num, address, branch, ifsc):
        self.set_font("DejaVu", "B", 11)
        self.cell(0, 8, "ACCOUNT DETAILS", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font("DejaVu", "", 10)
        self.cell(0, 6, f"Account Holder Name: {name}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.cell(0, 6, "Account Type: Savings", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.cell(0, 6, f"Account Number: {acc_num}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.multi_cell(0, 6, f"Customer's Address: {address}")
        self.cell(0, 6, f"Branch Name: {branch}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.cell(0, 6, f"IFSC: {ifsc}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.cell(0, 6, "Account Currency: INR", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

    def add_account_summary(self, opening, total_credits, total_debits, closing):
        self.set_font("DejaVu", "B", 11)
        self.cell(0, 8, "ACCOUNT SUMMARY", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font("DejaVu", "", 10)
        self.cell(0, 6, f"Opening Balance: INR {opening:,.2f}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.cell(0, 6, f"Total Credits: INR {total_credits:,.2f}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.cell(0, 6, f"Total Debits: INR {total_debits:,.2f}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.cell(0, 6, f"Ending Balance: INR {closing:,.2f}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

    def add_transaction_header(self):
        self.set_font("DejaVu", "B", 10)
        self.cell(30, 6, "Date", 1, align="C")
        self.cell(70, 6, "Description", 1, align="C")
        self.cell(30, 6, "Debits", 1, align="C")
        self.cell(30, 6, "Credits", 1, align="C")
        self.cell(30, 6, "Balance", 1, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def rotate(self, angle, x=None, y=None):
        from math import cos, sin, radians
        if angle != 0:
            angle = radians(angle)
            c, s = cos(angle), sin(angle)
            if x is None:
                x = self.get_x()
            if y is None:
                y = self.get_y()
            x *= self.k
            y = (self.h - y) * self.k
            self._out(f'q {c:.5f} {s:.5f} {-s:.5f} {c:.5f} {x:.5f} {y:.5f} cm')
        else:
            self._out("Q")

    def add_transaction_row(self, date, detail, debit, credit, balance):
        if not detail:  # Skip empty descriptions
            return
        self.set_font("DejaVu", "", 9)
        line_height = 5
        num_lines = len(self.multi_cell(70, line_height, detail, split_only=True))
        height = line_height * max(1, num_lines)

        # Check for page break
        if self.get_y() + height > self.h - 15:  # Leave room for footer
            self.add_page()
            self.add_transaction_header()

        x_start = self.get_x()
        y_start = self.get_y()

        self.multi_cell(30, height, date, border=1, align="C")
        self.set_xy(x_start + 30, y_start)
        self.multi_cell(70, line_height, detail, border=1)
        self.set_xy(x_start + 100, y_start)
        max_y = max(self.get_y(), y_start + height)
        self.set_y(y_start)

        self.set_xy(x_start + 100, y_start)
        self.cell(30, height, f"-INR {debit:,.2f}" if debit else "", border=1, align="R")
        self.cell(30, height, f"INR {credit:,.2f}" if credit else "", border=1, align="R")
        self.cell(30, height, f"INR {balance:,.2f}", border=1, align="R")
        self.set_y(max_y)

    def add_monthly_summary(self, summary_data, month):
        self.set_font("DejaVu", "B", 10)
        self.cell(0, 10, f"Summary for {month}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font("DejaVu", "", 9)
        self.cell(0, 6, f"Opening Balance: INR {summary_data['opening_balance']:,.2f}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.cell(0, 6, f"Total Credit: INR {summary_data['credit']:,.2f}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.cell(0, 6, f"Total Debit: INR {summary_data['debit']:,.2f}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.cell(0, 6, f"Closing Balance: INR {summary_data['closing_balance']:,.2f}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(5)

def upi_description(name):
    bank = random.choice(["SBIN0000891", "BKID0008203", "HDFC0002345"])
    user = name.lower().replace(" ", "")[:8]
    handle = random.choice(["@okicici", "@oksbi", "@okhdfcbank"])
    ref = random.randint(100000000000, 999999999999)
    masked_account = str(random.randint(10000000, 99999999))
    return f"{bank}/{name}/{masked_account}/{user}{handle}/UPI/{ref}/UPI/BRANCH"

def generate_transactions(start_balance, risk):
    transactions = []
    balance = float(start_balance)
    total_debit, total_credit = 0, 0
    monthly_summaries = defaultdict(lambda: {
        "opening_balance": 0,
        "debit": 0,
        "credit": 0,
        "closing_balance": 0
    })

    weights = {
        "low": [4, 3, 2, 1],
        "moderate": [3, 3, 3, 1],
        "high": [2, 2, 5, 1]
    }

    start_date = datetime(2025, 2, 1)
    end_date = datetime(2025, 8, 11)
    num_days = (end_date - start_date).days
    total_txns = random.randint(80, 120)

    prev_month = None
    for _ in range(total_txns):
        tx_date = start_date + timedelta(days=random.randint(0, num_days))
        month_key = tx_date.strftime("%b %Y")
        category = random.choices(list(transaction_categories.keys()), weights[risk])[0]
        detail = upi_description(fake.first_name())
        tag = f"[{random.choice(transaction_categories[category])}]"
        txn_id = f"TXN{random.randint(10000000, 99999999)}"
        utr = str(random.randint(100000000000, 999999999999))
        debit = credit = 0

        if category == "Income":
            credit = random.randint(5000, 20000)
            balance += credit
            total_credit += credit
        else:
            debit = random.randint(300, 5000)
            balance -= debit
            total_debit += debit

        full_detail = f"{detail} {tag}\nTxnID: {txn_id} | UTR: {utr}"
        transactions.append({
            "date": tx_date,
            "detail": full_detail,
            "debit": debit,
            "credit": credit,
            "balance": round(balance, 2),
            "month": month_key
        })

        if prev_month != month_key:
            if prev_month is not None:
                monthly_summaries[prev_month]["closing_balance"] = balance
            monthly_summaries[month_key]["opening_balance"] = balance - credit + debit
            prev_month = month_key

        monthly_summaries[month_key]["debit"] += debit
        monthly_summaries[month_key]["credit"] += credit
        monthly_summaries[month_key]["closing_balance"] = balance

    transactions.sort(key=lambda x: x["date"])
    return transactions, monthly_summaries, total_credit, total_debit, balance

def generate_statement(name, risk):
    bank_name = random.choice(list(bank_logos.keys()))
    pdf = BankStatementPDF()
    pdf.bank_name = bank_name
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Generate transaction data
    opening_balance = round(random.uniform(1000, 100000), 2)
    txns, monthly_summaries, total_credit, total_debit, final_balance = generate_transactions(opening_balance, risk)

    if not txns:
        print("No transactions were generated.")
        return

    # Set date range
    start_date = txns[0]["date"]
    end_date = txns[-1]["date"]
    pdf.date_range_text = f"Period: {start_date.strftime('%d %b %Y')} - {end_date.strftime('%d %b %Y')}"

    # Add bank details
    acc_num = fake.random_number(digits=10, fix_len=True)
    addr = fake.address().replace("\n", ", ")
    branch = fake.city().upper()
    ifsc = "IDIB000" + ''.join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=4))

    pdf.add_account_details(name, acc_num, addr, branch, ifsc)
    pdf.add_account_summary(opening_balance, total_credit, total_debit, final_balance)
    pdf.add_transaction_header()

    # Group transactions by month
    txns_by_month = defaultdict(list)
    for txn in txns:
        txns_by_month[txn["month"]].append(txn)

    # Add transactions and monthly summaries
    for month, monthly_txns in txns_by_month.items():
        pdf.set_font("DejaVu", "B", 12)
        pdf.cell(0, 10, f"{month}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        pdf.add_transaction_header()
        for txn in monthly_txns:
            pdf.add_transaction_row(
                txn["date"].strftime("%d-%m-%Y"),
                txn["detail"],
                txn["debit"],
                txn["credit"],
                txn["balance"]
            )
        pdf.add_monthly_summary(monthly_summaries[month], month)

    # Save PDF
    if not os.path.exists("statements"):
        os.makedirs("statements")
    filename = f"statements/{name.replace(' ', '_')}_{risk}.pdf"
    pdf.output(filename)
    print(f"Generated statement for {name} with risk {risk}: {filename}")

# Generate statements for users
for name in names[:30]:
    risk = random.choice(risk_profiles)
    generate_statement(name, risk)

print("✅ Bank statements generated in the 'statements' folder.")