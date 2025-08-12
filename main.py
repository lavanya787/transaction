import os
from utils.ocr_extract import extract_text_from_pdf
from utils.preprocess import parse_transactions
from utils.calculate_metrics import calculate_metrics
from utils.report_generator import generate_report
from utils.classify import classify_risk
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\lavan\Downloads\tesseract-ocr-w64-setup-5.5.0.20241111.exe'

pdf_dir = "statements"
csv_dir = "extracted_csvs"
report_dir = "processed_reports"

for file in os.listdir(pdf_dir):
    if file.endswith(".pdf"):
        text = extract_text_from_pdf(os.path.join(pdf_dir, file))
        df = parse_transactions(text)
        df.to_csv(os.path.join(csv_dir, file.replace(".pdf", ".csv")), index=False)

        metrics = calculate_metrics(df)
        risk = classify_risk(metrics)
        print(f"File: {file}, Risk Level: {risk}")

        generate_report(metrics, os.path.join(report_dir, file.replace(".pdf", "_report.pdf")))
