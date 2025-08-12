import matplotlib.pyplot as plt
from fpdf import FPDF

def generate_report(metrics: dict, out_file: str):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "Bank Statement Financial Report", ln=True, align='C')

    for k, v in metrics.items():
        pdf.cell(200, 10, f"{k}: {round(v, 2)}", ln=True)

    pdf.output(out_file)
