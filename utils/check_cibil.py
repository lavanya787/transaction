import pandas as pd
import os
import numpy as np
#risk_level = "Moderate"

def check_cibil_risk(cibil_score, bank_score=0.0):
    """Determine risk level and action based on CIBIL and bank score."""
    try:
        if cibil_score >= 750 and bank_score >= 70:
            return "Low", "Approve with favorable terms"
        elif cibil_score >= 700 and bank_score >= 60:
            return "Moderate", "Approve with standard terms"
        elif cibil_score >= 650 and bank_score >= 50:
            return "Moderate", "Approve with conditions"
        else:
            return "High", "Reject"
    except Exception as e:
        print(f"[ERROR] Failed to check CIBIL risk: {e}")
        return "High", "Reject"

def load_cibil_scores(cibil_file):
    """Load CIBIL scores from CSV (replace with API call in future)."""
    if not os.path.exists(cibil_file):
        print(f"[WARNING] CIBIL scores file {cibil_file} not found, skipping files without scores")
        return {}
    try:
        cibil_df = pd.read_csv(cibil_file)
        return dict(zip(cibil_df["File"], cibil_df["CIBIL Score"]))
    except Exception as e:
        print(f"[WARNING] Failed to load CIBIL scores: {e}, skipping files without scores")
        return {}

def load_loan_applications(loan_file):
    """Load loan application data for collateral and co-applicant CIBIL."""
    if not os.path.exists(loan_file):
        print(f"[WARNING] Loan applications file {loan_file} not found, assuming no collateral/co-applicant")
        return {}
    try:
        loan_df = pd.read_csv(loan_file)
        return dict(zip(loan_df["File"], zip(loan_df["Collateral Available"], loan_df["Co-applicant CIBIL"])))
    except Exception as e:
        print(f"[WARNING] Failed to load loan applications: {e}, assuming no collateral/co-applicant")
        return {}

def check_exception(bank_scores, file_name, cibil_score, monthly_metrics_file):
    """Check for exception: low CIBIL but strong bank statement."""
    if cibil_score >= 600:
        return False, None
    try:
        monthly_metrics = pd.read_csv(monthly_metrics_file)
        incomes = monthly_metrics["Average Monthly Income"]
        dti = monthly_metrics["Debt-to-Income (%)"].mean()
        net_surplus = monthly_metrics["Net Surplus"].sum()
        recurring_amount = (dti / 100) * incomes.mean() if not pd.isna(dti) and incomes.mean() != 0 else 0.0
        cv = incomes.std() / incomes.mean() if len(incomes) >= 2 and incomes.mean() != 0 else float('inf')
        if cv < 0.3 and dti < 30 and net_surplus >= 2 * recurring_amount:
            return True, "Approve smaller loan with higher interest rate"
        return False, None
    except Exception as e:
        print(f"[WARNING] Failed to check exception for {file_name}: {e}")
        return False, None

def check_cibil(bank_scores_file, cibil_file, metrics_folder, loan_file, output_file):
    """Apply CIBIL-based decision logic with collateral/co-applicant checks."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    cibil_scores = load_cibil_scores(cibil_file)
    loan_applications = load_loan_applications(loan_file)
    
    try:
        bank_scores = pd.read_csv(bank_scores_file)
    except Exception as e:
        print(f"[ERROR] Failed to load bank scores: {e}")
        with open("outputs/cibil_decisions.log", "a") as f:
            f.write(f"Error: Failed to load bank scores: {e}\n")
        return

    decisions = []
    for _, row in bank_scores.iterrows():
        file_name = row["File"]
        bank_score = row["Total Score"]
        cibil_score = cibil_scores.get(file_name)
        collateral_available, co_applicant_cibil = loan_applications.get(file_name, ("N", "N/A"))

        # Handle missing CIBIL score
        if cibil_score is None:
            action = "Pending (No CIBIL score)"
            risk_level = "Unknown"
        else:
            # Check for collateral or strong co-applicant
            has_collateral = collateral_available == "Y"
            has_strong_co_applicant = co_applicant_cibil != "N/A" and pd.to_numeric(co_applicant_cibil, errors='coerce') >= 750

            # Check for exception if CIBIL < 600
            if cibil_score < 600:
                metrics_file = os.path.join(metrics_folder, file_name)
                is_exception, exception_action = check_exception(bank_scores, file_name, cibil_score, metrics_file)
                if is_exception:
                    action = exception_action
                    risk_level = "High (Exception)"
                elif has_collateral or has_strong_co_applicant:
                    action = "Approve with collateral/co-applicant"
                    risk_level = "High (Mitigated)"
                else:
                    action = "Reject or require co-applicant/guarantor"
                    risk_level = "High"
            # Standard decision rules
            elif bank_score >= 80 and cibil_score >= 750:
                action = "Approve with favorable terms"
                risk_level = "Low"
            elif 60 <= bank_score <= 79 and 600 <= cibil_score < 750:
                action = "Approve with caution"
                risk_level = "Moderate"
            else:
                if has_collateral or has_strong_co_applicant:
                    action = "Approve with collateral/co-applicant"
                    risk_level = "Moderate/High (Mitigated)"
                else:
                    action = "Reject or require co-applicant/guarantor"
                    risk_level = "Moderate" if cibil_score >= 600 else "High"

        decisions.append({
            "File": file_name,
            "CIBIL Score": cibil_score if cibil_score is not None else "N/A",
            "Bank Score": bank_score,
            "Risk Level": risk_level,
            "Action": action,
            "Collateral Available": collateral_available,
            "Co-applicant CIBIL": co_applicant_cibil
        })
        print(f"✅ Processed {file_name}: CIBIL={cibil_score if cibil_score is not None else 'N/A'}, Bank Score={bank_score:.2f}, Risk={risk_level}, Action={action}")
        with open("outputs/cibil_decisions.log", "a") as f:
            f.write(f"File: {file_name}, CIBIL: {cibil_score if cibil_score is not None else 'N/A'}, Bank Score: {bank_score:.2f}, Risk: {risk_level}, Action: {action}, Collateral: {collateral_available}, Co-applicant CIBIL: {co_applicant_cibil}\n")

    if decisions:
        decisions_df = pd.DataFrame(decisions)
        decisions_df.to_csv(output_file, index=False)
        print(f"✅ Decisions saved to {output_file}")
    else:
        print("[WARNING] No decisions made")
        with open("outputs/cibil_decisions.log", "a") as f:
            f.write("Error: No decisions made\n")

if __name__ == "__main__":
    bank_scores_file = "outputs/bank_scores.csv"
    cibil_file = "cibil_scores.csv"
    metrics_folder = "outputs/metrics"
    loan_file = "loan_applications.csv"
    output_file = "outputs/final_decisions.csv"
    check_cibil(bank_scores_file, cibil_file, metrics_folder, loan_file, output_file)
