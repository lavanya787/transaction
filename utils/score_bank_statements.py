import pandas as pd
import os
import numpy as np
from utils.check_cibil import check_cibil_risk

def load_loan_applications(loan_file):
    """Load loan application data for loan purpose and proposed EMI."""
    if not os.path.exists(loan_file):
        print(f"[WARNING] Loan applications file {loan_file} not found, assuming neutral loan purpose and zero EMI")
        return {}
    try:
        loan_df = pd.read_csv(loan_file)
        if "Proposed EMI" not in loan_df.columns:
            print(f"[WARNING] 'Proposed EMI' column missing, defaulting to 0.0")
            loan_df["Proposed EMI"] = 0.0
        loan_df["Proposed EMI"] = pd.to_numeric(loan_df["Proposed EMI"], errors='coerce').fillna(0.0)
        return dict(zip(loan_df["File"], zip(loan_df["Loan Purpose"], loan_df["Proposed EMI"])))
    except Exception as e:
        print(f"[WARNING] Failed to load loan applications: {e}, assuming neutral loan purpose and zero EMI")
        return {}

def calculate_income_stability(monthly_metrics, categorized_file):
    """Calculate income stability score with employment check (25 points)."""
    if monthly_metrics.empty or "Average Monthly Income" not in monthly_metrics.columns:
        return 0.0, False
    incomes = monthly_metrics["Average Monthly Income"]
    if len(incomes) < 2 or incomes.mean() == 0:
        return 5.0, False
    cv = incomes.std() / incomes.mean()
    is_salary = False
    try:
        if os.path.exists(categorized_file):
            categorized_df = pd.read_csv(categorized_file)
            income_desc = categorized_df[categorized_df["Category"] == "Income"]["Description"].str.lower()
            salary_keywords = ["salary", "wages", "payroll"]
            is_salary = any(any(kw in desc for kw in salary_keywords) for desc in income_desc)
        else:
            print(f"[WARNING] Categorized file {categorized_file} not found")
    except Exception as e:
        print(f"[WARNING] Failed to check employment stability: {e}")
    score = 25.0 if cv < 0.3 else 15.0 if cv < 0.6 else 5.0
    return score, is_salary

def calculate_net_surplus_score(net_surplus):
    """Calculate score based on net surplus relative to proposed EMI."""
    # Convert Series to scalar if necessary
    if isinstance(net_surplus, pd.Series):
        net_surplus = net_surplus.iloc[0] if not net_surplus.empty else 0

def calculate_dti_score(dti):
    """Calculate DTI score (20 points)."""
    if pd.isna(dti):
        return 0.0
    if dti < 40:
        return 20.0
    elif dti <= 60:
        return 10.0
    else:
        return 5.0

def calculate_expense_discipline(monthly_metrics, aggregated_metrics):
    """Calculate expense discipline score (15 points)."""
    if monthly_metrics.empty or "Average Monthly Expenses" not in monthly_metrics.columns:
        return 0.0
    total_expenses = monthly_metrics["Average Monthly Expenses"]
    income = monthly_metrics["Average Monthly Income"]
    discretionary = monthly_metrics.get("Discretionary Expenses", total_expenses * 0.2)
    if total_expenses.mean() == 0 or income.mean() == 0:
        return 0.0
    proportion = (discretionary / income).mean()
    if proportion < 0.2:
        return 15.0
    elif proportion <= 0.4:
        return 7.5
    else:
        return 2.5

def calculate_red_flag_score(red_flag_count):
    """Calculate red flag score (15 points)."""
    if pd.isna(red_flag_count):
        return 0.0
    if red_flag_count == 0:
        return 15.0
    elif red_flag_count <= 2:
        return 7.5
    else:
        return 2.5

def calculate_score(monthly_metrics, categorized_file, aggregated_metrics):
    """Calculate score based on metrics."""
    score = 0
    components = []
    
    # Income and surplus
    if aggregated_metrics["Average Monthly Income"] > 50000:
        score += 20
        components.append("High income")
    
    # DTI and savings
    if aggregated_metrics["DTI Ratio"] <= 40:
        score += 20
        components.append("Low DTI")
    if aggregated_metrics["Savings Rate"] > 10:
        score += 10
        components.append("Good savings rate")
    
    # Red flags and stability
    if aggregated_metrics["Red Flag Count"] == 0:
        score += 20
        components.append("No red flags")
    if aggregated_metrics["Income Stability"]:
        score += 10
        components.append("Stable income")
    
    return score, components

def score_and_decide(metrics_df, cibil_score, categorized_file=None):
    """
    Score bank statements and decide loan eligibility based on metrics and CIBIL score.
    """
    if metrics_df.empty:
        return {
            "Total Score": 0,
            "Risk Level": "High",
            "Action": "Reject",
            "Reason": "No valid metrics available"
        }

    # Extract metrics with safe fallbacks
    dti_ratio = float(metrics_df['DTI Ratio'].iloc[0]) if 'DTI Ratio' in metrics_df.columns else 0.0
    avg_balance = float(metrics_df['Average Closing Balance'].iloc[0]) if 'Average Closing Balance' in metrics_df.columns else 0.0
    bounced_cheques = int(metrics_df['Bounced Cheques Count'].iloc[0]) if 'Bounced Cheques Count' in metrics_df.columns else 0
    emi_payments = float(metrics_df['High-Cost EMI Payments'].iloc[0]) if 'High-Cost EMI Payments' in metrics_df.columns else 0.0
    negative_balance_days = int(metrics_df['Negative Balance Days'].iloc[0]) if 'Negative Balance Days' in metrics_df.columns else 0

    avg_income = metrics_df["Average Monthly Income"].iloc[0] if "Average Monthly Income" in metrics_df.columns else 0.0
    avg_expenses = metrics_df["Average Monthly Expenses"].iloc[0] if "Average Monthly Expenses" in metrics_df.columns else 0.0
    net_surplus = metrics_df["Net Surplus"].iloc[0] if "Net Surplus" in metrics_df.columns else (avg_income - avg_expenses)
    dti_ratio = metrics_df["DTI Ratio"].iloc[0] if "DTI Ratio" in metrics_df.columns else (
        (avg_expenses / avg_income * 100) if avg_income > 0 else 0.0
    )
    savings_rate = metrics_df["Savings Rate"].iloc[0] if "Savings Rate" in metrics_df.columns else 0.0
    red_flag_count = metrics_df["Red Flag Count"].iloc[0] if "Red Flag Count" in metrics_df.columns else 0
    discretionary_expenses = metrics_df["Discretionary Expenses"].iloc[0] if "Discretionary Expenses" in metrics_df.columns else 0.0
    income_stability = metrics_df["Income Stability"].iloc[0] if "Income Stability" in metrics_df.columns else False
    avg_monthly_balance = metrics_df["Average Monthly Balance"].iloc[0] if "Average Monthly Balance" in metrics_df.columns else 0.0
    cash_withdrawals = metrics_df["Cash Withdrawals"].iloc[0] if "Cash Withdrawals" in metrics_df.columns else 0.0
    open_credit_accounts = metrics_df["Number of Open Credit Accounts"].iloc[0] if "Number of Open Credit Accounts" in metrics_df.columns else 0

    # Prepare aggregated metrics
    aggregated_metrics = {
        "Average Monthly Income": float(avg_income),
        "Average Monthly Expenses": float(avg_expenses),
        "Net Surplus": float(net_surplus),
        "DTI Ratio": float(dti_ratio),   # Keep consistent key
        "Savings Rate": float(savings_rate),
        "Red Flag Count": int(red_flag_count),
        "Discretionary Expenses": float(discretionary_expenses),
        "Income Stability": bool(income_stability),
        "Average Monthly Balance": float(avg_monthly_balance),
        "Cash Withdrawals": float(cash_withdrawals),
        "Number of Open Credit Accounts": int(open_credit_accounts)
    }

    if not categorized_file:
        print("[WARNING] categorized_file is None, using default loan purpose and EMI")
        categorized_file = ""  # Avoid NoneType error in calculate_score

    # Calculate score
    try:
        total_score, components = calculate_score(
            monthly_metrics=metrics_df,
            categorized_file=categorized_file,
            aggregated_metrics=pd.Series(aggregated_metrics)
        )
    except Exception as e:
        print(f"[ERROR] Failed to calculate score: {e}")
        return {
            "Total Score": 0,
            "Risk Level": "High",
            "Action": "Reject",
            "Reason": f"Error calculating score: {e}"
        }

    # Adjust score with CIBIL
    try:
        cibil_risk = check_cibil_risk(cibil_score)  # Assume returns "Low", "Moderate", or "High"
        cibil_modifier = 10 if cibil_risk == "Low" else 0 if cibil_risk == "Moderate" else -10
        components.append(f"CIBIL risk adjustment: {cibil_risk}")
    except Exception as e:
        print(f"[WARNING] Failed to check CIBIL risk: {e}, assuming Moderate")
        cibil_modifier = 0
    total_score = min(100, max(30, total_score + cibil_modifier))

    # Additional adjustments
    if cash_withdrawals < 0.1 * avg_income:
        total_score += 5
        components.append("Low cash withdrawals")
    if open_credit_accounts <= 3:
        total_score += 5
        components.append("Low number of open credit accounts")
    if income_stability:
        total_score += 5
        components.append("Stable income")

    # Ensure bounds
    total_score = min(100, max(30, total_score))

    # Risk level
    risk_level = "Low" if total_score >= 80 else "Moderate" if total_score >= 60 else "High"
    if (dti_ratio < 40 and cibil_score >= 700 and bounced_cheques == 0 and
        avg_balance >= 2 * emi_payments):
        action = "Approve"
        risk_level = "Low"
        reason = "Low DTI, high CIBIL, no bounces, sufficient balance"
    elif (dti_ratio > 50 or cibil_score < 600 or bounced_cheques > 1 or
          negative_balance_days > 0):
        action = "Reject"
        risk_level = "High"
        reason = "High DTI, low CIBIL, multiple bounces, or negative balances"
    else:
        action = "Review Manually"
        risk_level = "Moderate"
        reason = "Borderline metrics (DTI 40-50%, CIBIL 600-700)"
    return {
        "Total Score": total_score,
        "Risk Level": risk_level,
        "Action": action,
        "Reason": reason
    }
    # Final decision
    if risk_level == "Low":
        action = "Approve with standard terms"
    elif risk_level == "Moderate":
        action = "Approve with caution"
    else:
        action = "Reject"

    return {
        "Total Score": round(total_score, 1),
        "Risk Level": risk_level,
        "Action": action,
        "Reason": f"Score {total_score:.1f}, CIBIL {cibil_score}, Net Surplus {net_surplus:.2f}, DTI {dti_ratio:.2f}",
        "Components": components
    }

def score_bank_statements(input_folder, categorized_folder, loan_file, output_file):
    """Process metrics CSVs and calculate scores."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    scores = []

    if not os.path.exists(input_folder):
        print(f"[ERROR] Metrics folder {input_folder} not found")
        with open("outputs/scoring_errors.log", "a") as f:
            f.write(f"Error: Metrics folder {input_folder} not found\n")
        return

    for file in os.listdir(input_folder):
        if file.endswith("_metrics.csv") and file != "combined_metrics.csv":
            try:
                monthly_metrics = pd.read_csv(os.path.join(input_folder, file))
                base_name = file.replace("_metrics.csv", ".csv")
                categorized_file = os.path.join(categorized_folder, base_name)
                agg_metrics = {
                    "Average Monthly Income": monthly_metrics["Average Monthly Income"].mean(),
                    "Average Monthly Expenses": monthly_metrics["Average Monthly Expenses"].mean(),
                    "Net Surplus": monthly_metrics["Net Surplus"].sum(),
                    "Debt-to-Income (%)": monthly_metrics["Debt-to-Income (%)"].mean(),
                    "Savings Rate (%)": monthly_metrics["Savings Rate (%)"].mean(),
                    "Red Flag Count": monthly_metrics["Red Flag Count"].sum(),
                    "Discretionary Expenses": monthly_metrics["Discretionary Expenses"].mean(),
                    "Outliers Detected": monthly_metrics["Outliers Detected"].sum()
                }
                agg_metrics = pd.Series(agg_metrics)
                total_score, component_scores = calculate_score(monthly_metrics, categorized_file, agg_metrics)
                scores.append({
                    "File": file,
                    "Total Score": round(total_score, 2),
                    **component_scores
                })
                print(f"✅ Scored {file}: {total_score:.2f}/100")
                with open("outputs/scoring_details.log", "a") as f:
                    f.write(f"File: {file}, Total Score: {total_score:.2f}, Components: {component_scores}\n")
            except Exception as e:
                print(f"⚠️ Failed to score {file}: {e}")
                scores.append({
                    "File": file,
                    "Total Score": 0.0,
                    "Income Stability": 0.0,
                    "Net Surplus": 0.0,
                    "DTI Ratio": 0.0,
                    "Expense Discipline": 0.0,
                    "Red Flags": 0.0,
                    "Employment Stability": False,
                    "Loan Purpose Modifier": 0.0
                })
                with open("outputs/scoring_errors.log", "a") as f:
                    f.write(f"File: {file}, Error: {e}\n")
    
    if scores:
        scores_df = pd.DataFrame(scores)
        scores_df.to_csv(output_file, index=False)
        print(f"✅ Scores saved to {output_file}")
    else:
        print("[WARNING] No scores calculated")
        with open("outputs/scoring_errors.log", "a") as f:
            f.write("Error: No scores calculated\n")

if __name__ == "__main__":
    input_dir = "outputs/metrics"
    categorized_dir = "categorized_csvs"
    loan_file = "loan_applications.csv"
    output_file = "outputs/bank_scores.csv"
    score_bank_statements(input_dir, categorized_dir, loan_file, output_file)
