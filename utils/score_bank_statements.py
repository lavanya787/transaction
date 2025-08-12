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

def calculate_net_surplus_score(net_surplus, proposed_emi):
    """Calculate score based on net surplus relative to proposed EMI."""
    # Convert Series to scalar if necessary
    if isinstance(net_surplus, pd.Series):
        net_surplus = net_surplus.iloc[0] if not net_surplus.empty else 0
    if isinstance(proposed_emi, pd.Series):
        proposed_emi = proposed_emi.iloc[0] if not proposed_emi.empty else 0

    # Handle NaN values
    if pd.isna(net_surplus) or pd.isna(proposed_emi):
        return 5  # Low score for missing data
    if net_surplus >= 2 * proposed_emi:
        return 25  # Full score for strong surplus
    elif net_surplus >= proposed_emi:
        return 15  # Moderate score
    else:
        return 5  # Low score

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

def calculate_score(monthly_metrics, categorized_file, aggregated_metrics, proposed_emi, loan_purposes):
    """Calculate total score out of 100 with employment and loan purpose modifiers."""
    if monthly_metrics.empty or aggregated_metrics.empty:
        return 0.0, {
            "Income Stability": 0.0,
            "Net Surplus": 0.0,
            "DTI Ratio": 0.0,
            "Expense Discipline": 0.0,
            "Red Flags": 0.0,
            "Employment Stability": False,
            "Loan Purpose Modifier": 0.0
        }

    net_surplus = aggregated_metrics.get("Net Surplus", 0.0)
    dti = aggregated_metrics.get("Debt-to-Income (%)", 100.0)
    red_flag_count = aggregated_metrics.get("Red Flag Count", 0)

    income_stability, is_salary = calculate_income_stability(monthly_metrics, categorized_file or "")
    net_surplus_score = calculate_net_surplus_score(net_surplus, proposed_emi)
    dti_score = calculate_dti_score(dti)
    expense_discipline = calculate_expense_discipline(monthly_metrics, aggregated_metrics)
    red_flag_score = calculate_red_flag_score(red_flag_count)

    employment_bonus = 5.0 if is_salary else 0.0
    file_key = os.path.basename(categorized_file).replace(".csv", "_metrics.csv") if categorized_file else "default_metrics.csv"
    loan_purpose = loan_purposes.get(file_key, ("Neutral", 0.0))[0]
    productive_purposes = ["education", "business", "home improvement"]
    discretionary_purposes = ["luxury", "vacation"]
    loan_purpose_modifier = 5.0 if isinstance(loan_purpose, str) and loan_purpose.lower() in productive_purposes else -5.0 if isinstance(loan_purpose, str) and loan_purpose.lower() in discretionary_purposes else 0.0

    total_score = (
        income_stability * 0.25 +
        net_surplus_score * 0.25 +
        dti_score * 0.20 +
        expense_discipline * 0.15 +
        red_flag_score * 0.15 +
        employment_bonus +
        loan_purpose_modifier
    )
    total_score = max(total_score, 30.0)

    return total_score, {
        "Income Stability": income_stability,
        "Net Surplus": net_surplus_score,
        "DTI Ratio": dti_score,
        "Expense Discipline": expense_discipline,
        "Red Flags": red_flag_score,
        "Employment Stability": is_salary,
        "Loan Purpose Modifier": loan_purpose_modifier
    }

def estimate_proposed_emi(monthly_metrics, proposed_emi_from_file):
    """Use proposed EMI from file or estimate from DTI and income."""
    try:
        proposed_emi = float(proposed_emi_from_file)
        if proposed_emi > 0:
            return proposed_emi
    except (ValueError, TypeError):
        print(f"[WARNING] Invalid Proposed EMI {proposed_emi_from_file}, estimating from DTI")
    if monthly_metrics.empty or "Average Monthly Income" not in monthly_metrics.columns:
        return 0.0
    income = monthly_metrics["Average Monthly Income"].mean()
    dti = monthly_metrics["Debt-to-Income (%)"].mean()
    if pd.isna(income) or pd.isna(dti):
        return 0.0
    return (dti / 100) * income

def score_and_decide(metrics_df, cibil_score, loan_purpose, proposed_emi=0, categorized_file=None, loan_purposes=None):
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

    # Extract metrics with fallbacks
    avg_income = metrics_df["Average Monthly Income"].iloc[0] if "Average Monthly Income" in metrics_df.columns else 0
    avg_expenses = metrics_df["Average Monthly Expenses"].iloc[0] if "Average Monthly Expenses" in metrics_df.columns else 0
    avg_emi = metrics_df["Average Monthly EMI"].iloc[0] if "Average Monthly EMI" in metrics_df.columns else 0
    net_surplus = metrics_df["Net Surplus"].iloc[0] if "Net Surplus" in metrics_df.columns else 0
    dti_ratio = metrics_df["DTI Ratio"].iloc[0] if "DTI Ratio" in metrics_df.columns else 0
    savings_rate = metrics_df["Savings Rate"].iloc[0] if "Savings Rate" in metrics_df.columns else 0
    red_flag_count = metrics_df["Red Flag Count"].iloc[0] if "Red Flag Count" in metrics_df.columns else 0
    discretionary_expenses = metrics_df["Discretionary Expenses"].iloc[0] if "Discretionary Expenses" in metrics_df.columns else 0
    income_stability = metrics_df["Income Stability"].iloc[0] if "Income Stability" in metrics_df.columns else False

    # Prepare aggregated metrics
    aggregated_metrics = {
        "Average Monthly Income": avg_income,
        "Average Monthly Expenses": avg_expenses,
        "Average Monthly EMI": avg_emi,
        "Net Surplus": net_surplus,
        "Debt-to-Income (%)": dti_ratio,
        "Savings Rate (%)": savings_rate,
        "Red Flag Count": red_flag_count,
        "Discretionary Expenses": discretionary_expenses,
        "Income Stability": income_stability
    }

    # Use provided loan_purposes or default
    if loan_purposes is None:
        loan_purposes = {f"{loan_purpose}_metrics.csv": (loan_purpose, proposed_emi)} if categorized_file else {}
    if not categorized_file:
        print("[WARNING] categorized_file is None, using default loan purpose and EMI")
        categorized_file = ""  # Avoid NoneType error in calculate_score

    # Calculate score
    try:
        total_score, components = calculate_score(
            monthly_metrics=metrics_df,
            categorized_file=categorized_file,
            aggregated_metrics=pd.Series(aggregated_metrics),
            proposed_emi=proposed_emi,
            loan_purposes=loan_purposes
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
    except Exception as e:
        print(f"[WARNING] Failed to check CIBIL risk: {e}, assuming Moderate")
        cibil_modifier = 0
    total_score = min(100, max(30, total_score + cibil_modifier))

    # Risk level
    risk_level = "Low" if total_score >= 80 else "Moderate" if total_score >= 60 else "High"

    # Loan purpose impact
    purpose_weights = {"Education": 1.1, "Business": 1.0, "Home Improvement": 1.2, "Luxury": 0.8, "Vacation": 0.7, "Neutral": 1.0}
    purpose_factor = purpose_weights.get(loan_purpose, 1.0)
    adjusted_score = total_score * purpose_factor
    adjusted_score = max(0, min(100, adjusted_score))

    # Decision logic
    new_dti = ((avg_emi + proposed_emi) / avg_income * 100) if avg_income > 0 else 100
    if adjusted_score >= 80 and cibil_score >= 750 and new_dti <= 40:
        action = "Approve with standard terms"
        reason = "High score, good CIBIL, and low DTI"
    elif adjusted_score >= 60 and cibil_score >= 600 and new_dti <= 50:
        action = "Approve with caution"
        reason = "Moderate score, acceptable CIBIL, and DTI"
    else:
        action = "Reject"
        reason = f"Low score ({adjusted_score:.1f}), high DTI ({new_dti:.1f}%), or low CIBIL ({cibil_score})"

    # Adjust for low CIBIL
    if cibil_score < 600 and action == "Reject":
        adjusted_emi = proposed_emi * 0.5
        new_dti_adjusted = ((avg_emi + adjusted_emi) / avg_income * 100) if avg_income > 0 else 100
        if new_dti_adjusted <= 40:
            action = "Approve with higher interest (small loan)"
            reason = f"Low CIBIL ({cibil_score}) with adjusted EMI: ₹{adjusted_emi:,.2f}, DTI: {new_dti_adjusted:.1f}%"

    return {
        "Total Score": round(adjusted_score, 1),
        "Risk Level": risk_level,
        "Action": action,
        "Reason": reason
    }

def score_bank_statement(metrics_df, proposed_emi):
    score = 0
    income_stability = metrics_df["Income Stability"].iloc[0]
    surplus = metrics_df["Net Surplus"].iloc[0]
    dti = metrics_df["DTI Ratio"].iloc[0]
    discretionary_spending = metrics_df["Discretionary Spending Ratio"].iloc[0]
    red_flag_count = metrics_df["Red Flag Count"].iloc[0]

    # Income Stability (25%)
    score += 25 if income_stability else 10
    # Surplus (25%)
    score += 25 if surplus >= 2 * proposed_emi else (15 if surplus >= proposed_emi else 5)
    # DTI (20%)
    score += 20 if dti < 30 else (12 if dti < 50 else 5)
    # Expense Discipline (15%)
    score += 15 if discretionary_spending < 20 else (8 if discretionary_spending < 40 else 2)
    # Red Flags (15%)
    score += 15 if red_flag_count == 0 else (8 if red_flag_count <= 2 else 2)
    
    return score
def score_bank_statements(input_folder, categorized_folder, loan_file, output_file):
    """Process metrics CSVs and calculate scores."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    loan_purposes = load_loan_applications(loan_file)
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
                proposed_emi = loan_purposes.get(file, ("Neutral", 0.0))[1]
                proposed_emi = estimate_proposed_emi(monthly_metrics, proposed_emi)
                total_score, component_scores = calculate_score(monthly_metrics, categorized_file, agg_metrics, proposed_emi, loan_purposes)
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
