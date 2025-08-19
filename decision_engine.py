# decision_engine.py
import pandas as pd
import json
import os
import matplotlib.pyplot as plt


class DecisionEngine:
    """
    Loan Approval Decision Engine
    ---------------------------------
    Uses bank statement metrics + CIBIL score to decide loan approval status,
    with weighted scoring, explainable breakdown, and configurable parameters.
    """

    def __init__(self, config_file=None):
        # Load configuration if available (allows adjusting weights without changing code)
        self.config = self.load_config(config_file)

    def load_config(self, config_file):
        default_config = {
            "weights": {
                "income_stability": 25,
                "surplus": 25,
                "dti": 20,
                "expense_discipline": 15,
                "red_flags": 15
            },
            "purpose_weights": {
                "Education": 1.1, "Business": 1.0, "Home Improvement": 1.2,
                "Luxury": 0.8, "Vacation": 0.7, "Neutral": 1.0
            },
            "thresholds": {
                "approve_high": {"score": 80, "cibil": 750, "dti": 40},
                "approve_moderate": {"score": 60, "cibil": 600, "dti": 50}
            }
        }

        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, "r") as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                print(f"[WARNING] Failed to load config file: {e}. Using defaults.")

        return default_config

    def safe_get(self, df, col, default=0):
        return df[col].iloc[0] if col in df.columns else default

    # ----------------- Loan Decision -----------------    
    def score_and_decide(self, metrics_df, cibil_score, visualize=False):
        if metrics_df.empty:
            return {
                "Total Score": 0,
                "Risk Level": "High",
                "Action": "Reject",
                "Reason": "No valid metrics from bank statement",
                "Score Breakdown": {}
            }
        
        # Extract metrics safely
        avg_income = self.safe_get(metrics_df, "Average Monthly Income")
        avg_expenses = self.safe_get(metrics_df, "Average Monthly Expenses")
        avg_emi = self.safe_get(metrics_df, "Average Monthly EMI")
        net_surplus = self.safe_get(metrics_df, "Net Surplus")
        dti_ratio = self.safe_get(metrics_df, "DTI Ratio")
        savings_rate = self.safe_get(metrics_df, "Savings Rate")
        red_flag_count = self.safe_get(metrics_df, "Red Flag Count")
        discretionary_expenses = self.safe_get(metrics_df, "Discretionary Expenses")
        income_stability = self.safe_get(metrics_df, "Income Stability", False)
        
        w = self.config["weights"]

        # Scoring components
        score_breakdown = {
            "Income Stability": w["income_stability"] if income_stability else 10,
            "DTI": w["dti"] if dti_ratio < 30 else (12 if dti_ratio < 50 else 5),
            "Expense Discipline": w["expense_discipline"] if discretionary_expenses < 0.2 * avg_income else (8 if discretionary_expenses < 0.4 * avg_income else 2),
            "Red Flags": w["red_flags"] if red_flag_count == 0 else (8 if red_flag_count <= 2 else 2)
        }

        total_score = sum(score_breakdown.values())
        
        # Risk classification
        risk_level = "Low" if total_score >= 80 else "Moderate" if total_score >= 60 else "High"

        # Apply loan purpose weight

        # New DTI after loan

        # Decision thresholds
        th = self.config["thresholds"]

        if adjusted_score >= th["approve_high"]["score"] and cibil_score >= th["approve_high"]["cibil"] <= th["approve_high"]["dti"]:
            action = "Approve with standard terms"
            reason = "Strong financial metrics, high CIBIL, and low DTI"
        elif adjusted_score >= th["approve_moderate"]["score"] and cibil_score >= th["approve_moderate"]["cibil"]  <= th["approve_moderate"]["dti"]:
            action = "Approve with caution"
            reason = "Moderate score, acceptable CIBIL, and manageable DTI"
        else:
            action = "Reject"

        # Exception handling: low CIBIL but high repayment capacity
        if cibil_score < 600 and action == "Reject":
            new_dti_adjusted = ((avg_emi) / avg_income * 100) if avg_income > 0 else 100
            if new_dti_adjusted <= 40:
                action = "Approve with higher interest (small loan)"
                reason = f"Low CIBIL ({cibil_score}) but strong repayment capacity"
                adjusted_score = min(70, adjusted_score + 5)

        # Visualization
        if visualize:
            plt.figure(figsize=(8, 5))
            plt.bar(score_breakdown.keys(), score_breakdown.values(), color=["#4caf50", "#2196f3", "#ff9800", "#9c27b0", "#f44336"])
            plt.axhline(20, color="gray", linestyle="--", linewidth=0.7)
            plt.title(f"Loan Scoring Breakdown (Total: {adjusted_score:.1f})", fontsize=14)
            plt.ylabel("Points Awarded")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.show()

        return {
            "Total Score": round(adjusted_score, 1),
            "Risk Level": risk_level,
            "Action": action,
            "Reason": reason,
            "Score Breakdown": score_breakdown,
            "Metrics Used": {
                "Average Monthly Income": avg_income,
                "Average Monthly Expenses": avg_expenses,
                "Average Monthly EMI": avg_emi,
                "Net Surplus": net_surplus,
                "DTI Ratio": dti_ratio,
                "Savings Rate": savings_rate,
                "Red Flag Count": red_flag_count,
                "Discretionary Expenses": discretionary_expenses,
                "Income Stability": income_stability
            }
        }

# Example usage
if __name__ == "__main__":
    engine = DecisionEngine()

    example_metrics = pd.DataFrame([{
        "Average Monthly Income": 95000,
        "Average Monthly Expenses": 40000,
        "Average Monthly EMI": 12000,
        "Net Surplus": 43000,
        "DTI Ratio": 25,
        "Savings Rate": 0.3,
        "Red Flag Count": 0,
        "Discretionary Expenses": 15000,
        "Income Stability": True
    }])

    decision = engine.score_and_decide(example_metrics, cibil_score=730)
    print(json.dumps(decision, indent=2))
