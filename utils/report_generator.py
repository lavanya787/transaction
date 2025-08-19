# generate_reports.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class BankReportGenerator:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df["Month"] = self.df["Date"].dt.to_period("M")
        self.df["Year"] = self.df["Date"].dt.year
    
    # ================= A. SUMMARY & TRENDS =================
    def monthly_quarterly_summary(self):
        monthly = self.df.groupby("Month").agg({
            "Credit": "sum", "Debit": "sum"
        }).reset_index()
        monthly["Net Cash Flow"] = monthly["Credit"] - monthly["Debit"]

        quarterly = self.df.groupby(self.df["Date"].dt.to_period("Q")).agg({
            "Credit": "sum", "Debit": "sum"
        }).reset_index()
        quarterly["Net Cash Flow"] = quarterly["Credit"] - quarterly["Debit"]

        return monthly, quarterly
    
    def rolling_averages(self):
        monthly, _ = self.monthly_quarterly_summary()
        monthly["Rolling 3M Income"] = monthly["Credit"].rolling(3).mean()
        monthly["Rolling 3M Expense"] = monthly["Debit"].rolling(3).mean()
        return monthly
    
    def yoy_comparison(self):
        return self.df.groupby("Year").agg({
            "Credit": "sum", "Debit": "sum"
        }).reset_index()
    
    # ================= B. CATEGORY INSIGHTS =================
    def category_distribution(self):
        spending = self.df.groupby("Category")["Debit"].sum()
        income = self.df.groupby("Category")["Credit"].sum()
        total_expense = spending.sum()
        total_income = income.sum()
        return (spending / total_expense * 100).sort_values(ascending=False), \
               (income / total_income * 100).sort_values(ascending=False)
    
    def category_trends(self):
        return self.df.groupby(["Month", "Category"]).agg({
            "Credit": "sum", "Debit": "sum"
        }).reset_index()
    
    # ================= C. RISK & BEHAVIOR =================
    def risk_metrics(self, low_balance_threshold=5000):
        avg_income = self.df["Credit"].mean()
        income_stability = np.std(self.df["Credit"]) / avg_income if avg_income else 0
        expense_ratio = self.df["Debit"].sum() / self.df["Credit"].sum() if self.df["Credit"].sum() else 0
        savings_rate = (self.df["Credit"].sum() - self.df["Debit"].sum()) / self.df["Credit"].sum() if self.df["Credit"].sum() else 0
        low_balance_days = (self.df["Balance"] < low_balance_threshold).sum()
        overdrafts = (self.df["Balance"] < 0).sum()
        high_risk_txns = self.df[(self.df["Debit"] > 50000) |
                                 (self.df["Category"].str.contains("Loan|Gambling", case=False, na=False))]
        
        return {
            "Income Stability Index": round(income_stability, 3),
            "Expense-to-Income Ratio": round(expense_ratio, 3),
            "Savings Rate": round(savings_rate, 3),
            "Low Balance Days": int(low_balance_days),
            "Overdraft Count": int(overdrafts),
            "High Risk Transactions": high_risk_txns
        }
    
    # ================= D. CASH FLOW HEALTH =================
    def cash_flow_health(self):
        monthly, _ = self.monthly_quarterly_summary()
        negative_months = monthly[monthly["Net Cash Flow"] < 0]
        avg_balance = self.df["Balance"].mean()
        avg_expense = self.df["Debit"].mean()
        liquidity_ratio = avg_balance / avg_expense if avg_expense else 0
        return {
            "Negative Months": len(negative_months),
            "Liquidity Ratio": round(liquidity_ratio, 2),
            "Avg Monthly Balance": round(avg_balance, 2)
        }
    
    # ================= E. TRANSACTION INSIGHTS =================
    def top_transactions(self, n=10):
        return (self.df.nlargest(n, "Debit")[["Date", "Description", "Debit"]],
                self.df.nlargest(n, "Credit")[["Date", "Description", "Credit"]])
    
    def recurring_and_one_off(self):
        recurring = self.df[self.df["Recurring"] == "Yes"]
        one_off_high = self.df[(self.df["Debit"] > 50000) & (self.df["Recurring"] == "No")]
        return recurring, one_off_high
    
    # ================= F. COMPLIANCE & VERIFICATION =================
    def account_verification(self, expected_name=None, expected_ifsc=None):
        return {
            "Account Holder Match": expected_name.lower() in str(self.df.get("Account Holder", "")).lower() if expected_name else None,
            "IFSC Match": expected_ifsc.lower() in str(self.df.get("IFSC", "")).lower() if expected_ifsc else None
        }
    
    def salary_verification(self, min_months=3):
        salary_txns = self.df[self.df["Category"].str.lower() == "salary"]
        return len(salary_txns) >= min_months
    
    # ================= G. VISUALIZATION HELPERS =================
    def plot_cash_flow(self):
        monthly, _ = self.monthly_quarterly_summary()
        plt.figure(figsize=(8,5))
        plt.plot(monthly["Month"].astype(str), monthly["Net Cash Flow"], marker='o')
        plt.xticks(rotation=45)
        plt.title("Net Cash Flow Trend")
        plt.xlabel("Month")
        plt.ylabel("Net Cash Flow (INR)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    df = pd.read_csv("categorized_transactions.csv", parse_dates=["Date"])
    br = BankReportGenerator(df)
    print("Income vs Expense:", br.category_distribution())
    print("Risk Metrics:", br.risk_metrics())
    print("Cash Flow Health:", br.cash_flow_health())
    br.plot_cash_flow()
