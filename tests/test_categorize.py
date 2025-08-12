import unittest
from utils.categorize_transactions import categorize_description, load_category_rules

class TestCategorization(unittest.TestCase):
    def setUp(self):
        self.rules = load_category_rules()

    def test_income_keywords(self):
        self.assertEqual(categorize_description("Salary credited via NEFT"), "Income")
        self.assertEqual(categorize_description("IMPS-CR PAYROLL"), "Income")

    def test_fixed_expenses_keywords(self):
        self.assertEqual(categorize_description("Paid house rent for July"), "Fixed_Expenses")
        self.assertEqual(categorize_description("LIC Insurance premium paid"), "Fixed_Expenses")

    def test_discretionary_keywords(self):
        self.assertEqual(categorize_description("Swiggy order #TXN123"), "Discretionary_Expenses")
        self.assertEqual(categorize_description("Uber ride from Airport"), "Discretionary_Expenses")

    def test_savings_keywords(self):
        self.assertEqual(categorize_description("Invested in Axis SIP"), "Savings")
        self.assertEqual(categorize_description("Fixed Deposit created"), "Savings")

    def test_red_flags_keywords(self):
        self.assertEqual(categorize_description("ATM withdrawal failed - reversal"), "Red_Flags")
        self.assertEqual(categorize_description("Late fee applied"), "Red_Flags")

    def test_uncategorized_fallback(self):
        self.assertEqual(categorize_description("Some unknown transaction xyz"), "Uncategorized")

    def test_valid_category_set(self):
        valid_categories = set(self.rules.keys()).union({"Uncategorized"})
        example_inputs = [
            "Salary via NEFT", "Rent to landlord", "Zomato lunch", "SIP Mutual Fund", "Overdraft penalty", "UNKNOWN TEXT"
        ]
        for inp in example_inputs:
            cat = categorize_description(inp)
            self.assertIn(cat, valid_categories, f"{inp} ➝ Invalid category: {cat}")

if __name__ == '__main__':
    unittest.main()
