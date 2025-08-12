import joblib

def load_model():
    return joblib.load("models/risk_classifier.pkl")

def classify_risk(metrics):
    model = load_model()
    X = [[
        metrics["Average Monthly Income"],
        metrics["Average Monthly Expense"],
        metrics["DTI Ratio"],
        metrics["Net Surplus"]
    ]]
    return model.predict(X)[0]
