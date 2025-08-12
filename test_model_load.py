import pickle
import os

model_path = r"C:\Users\lavan\OneDrive\Desktop\bank_transaction_tool\models\category_classifier.pkl"
try:
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    print("Model loaded successfully:", model_data.keys())
    print("Vectorizer:", type(model_data["vectorizer"]).__name__)
    print("Classifier:", type(model_data["classifier"]).__name__)
    print("Label Encoder:", model_data["label_encoder"])
except Exception as e:
    print("Error loading model:", str(e))