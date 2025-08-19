"""
Advanced, self-learning transaction categorizer.

Works with:
  - Frontend upload -> ocr_extract.py -> (DataFrame) -> categorize_transactions.py
  - Or directly with labeled CSV/XLSX for training.

Features:
  - Models: 'lr' (SGD Logistic Regression), 'xgb' (XGBoost), 'hybrid' (LR + XGB Voting), 'bert' (Transformer)
  - Confidence scores for non-BERT; BERT softmax scores
  - Incremental self-learning: partial_fit (LR) + warm start (XGB)
  - No hardcoded paths: all functions accept paths/DFs and optional model_dir
  - Handles new/unseen categories by refitting label encoder (and BERT head)
  - HashingVectorizer + amount feature for classic models; BERT uses tokenizer
  - Optional external vectorizer/classifier for compatibility with app.py

Install deps (as needed):
  pip install scikit-learn xgboost joblib transformers torch scipy pandas numpy

Typical wiring:
  df = extract_data(file_path)                # from ocr_extract.py
  trained = TransactionCategorizer().train(df_labeled, model_type='hybrid')
  out_df = TransactionCategorizer().predict(df_unlabeled, model_type='hybrid')
"""

import os
import io
import re
import json
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Union, List, Dict, Tuple
from datetime import datetime

# Classic ML
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
from scipy.sparse import hstack

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False
# Load category maps
BASE_DIR = os.path.dirname(__file__)
with open(os.path.join(BASE_DIR, "category_tag_map.json"), "r") as f:
    CATEGORY_TAG_MAP = json.load(f)

with open(os.path.join(BASE_DIR, "keyword_map.json"), "r") as f:
    KEYWORD_MAP = json.load(f)

# Optional Transformers / Torch
try:
    import torch
    from transformers import (
        BertTokenizer,
        BertForSequenceClassification,
        Trainer,
        TrainingArguments,
    )
    _HAS_BERT = True
except Exception:
    _HAS_BERT = False

# Create or load a vectorizer
vectorizer = HashingVectorizer(n_features=2**18, alternate_sign=False)
# ---------------------------
# Utilities
# ---------------------------
def _default_model_dir(model_dir: Optional[str]) -> str:
    """Resolve model directory."""
    if model_dir:
        return model_dir
    return os.environ.get("MODEL_DIR", "models")

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _clean_text(s: str) -> str:
    s = str(s) if s is not None else ""
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _require_cols(df: pd.DataFrame, cols: List[str], ctx: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{ctx}: missing required columns {missing}. Available: {list(df.columns)}")

def _maybe_numeric(s):
    try:
        return float(s)
    except Exception:
        return 0.0

def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

# ... (rest of the imports and TransactionCategorizer class remain as provided earlier)
def rule_based_categorize(desc: str) -> str:
    """Categorize a transaction description using regex and keyword maps."""
    if not isinstance(desc, str):
        return "Uncategorized"
    desc = desc.lower()

    # ✅ Pass 1: Regex-based matching from category_tag_map
    for category, patterns in CATEGORY_TAG_MAP.items():
        for pattern in patterns:
            if re.search(pattern, desc, re.IGNORECASE):
                return category

    # ✅ Pass 2: Direct keyword lookup from keyword_map
    for keyword, category in KEYWORD_MAP.items():
        if keyword in desc:
            return category

    # ✅ Default fallback
    return "Uncategorized"


def categorize_transactions(df, tc, model_type="hybrid", add_confidence=False):
    """Categorize transactions using hybrid ML and rule-based approach."""
    if df.empty:
        print("[WARNING] Empty DataFrame in categorize_transactions")
        return df

    df = df.copy()
    if "Category" not in df.columns:
        df["Category"] = "Uncategorized"

    if model_type == "hybrid" and tc.vectorizer and tc.classifier:
        try:
            # ML-based categorization
            X = tc.vectorizer.transform(df["Description"].astype(str))
            predictions = tc.classifier.predict(X)
            confidences = tc.classifier.predict_proba(X).max(axis=1) if add_confidence else None
            df["Category"] = predictions
            if add_confidence:
                df["Confidence"] = confidences
        except Exception as e:
            print(f"[WARNING] ML categorization failed: {e}, falling back to rule-based")
            df["Category"] = df["Description"].apply(rule_based_categorize)
    else:
        print("[WARNING] Model not trained. Using only rule-based categories.")
        df["Category"] = df["Description"].apply(rule_based_categorize)

    if add_confidence and "Confidence" not in df.columns:
        df["Confidence"] = 1.0  # Default confidence for rule-based

    print(f"[DEBUG] Categorized df types: {df.dtypes}")
    return df
# ---------------------------
# Core Categorizer
# ---------------------------
class TransactionCategorizer:
    """
    Advanced self-learning categorizer with optional external vectorizer/classifier support.

    Public API:
      - train(labeled: Union[str, pd.DataFrame], model_type='hybrid', model_dir=None, cv_folds=5)
      - predict(unlabeled_df: pd.DataFrame, model_type='hybrid', model_dir=None) -> pd.DataFrame
      - update(new_labeled: Union[str, pd.DataFrame], model_type='hybrid', model_dir=None)

    Expected df columns:
      - For training: Description, Amount, Category (strings)
      - For prediction: Description, Amount (Category will be added)
    """

    def __init__(self, vectorizer=None, classifier=None, keyword_map_file="keyword_map.json", category_tag_file="category_tag_map.json"):
        """
        Initialize with optional external vectorizer and classifier for compatibility with app.py.
        
        Args:
            vectorizer (HashingVectorizer or TfidfVectorizer, optional): Pre-trained vectorizer.
            classifier (SGDClassifier or RandomForestClassifier or VotingClassifier, optional): Pre-trained classifier.
            keyword_map_file (str): Path to keyword mapping JSON file.
            category_tag_file (str): Path to category tag mapping JSON file.
        """
        self.vectorizer = vectorizer if vectorizer else HashingVectorizer(n_features=2**18, alternate_sign=False)
        self.classifier = classifier
        self.keyword_map = self._load_json(keyword_map_file)
        self.category_tag_map = self._load_json(category_tag_file)

    def _load_json(self, path):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    # ---------- IO helpers ----------
    def _paths(self, model_type: str, model_dir: Optional[str]) -> Dict[str, str]:
        base = _default_model_dir(model_dir)
        _ensure_dir(base)
        if model_type == "xgb":
            model_path = os.path.join(base, "xgb_model.json")
        elif model_type == "bert":
            model_path = base
        else:
            model_path = os.path.join(base, f"{model_type}_model.pkl")

        label_path = os.path.join(base, "label_encoder.pkl")
        meta_path = os.path.join(base, "model_meta.json")
        return {"model": model_path, "labels": label_path, "meta": meta_path, "dir": base}

    def _assign_tag(self, category):
        for tag, categories in self.category_tag_map.items():
            if category in categories:
                return tag
        return "Uncategorized"

    def _save_meta(self, meta_path: str, data: Dict):
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _load_meta(self, meta_path: str) -> Dict:
        if not os.path.exists(meta_path):
            return {}
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ---------- Preprocess ----------
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        _require_cols(df, ["Description"], "preprocess")
        df["Description"] = df["Description"].fillna("").map(_clean_text)
        if "Amount" not in df.columns:
            if {"Credit", "Debit"}.issubset(df.columns):
                df["Amount"] = df["Credit"].fillna(0).map(_maybe_numeric) - df["Debit"].fillna(0).map(_maybe_numeric)
            else:
                df["Amount"] = 0.0
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
        return df

    # ---------- Model builders ----------
    def _build_lr(self) -> SGDClassifier:
        return SGDClassifier(loss="log_loss", random_state=42)

    def _build_xgb(self, num_classes: int) -> XGBClassifier:
        if not _HAS_XGB:
            raise RuntimeError("XGBoost not installed. Install with: pip install xgboost")
        return XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            num_class=num_classes,
            random_state=42,
            tree_method="hist",
            eval_metric="mlogloss",
            enable_categorical=False,
        )

    def _build_bert(self, num_labels: int):
        if not _HAS_BERT:
            raise RuntimeError("Transformers/Torch not installed. Install with: pip install transformers torch")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
        return tokenizer, model

    # ---------- Vectorize ----------
    def _vectorize(self, texts: List[str], amounts: np.ndarray):
        if isinstance(self.vectorizer, (HashingVectorizer, TfidfVectorizer)):
            X_text = self.vectorizer.transform(texts)
            X = hstack((X_text, amounts.reshape(-1, 1)))
            return X
        raise ValueError("Unsupported vectorizer type. Use HashingVectorizer or TfidfVectorizer.")

    def _detect_recurring(self, df, min_occurrences=3, date_window=40):
        recurring_flags = []
        recurring_descs = set()
        grouped = df.groupby("Category")
        for cat, group in grouped:
            if len(group) >= min_occurrences:
                sorted_dates = sorted(group["Date"].dropna().unique())
                if len(sorted_dates) >= 2:
                    gaps = [(sorted_dates[i+1] - sorted_dates[i]).days for i in range(len(sorted_dates)-1)]
                    if gaps and all(abs(g - np.median(gaps)) <= date_window for g in gaps):
                        recurring_descs.add(cat)
        for _, row in df.iterrows():
            recurring_flags.append("Yes" if row["Category"] in recurring_descs else "No")
        df["Recurring"] = recurring_flags
        return df

    # ---------- Train ----------
    def train(
        self,
        labeled: Union[str, pd.DataFrame, io.BytesIO],
        model_type: str = "hybrid",
        model_dir: Optional[str] = None,
        cv_folds: int = 5,
    ) -> Dict[str, Union[str, float]]:
        if isinstance(labeled, (str, os.PathLike)):
            if str(labeled).lower().endswith(".csv"):
                df = pd.read_csv(labeled)
            else:
                df = pd.read_excel(labeled)
        elif isinstance(labeled, pd.DataFrame):
            df = labeled
        else:
            raise ValueError("Unsupported 'labeled' input; pass DataFrame or file path.")

        df = self.preprocess(df)
        _require_cols(df, ["Category"], "train")
        texts = df["Description"].tolist()
        amounts = df["Amount"].values
        labels = df["Category"].astype(str).tolist()

        paths = self._paths(model_type, model_dir)

        label_encoder = LabelEncoder()
        label_encoder.fit(labels)
        joblib.dump(label_encoder, paths["labels"])

        y = label_encoder.transform(labels)
        classes_count = len(label_encoder.classes_)

        meta = {"model_type": model_type, "trained_at": _timestamp(), "num_classes": classes_count}
        self._save_meta(paths["meta"], meta)

        if model_type == "bert":
            if not _HAS_BERT:
                raise RuntimeError("Transformers/Torch not installed.")
            tokenizer, model = self._build_bert(num_labels=classes_count)
            class _TorchDS(torch.utils.data.Dataset):
                def __init__(self, texts, labels, tokenizer, max_len=128):
                    self.texts = texts
                    self.labels = labels
                    self.tokenizer = tokenizer
                    self.max_len = max_len
                def __len__(self): return len(self.texts)
                def __getitem__(self, idx):
                    t = str(self.texts[idx])
                    enc = tokenizer(
                        t,
                        add_special_tokens=True,
                        max_length=self.max_len,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    )
                    return {
                        "input_ids": enc["input_ids"].squeeze(0),
                        "attention_mask": enc["attention_mask"].squeeze(0),
                        "labels": torch.tensor(self.labels[idx], dtype=torch.long),
                    }
            ds = _TorchDS(texts, y, tokenizer)
            args = TrainingArguments(
                output_dir=paths["dir"],
                num_train_epochs=3,
                per_device_train_batch_size=16,
                save_total_limit=2,
                logging_dir=os.path.join(paths["dir"], "logs"),
                learning_rate=2e-5,
                evaluation_strategy="no",
            )
            trainer = Trainer(model=model, args=args, train_dataset=ds)
            trainer.train()
            model.save_pretrained(paths["dir"])
            tokenizer.save_pretrained(paths["dir"])
            return {"status": "ok", "model_dir": paths["dir"], "model_type": model_type}

        X = self._vectorize(texts, amounts)

        lr = self._build_lr()
        if model_type == "lr":
            model = lr
        elif model_type == "xgb":
            if not _HAS_XGB:
                raise RuntimeError("XGBoost not installed.")
            model = self._build_xgb(num_classes=classes_count)
        elif model_type == "hybrid":
            if not _HAS_XGB:
                raise RuntimeError("Hybrid needs xgboost.")
            xgb = self._build_xgb(num_classes=classes_count)
            model = VotingClassifier(estimators=[("lr", lr), ("xgb", xgb)], voting="soft")
        else:
            raise ValueError("Unsupported model_type. Choose 'lr' | 'xgb' | 'hybrid' | 'bert'.")

        try:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
            print(f"[CV] {model_type} accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
        except Exception as e:
            print(f"[CV] skipped due to error: {e}")

        model.fit(X, y)

        if model_type == "xgb":
            model.save_model(paths["model"])
        else:
            joblib.dump(model, paths["model"])

        return {
            "status": "ok",
            "model_path": paths["model"],
            "labels_path": paths["labels"],
            "model_type": model_type,
        }

    # ---------- Predict ----------
        # ---------- Predict ----------
    def predict(
        self,
        unlabeled_df: pd.DataFrame,
        model_type: str = "hybrid",
        model_dir: Optional[str] = None,
        add_confidence: bool = True,
    ) -> pd.DataFrame:
        """
        Predict categories for new transactions.
        Requires columns ['Description','Amount'].
        Returns df with 'Category' and optionally 'Confidence'.
        """
        df = self.preprocess(unlabeled_df)
        texts = df["Description"].tolist()
        amounts = df["Amount"].values

        paths = self._paths(model_type, model_dir)
        label_encoder = None
        if os.path.exists(paths["labels"]):
            try:
                label_encoder = joblib.load(paths["labels"])
            except Exception as e:
                print(f"Warning: Failed to load label encoder from {paths['labels']}: {e}")
        else:
            print(f"Warning: Label encoder file {paths['labels']} not found. Training required.")
        
        if label_encoder is None:
            raise ValueError("Label encoder not available. Please train the model first using the train method.")

        categories, tags, sources, confidences = [], [], [], []
        
        if not hasattr(self, "label_encoder") or self.label_encoder is None:
            raise ValueError("⚠️ Model not trained. Please run `train()` or `load_model()` before prediction.")

        # Pass 1: Keyword-based mapping
        for desc in texts:
            cat = self._keyword_lookup(desc)
            if cat:
                categories.append(cat)
                tags.append(self._assign_tag(cat))
                sources.append("Keyword")
                confidences.append(1.0)
            else:
                categories.append(None)
                tags.append(None)
                sources.append(None)
                confidences.append(None)
        
        # Pass 2: ML prediction for missing categories
        idx_to_predict = [i for i, c in enumerate(categories) if c is None]
        if idx_to_predict:
            X_texts = [texts[i] for i in idx_to_predict]
            X_amounts = np.array([amounts[i] for i in idx_to_predict])
            
            if self.classifier:
                # Use only text features for pre-trained classifier from app.py
                X_text = self.vectorizer.transform(X_texts)
                pred_ids = self.classifier.predict(X_text)
                if add_confidence and hasattr(self.classifier, "predict_proba"):
                    conf = np.max(self.classifier.predict_proba(X_text), axis=1)
                elif add_confidence and hasattr(self.classifier, "decision_function"):
                    dec = self.classifier.decision_function(X_text)
                    if dec.ndim == 1:
                        dec = np.vstack([-dec, dec]).T
                    conf = np.max(_softmax(dec), axis=1)
                else:
                    conf = None
            elif model_type == "bert":
                if not _HAS_BERT:
                    raise RuntimeError("Transformers/Torch not installed.")
                tokenizer = BertTokenizer.from_pretrained(paths["dir"])
                model = BertForSequenceClassification.from_pretrained(paths["dir"])
                model.eval()
                preds, confs = [], []
                for t in texts:
                    enc = tokenizer(t, return_tensors="pt", padding=True, truncation=True, max_length=128)
                    with torch.no_grad():
                        logits = model(**enc).logits
                        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    pred_id = int(np.argmax(probs))
                    preds.append(pred_id)
                    confs.append(float(np.max(probs)))
                pred_ids = np.array(preds)
                conf = np.array(confs)
            else:
                # Use text + amount features for internal models
                X = self._vectorize(X_texts, X_amounts)
                if model_type == "xgb":
                    if not _HAS_XGB:
                        raise RuntimeError("xgboost not available.")
                    model = XGBClassifier()
                    model.load_model(paths["model"])
                else:
                    model = joblib.load(paths["model"])
                pred_ids = model.predict(X)
                if add_confidence and hasattr(model, "predict_proba"):
                    conf = np.max(model.predict_proba(X), axis=1)
                elif add_confidence and hasattr(model, "decision_function"):
                    dec = model.decision_function(X)
                    if dec.ndim == 1:
                        dec = np.vstack([-dec, dec]).T
                    conf = np.max(_softmax(dec), axis=1)
                else:
                    conf = None

            # Assign predictions and confidence
            for i, idx in enumerate(idx_to_predict):
                categories[idx] = label_encoder.inverse_transform([pred_ids[i]])[0]
                tags[idx] = self._assign_tag(categories[idx])
                sources[idx] = "ML" if self.classifier else model_type.upper()
                confidences[idx] = conf[i] if conf is not None else 1.0

        df["Category"] = categories
        df["Tag"] = tags
        df["Source"] = sources
        if add_confidence:
            df["Confidence"] = confidences
        if "Date" in df.columns and "Category" in df.columns:
            df = self._detect_recurring(df)
        return df
    # ---------- Update (self-learning) ----------
    def update(
        self,
        new_labeled: Union[str, pd.DataFrame],
        model_type: str = "hybrid",
        model_dir: Optional[str] = None,
    ) -> Dict[str, str]:
        if isinstance(new_labeled, (str, os.PathLike)):
            if str(new_labeled).lower().endswith(".csv"):
                df = pd.read_csv(new_labeled)
            else:
                df = pd.read_excel(new_labeled)
        elif isinstance(new_labeled, pd.DataFrame):
            df = new_labeled
        else:
            raise ValueError("Unsupported 'new_labeled' input; pass DataFrame or file path.")

        df = self.preprocess(df)
        _require_cols(df, ["Category"], "update")
        texts = df["Description"].tolist()
        amounts = df["Amount"].values
        labels = df["Category"].astype(str).tolist()

        paths = self._paths(model_type, model_dir)
        if not os.path.exists(paths["labels"]):
            raise FileNotFoundError("Label encoder not found. Train first.")

        label_encoder: LabelEncoder = joblib.load(paths["labels"])
        existing = set(label_encoder.classes_)
        new_all = sorted(existing | set(labels))
        if new_all != list(label_encoder.classes_):
            print("[INFO] New categories detected; refitting label encoder.")
            label_encoder.fit(new_all)
            joblib.dump(label_encoder, paths["labels"])
        y = label_encoder.transform(labels)
        num_classes = len(label_encoder.classes_)

        if model_type == "bert":
            if not _HAS_BERT:
                raise RuntimeError("Transformers/Torch not installed.")
            tokenizer = BertTokenizer.from_pretrained(paths["dir"])
            model = BertForSequenceClassification.from_pretrained(paths["dir"])
            if model.config.num_labels != num_classes:
                print("[INFO] Resizing BERT classifier for new classes.")
                model.classifier = torch.nn.Linear(model.config.hidden_size, num_classes)
                model.config.num_labels = num_classes
            class _TorchDS(torch.utils.data.Dataset):
                def __init__(self, texts, labels, tokenizer, max_len=128):
                    self.texts, self.labels, self.tok, self.max_len = texts, labels, tokenizer, max_len
                def __len__(self): return len(self.texts)
                def __getitem__(self, idx):
                    enc = tokenizer(
                        str(self.texts[idx]),
                        add_special_tokens=True, max_length=self.max_len,
                        padding="max_length", truncation=True, return_tensors="pt"
                    )
                    return {
                        "input_ids": enc["input_ids"].squeeze(0),
                        "attention_mask": enc["attention_mask"].squeeze(0),
                        "labels": torch.tensor(self.labels[idx], dtype=torch.long),
                    }
            ds = _TorchDS(texts, y, tokenizer)
            args = TrainingArguments(
                output_dir=paths["dir"],
                num_train_epochs=1,
                per_device_train_batch_size=16,
                save_total_limit=2,
                logging_dir=os.path.join(paths["dir"], "logs"),
                learning_rate=2e-5,
                evaluation_strategy="no",
            )
            trainer = Trainer(model=model, args=args, train_dataset=ds)
            trainer.train()
            model.save_pretrained(paths["dir"])
            tokenizer.save_pretrained(paths["dir"])
            return {"status": "ok", "model_dir": paths["dir"], "updated": True}

        X = self._vectorize(texts, amounts)
        if self.classifier:
            if isinstance(self.classifier, SGDClassifier):
                self.classifier.partial_fit(X, y, classes=np.arange(num_classes))
                joblib.dump(self.classifier, paths["model"])
            elif isinstance(self.classifier, XGBClassifier):
                booster = self.classifier.get_booster()
                self.classifier.set_params(num_class=num_classes)
                self.classifier.fit(X, y, xgb_model=booster)
                self.classifier.save_model(paths["model"])
            elif isinstance(self.classifier, VotingClassifier):
                lr = self.classifier.named_estimators_["lr"]
                xgb = self.classifier.named_estimators_["xgb"]
                lr.partial_fit(X, y, classes=np.arange(num_classes))
                booster = xgb.get_booster()
                xgb.set_params(num_class=num_classes)
                xgb.fit(X, y, xgb_model=booster)
                joblib.dump(self.classifier, paths["model"])
        elif model_type == "xgb":
            if not _HAS_XGB:
                raise RuntimeError("xgboost not available.")
            model = XGBClassifier()
            model.load_model(paths["model"])
            booster = model.get_booster()
            model.set_params(num_class=num_classes)
            model.fit(X, y, xgb_model=booster)
            model.save_model(paths["model"])
        elif model_type == "lr":
            model: SGDClassifier = joblib.load(paths["model"])
            model.partial_fit(X, y, classes=np.arange(num_classes))
            joblib.dump(model, paths["model"])
        elif model_type == "hybrid":
            if not _HAS_XGB:
                raise RuntimeError("Hybrid needs xgboost.")
            model: VotingClassifier = joblib.load(paths["model"])
            lr: SGDClassifier = model.named_estimators_["lr"]
            xgb: XGBClassifier = model.named_estimators_["xgb"]
            lr.partial_fit(X, y, classes=np.arange(num_classes))
            booster = xgb.get_booster()
            xgb.set_params(num_class=num_classes)
            xgb.fit(X, y, xgb_model=booster)
            joblib.dump(model, paths["model"])
        else:
            raise ValueError("Unsupported model_type for update.")

        return {"status": "ok", "model_path": paths["model"], "updated": True}

    def _keyword_lookup(self, desc):
        desc_lower = _clean_text(desc)
        for keyword, category in self.keyword_map.items():
            if re.search(keyword, desc_lower):
                return category
        return None

# ---------------------------
# CLI (optional)
# ---------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Advanced Transaction Categorizer")
    parser.add_argument("--mode", choices=["train", "predict", "update"], required=True)
    parser.add_argument("--infile", help="Path to labeled/unlabeled CSV/XLSX for train/predict/update")
    parser.add_argument("--model_type", default="hybrid", choices=["lr", "xgb", "hybrid", "bert"])
    parser.add_argument("--model_dir", default=None, help="Directory to load/save models")
    parser.add_argument("--outfile", default=None, help="Where to save predictions (CSV). If omitted, prints head().")
    args = parser.parse_args()

    tc = TransactionCategorizer()

    if args.mode == "train":
        if not args.infile:
            raise ValueError("--infile is required for training")
        metrics = tc.train(args.infile, model_type=args.model_type, model_dir=args.model_dir)
        print(json.dumps(metrics, indent=2))

    elif args.mode == "predict":
        if not args.infile:
            raise ValueError("--infile is required for prediction")
        if args.infile.lower().endswith(".csv"):
            df_in = pd.read_csv(args.infile)
        else:
            df_in = pd.read_excel(args.infile)
        out = tc.predict(df_in, model_type=args.model_type, model_dir=args.model_dir)
        if args.outfile:
            out.to_csv(args.outfile, index=False)
            print(f"[OK] Saved predictions to {args.outfile}")
        else:
            print(out.head())

    elif args.mode == "update":
        if not args.infile:
            raise ValueError("--infile is required for update")
        result = tc.update(args.infile, model_type=args.model_type, model_dir=args.model_dir)
        print(json.dumps(result, indent=2))