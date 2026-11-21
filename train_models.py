"""Train leakage-safe intent models and create evaluation artifacts."""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, top_k_accuracy_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
REQUIRED = {"text", "category"}

def clean_text(value: object) -> str:
    text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_split(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}. See README.md for setup.")
    frame = pd.read_csv(path)
    if not REQUIRED.issubset(frame.columns):
        raise ValueError(f"{path.name} must contain columns {sorted(REQUIRED)}; found {list(frame.columns)}")
    frame = frame[["text", "category"]].copy()
    frame["text"] = frame["text"].map(clean_text)
    frame["category"] = frame["category"].astype("string").str.strip()
    frame = frame.dropna(subset=["text", "category"])
    frame = frame[(frame.text.str.len() > 0) & (frame.category.str.len() > 0)]
    return frame.reset_index(drop=True)

def source_quality(path: Path) -> dict[str, int]:
    """Measure source problems before cleanup so the audit does not hide removals."""
    raw = pd.read_csv(path)
    missing = REQUIRED - set(raw.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")
    text = raw["text"].fillna("").astype(str).str.strip()
    category = raw["category"].fillna("").astype(str).str.strip()
    return {"source_rows": len(raw), "missing_text": int(raw["text"].isna().sum()),
            "blank_text": int(text.eq("").sum()), "missing_category": int(raw["category"].isna().sum()),
            "blank_category": int(category.eq("").sum())}

def make_pipeline(model: str) -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True,
                                 strip_accents="unicode", max_features=250_000)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1,
                                 sublinear_tf=True, max_features=300_000)),
    ])
    estimator = (LogisticRegression(max_iter=1200, C=4, solver="lbfgs") if model == "logistic"
                 else LinearSVC(C=1.5))
    return Pipeline([("features", features), ("classifier", estimator)])

def get_probabilities(model: Pipeline, X: pd.Series) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    scores = model.decision_function(X)
    # Softmax-like normalized scores support ranking and confidence routing.
    scores = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(np.clip(scores, -50, 50))
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)

def metric_row(name: str, y_true: pd.Series, y_pred: np.ndarray, probs: np.ndarray, labels: np.ndarray) -> dict:
    return {"model": name, "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "top3_accuracy": top_k_accuracy_score(y_true, probs, k=min(3, len(labels)), labels=labels)}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=ROOT / "train.csv")
    parser.add_argument("--test", type=Path, default=ROOT / "test.csv")
    parser.add_argument("--deep-learning", action="store_true", help="Train optional Keras BiLSTM comparison")
    args = parser.parse_args()
    ART.mkdir(parents=True, exist_ok=True)
    train = load_split(args.train)
    test = load_split(args.test)
    train_quality, test_quality = source_quality(args.train), source_quality(args.test)
    overlap = set(train.text) & set(test.text)
    labels = np.array(sorted(set(train.category)))
    unseen_test_labels = sorted(set(test.category) - set(train.category))
    audit = {"train_rows_after_cleanup": len(train), "test_rows_after_cleanup": len(test),
             "train_source_quality": train_quality, "test_source_quality": test_quality,
             "train_rows_dropped": train_quality["source_rows"] - len(train),
             "test_rows_dropped": test_quality["source_rows"] - len(test),
             "train_categories": int(train.category.nunique()), "test_categories": int(test.category.nunique()),
             "duplicate_train_text_rows": int(train.text.duplicated().sum()), "train_test_exact_text_overlap": len(overlap),
             "train_labels_absent_from_test": sorted(set(train.category) - set(test.category)),
             "test_labels_absent_from_train": unseen_test_labels,
             "train_class_counts": train.category.value_counts().sort_index().to_dict(),
             "test_class_counts": test.category.value_counts().sort_index().to_dict()}
    (ART / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if unseen_test_labels:
        raise ValueError(f"The held-out test split contains labels absent from training: {unseen_test_labels}")
    X_train, X_val, y_train, y_val = train_test_split(train.text, train.category, test_size=.15,
        random_state=42, stratify=train.category)
    candidates: dict[str, Pipeline] = {}
    validation_rows = []
    for name in ("logistic", "linear_svc"):
        pipeline = make_pipeline(name).fit(X_train, y_train)
        if name == "linear_svc":
            model = CalibratedClassifierCV(pipeline, method="sigmoid", cv=3).fit(X_train, y_train)
        else:
            model = pipeline
        pred = model.predict(X_val)
        probs = get_probabilities(model, X_val)
        validation_rows.append(metric_row(name, y_val, pred, probs, labels))
        candidates[name] = model
    pd.DataFrame(validation_rows).sort_values("macro_f1", ascending=False).to_csv(ART / "validation_metrics.csv", index=False)
    winner = max(validation_rows, key=lambda row: row["macro_f1"])["model"]
    # Refit selected estimator on all supplied training rows. The supplied test split stays untouched.
    if winner == "linear_svc":
        chosen = CalibratedClassifierCV(make_pipeline("linear_svc"), method="sigmoid", cv=3).fit(train.text, train.category)
    else:
        chosen = make_pipeline("logistic").fit(train.text, train.category)
    test_probs = get_probabilities(chosen, test.text)
    test_pred = chosen.predict(test.text)
    test_metrics = metric_row(winner, test.category, test_pred, test_probs, labels)
    (ART / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    pd.DataFrame(classification_report(test.category, test_pred, output_dict=True, zero_division=0)).T.to_csv(ART / "classification_report.csv")
    pd.DataFrame({"text": test.text, "actual": test.category, "predicted": test_pred,
                  "confidence": test_probs.max(axis=1)}).to_csv(ART / "test_predictions.csv", index=False)
    joblib.dump(chosen, ART / "intent_model.joblib")
    metadata = {"model": winner, "labels": chosen.classes_.tolist(), "trained_at_utc": datetime.now(timezone.utc).isoformat(),
                "thresholds": {"high": 0.80, "medium": 0.55}, "source": "User-provided Banking77-style text/category CSV split"}
    (ART / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Data audit:\n", json.dumps(audit, indent=2))
    print("Validation comparison:\n", pd.DataFrame(validation_rows).to_string(index=False))
    print("Held-out test metrics:\n", json.dumps(test_metrics, indent=2))
    print(f"Saved project artifacts to {ART}")
    if args.deep_learning:
        train_deep_learning(train, test)

def train_deep_learning(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Optional Keras BiLSTM; kept separate so baseline setup stays lightweight."""
    try:
        import tensorflow as tf
        from sklearn.preprocessing import LabelEncoder
    except ImportError as exc:
        raise SystemExit("Optional deep learning requires: pip install tensorflow") from exc
    tf.keras.utils.set_random_seed(42)
    encoder = LabelEncoder().fit(train.category)
    max_tokens, sequence_len = 30_000, 60
    vectorizer = tf.keras.layers.TextVectorization(max_tokens=max_tokens, output_mode="int",
        output_sequence_length=sequence_len, standardize="lower_and_strip_punctuation")
    vectorizer.adapt(train.text.to_numpy())
    inputs = tf.keras.Input(shape=(), dtype=tf.string)
    x = vectorizer(inputs)
    x = tf.keras.layers.Embedding(max_tokens, 96, mask_zero=True)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, dropout=.2))(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(.3)(x)
    outputs = tf.keras.layers.Dense(len(encoder.classes_), activation="softmax")(x)
    net = tf.keras.Model(inputs, outputs)
    net.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    X_a, X_v, y_a, y_v = train_test_split(train.text.to_numpy(), encoder.transform(train.category),
        test_size=.15, random_state=42, stratify=train.category)
    net.fit(X_a, y_a, validation_data=(X_v, y_v), epochs=12, batch_size=64,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)], verbose=2)
    probs = net.predict(test.text.to_numpy(), verbose=0)
    pred = encoder.inverse_transform(probs.argmax(axis=1))
    scores = metric_row("bilstm", test.category, pred, probs, encoder.classes_)
    (ART / "deep_learning_test_metrics.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
    net.save(ART / "intent_bilstm.keras")
    (ART / "deep_learning_labels.json").write_text(json.dumps(encoder.classes_.tolist()), encoding="utf-8")
    print("BiLSTM held-out test metrics:\n", json.dumps(scores, indent=2))

if __name__ == "__main__":
    main()
