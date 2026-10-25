"""
Toxic Comment Detection: TF-IDF + Logistic Regression (VS Code ready)

Usage (run from the integrated terminal):
    python toxic_detector.py                    # train on small_toxic_dataset.csv
    python toxic_detector.py --data train.csv   # Kaggle Jigsaw train.csv
    python toxic_detector.py --hf               # Civil Comments from Hugging Face
    python toxic_detector.py --predict "text"   # score one comment (needs trained model)
    python toxic_detector.py --predict          # interactive mode

Outputs (saved in the 'outputs' folder next to this file):
    model.joblib, report.txt, confusion_matrix.png, label_counts.png
"""

import argparse
import re
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # save figures to files; no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs"
DEFAULT_DATA = BASE_DIR / "small_toxic_dataset.csv"
MODEL_PATH = OUT_DIR / "model.joblib"

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# Column names used by different datasets -> our standard names
TEXT_COLUMNS = ["comment_text", "text", "comment"]
LABEL_ALIASES = {
    "toxic": ["toxic", "toxicity"],
    "severe_toxic": ["severe_toxic", "severe_toxicity"],
    "obscene": ["obscene"],
    "threat": ["threat"],
    "insult": ["insult"],
    "identity_hate": ["identity_hate", "identity_attack"],
}


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def standardise(df):
    """Rename columns to the standard names and convert labels to 0/1."""
    text_col = next((c for c in TEXT_COLUMNS if c in df.columns), None)
    if text_col is None:
        sys.exit(f"No text column found. Expected one of {TEXT_COLUMNS}. Found: {list(df.columns)}")
    out = pd.DataFrame({"text": df[text_col].astype(str)})
    for label, aliases in LABEL_ALIASES.items():
        col = next((a for a in aliases if a in df.columns), None)
        if col is None:
            sys.exit(f"Label column for '{label}' not found. Expected one of {aliases}.")
        out[label] = (pd.to_numeric(df[col], errors="coerce").fillna(0) >= 0.5).astype(int)
    return out


def load_csv(path):
    path = Path(path)
    if not path.exists():
        sys.exit(f"File not found: {path}\nPut the CSV in {BASE_DIR} or pass the full path with --data.")
    return standardise(pd.read_csv(path))


def load_hf(train_size=100000):
    try:
        from datasets import load_dataset
        ds = load_dataset("google/civil_comments", split="train")
    except Exception as e:  # network, name change, library missing
        sys.exit(f"Could not load Civil Comments from Hugging Face: {e}\n"
                 "Use --data with a CSV file instead.")
    ds = ds.shuffle(seed=42).select(range(min(train_size, len(ds))))
    return standardise(ds.to_pandas())


def train_and_evaluate(df, test_size=0.25):
    df = df.copy()
    df["clean"] = df["text"].apply(clean_text)

    stratify = df["toxic"] if df["toxic"].value_counts().min() >= 2 else None
    train, test = train_test_split(df, test_size=test_size, random_state=42, stratify=stratify)
    print(f"Train: {len(train)} | Test: {len(test)}")

    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2),
                            stop_words="english", sublinear_tf=True)
    X_train = tfidf.fit_transform(train["clean"])
    X_test = tfidf.transform(test["clean"])

    model = OneVsRestClassifier(
        LogisticRegression(C=4, max_iter=1000, class_weight="balanced", solver="liblinear")
    )
    model.fit(X_train, train[LABELS])

    prob = model.predict_proba(X_test)
    pred = (prob >= 0.5).astype(int)
    y_true = test[LABELS].values

    # per-label ROC-AUC (skipped when the test set has only one class for a label)
    aucs = {}
    for i, label in enumerate(LABELS):
        aucs[label] = roc_auc_score(y_true[:, i], prob[:, i]) if len(np.unique(y_true[:, i])) == 2 else float("nan")
    macro_auc = np.nanmean(list(aucs.values()))
    macro_f1 = f1_score(y_true, pred, average="macro", zero_division=0)

    report = classification_report(y_true, pred, target_names=LABELS, zero_division=0)
    summary = (f"{report}\nMacro F1: {macro_f1:.3f}\nMacro ROC-AUC: {macro_auc:.3f}\n"
               "Per-label ROC-AUC: " + ", ".join(f"{k}={v:.2f}" for k, v in aucs.items()))
    print("\n" + summary)

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "report.txt").write_text(summary, encoding="utf-8")

    # charts (saved as images)
    plt.figure(figsize=(6, 3.4))
    df[LABELS].sum().plot(kind="bar", color="#2E75B6")
    plt.title("Label counts"); plt.ylabel("Comments"); plt.xticks(rotation=25, ha="right")
    plt.tight_layout(); plt.savefig(OUT_DIR / "label_counts.png", dpi=150); plt.close()

    cm = confusion_matrix(test["toxic"], pred[:, 0], labels=[0, 1])
    plt.figure(figsize=(4, 3.4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Not toxic", "Toxic"], yticklabels=["Not toxic", "Toxic"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion matrix: toxic")
    plt.tight_layout(); plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=150); plt.close()

    # refit on all data for the saved model
    X_all = tfidf.fit_transform(df["clean"])
    model.fit(X_all, df[LABELS])
    joblib.dump({"tfidf": tfidf, "model": model, "labels": LABELS}, MODEL_PATH)
    print(f"\nSaved model and results in: {OUT_DIR}")


def predict(text, bundle, threshold=0.5):
    probs = bundle["model"].predict_proba(bundle["tfidf"].transform([clean_text(text)]))[0]
    scores = dict(zip(bundle["labels"], probs))
    flagged = [k for k, v in scores.items() if v >= threshold]
    return scores, flagged


def run_predict(text):
    if not MODEL_PATH.exists():
        sys.exit("No trained model found. Run 'python toxic_detector.py' first.")
    bundle = joblib.load(MODEL_PATH)

    def show(t):
        scores, flagged = predict(t, bundle)
        print(f"\nComment: {t}")
        for k, v in scores.items():
            print(f"  {k:<14} {v:.2f} {'<-- flagged' if v >= 0.5 else ''}")
        print("Result:", ", ".join(flagged) if flagged else "clean")

    if text:
        show(text)
    else:
        print("Type a comment (or 'q' to quit).")
        while True:
            t = input("\n> ").strip()
            if t.lower() in {"q", "quit", "exit"}:
                break
            if t:
                show(t)


def main():
    ap = argparse.ArgumentParser(description="Toxic comment detection")
    ap.add_argument("--data", help="path to a CSV file (Jigsaw, Civil Comments or sample format)")
    ap.add_argument("--hf", action="store_true", help="load Civil Comments from Hugging Face")
    ap.add_argument("--predict", nargs="?", const="", metavar="TEXT",
                    help="score a comment with the saved model (no text = interactive)")
    args = ap.parse_args()

    if args.predict is not None:
        run_predict(args.predict)
        return

    df = load_hf() if args.hf else load_csv(args.data or DEFAULT_DATA)
    print(f"Loaded {len(df)} comments")
    print(df[LABELS].sum().to_string())
    train_and_evaluate(df)


if __name__ == "__main__":
    main()
