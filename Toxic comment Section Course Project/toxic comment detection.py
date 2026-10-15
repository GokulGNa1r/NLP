"""
Toxic Comment Detection (NLP)
TF-IDF + Logistic Regression baseline on the Civil Comments dataset.

Install first:
    pip install datasets pandas numpy scikit-learn matplotlib seaborn joblib

Run:
    python toxic_comment_detection.py
"""

import re

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.multiclass import OneVsRestClassifier

LABELS = ["toxicity", "severe_toxicity", "obscene",
          "threat", "insult", "identity_attack"]


# ---------- Phase 1: Load data ----------
def load_data(train_size=100000, test_size=20000):
    ds = load_dataset("google/civil_comments")
    train = ds["train"].shuffle(seed=42).select(range(train_size)).to_pandas()
    test = ds["test"].shuffle(seed=42).select(range(test_size)).to_pandas()

    # scores are 0 to 1, so convert them to 0/1
    for label in LABELS:
        train[label] = (train[label] >= 0.5).astype(int)
        test[label] = (test[label] >= 0.5).astype(int)

    print("Train:", train.shape, "Test:", test.shape)
    return train, test


# ---------- Phase 2: EDA ----------
def run_eda(train):
    counts = train[LABELS].sum().sort_values(ascending=False)
    print("\nLabel counts:\n", counts)
    counts.plot(kind="bar", title="Label counts", figsize=(7, 4))
    plt.tight_layout()
    plt.show()


# ---------- Phase 3: Clean text ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------- Phase 4: Features and model ----------
def train_model(train):
    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2),
                            stop_words="english", sublinear_tf=True)
    X_train = tfidf.fit_transform(train["clean"])

    model = OneVsRestClassifier(
        LogisticRegression(C=4, max_iter=1000,
                           class_weight="balanced", solver="liblinear")
    )
    model.fit(X_train, train[LABELS])
    return tfidf, model


# ---------- Phase 5: Evaluate ----------
def evaluate(model, tfidf, test):
    X_test = tfidf.transform(test["clean"])
    y_test = test[LABELS]

    y_prob = model.predict_proba(X_test)
    y_pred = (y_prob >= 0.5).astype(int)

    print("\n" + classification_report(y_test, y_pred,
                                       target_names=LABELS, zero_division=0))
    print("Macro ROC-AUC:", round(roc_auc_score(y_test, y_prob, average="macro"), 4))
    print("Macro F1     :", round(f1_score(y_test, y_pred, average="macro"), 4))

    cm = confusion_matrix(y_test["toxicity"], y_pred[:, 0])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not toxic", "Toxic"],
                yticklabels=["Not toxic", "Toxic"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix: toxicity")
    plt.show()


# ---------- Phase 6: Predict ----------
def predict_comment(text, model, tfidf, threshold=0.5):
    probs = model.predict_proba(tfidf.transform([clean_text(text)]))[0]
    result = {l: round(float(p), 3) for l, p in zip(LABELS, probs)}
    flagged = [l for l, p in result.items() if p >= threshold]
    return result, (flagged if flagged else ["clean"])


def main():
    train, test = load_data()
    run_eda(train)

    train["clean"] = train["text"].apply(clean_text)
    test["clean"] = test["text"].apply(clean_text)

    tfidf, model = train_model(train)
    evaluate(model, tfidf, test)

    print("\n--- Sample predictions ---")
    samples = [
        "Thanks, this was really helpful!",
        "You are an idiot and nobody wants you here.",
        "I will find you and hurt you.",
    ]
    for s in samples:
        scores, flags = predict_comment(s, model, tfidf)
        print(f"\n{s}\n -> {flags}\n {scores}")

    joblib.dump(model, "toxic_model.pkl")
    joblib.dump(tfidf, "tfidf.pkl")
    print("\nSaved toxic_model.pkl and tfidf.pkl")


if __name__ == "__main__":
    main()
