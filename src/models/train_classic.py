import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.data.load_data import load_data
from src.data.preprocessing import clean_text_classic
from src.features.tfidf_features import create_tfidf_features
from src.evaluation.metrics import calculate_metrics


# =========================
# 1. Load data
# =========================

train, test = load_data(
    "data/train.csv",
    "data/test.csv"
)


# =========================
# 2. Prepare text
# =========================

train["text_clean"] = train["text"].fillna("").apply(
    lambda x: clean_text_classic(
        x,
        remove_stopwords=False,
        stemming=True
    )
)

test["text_clean"] = test["text"].fillna("").apply(
    lambda x: clean_text_classic(
        x,
        remove_stopwords=False,
        stemming=True
    )
)


X_text = train["text_clean"]
y = train["target"]

X_test_text = test["text_clean"]


# =========================
# 3. Train/validation split
# =========================

from sklearn.model_selection import train_test_split

X_train_text, X_val_text, y_train, y_val = train_test_split(
    X_text,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 4. TF-IDF
# =========================

from src.utils.config import load_config

CONFIG_PATH = PROJECT_ROOT / "configs" / "classic_config.yaml"

config = load_config(CONFIG_PATH)

X_train, X_val, vectorizer = create_tfidf_features(
    X_train_text,
    X_val_text,
    max_features=config["tfidf"]["max_features"],
    ngram_range=tuple(config["tfidf"]["ngram_range"])
)


# =========================
# 5. Logistic Regression
# =========================

logistic_model = LogisticRegression(
    C=config["logistic_regression"]["C"],
    max_iter=config["logistic_regression"]["max_iter"],
    random_state=42
)

logistic_model.fit(X_train, y_train)

logistic_predictions = logistic_model.predict(X_val)

logistic_metrics = calculate_metrics(
    y_val,
    logistic_predictions
)

print("TF-IDF + Logistic Regression")
print(logistic_metrics)


# =========================
# 6. Gradient Boosting
# =========================

gradient_model = GradientBoostingClassifier(
    n_estimators=config["gradient_boosting"]["n_estimators"],
    random_state=42
)

gradient_model.fit(
    X_train.toarray(),
    y_train
)

gradient_predictions = gradient_model.predict(
    X_val.toarray()
)

gradient_metrics = calculate_metrics(
    y_val,
    gradient_predictions
)

print("\nTF-IDF + Gradient Boosting")
print(gradient_metrics)


# =========================
# 7. Save results
# =========================

results = pd.DataFrame([
    {
        "Approach": "Classic NLP",
        "Vectorization": "TF-IDF",
        "Model": "Logistic Regression",
        **logistic_metrics
    },
    {
        "Approach": "Classic NLP",
        "Vectorization": "TF-IDF",
        "Model": "Gradient Boosting",
        **gradient_metrics
    }
])

results_path = PROJECT_ROOT / "results" / "classic_metrics.csv"

results.to_csv(
    results_path,
    index=False
)

print(f"\nResults saved to: {results_path}")
