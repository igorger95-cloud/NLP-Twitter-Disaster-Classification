import sys
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.data.load_data import load_data
from src.data.preprocessing import clean_text_classic
from src.features.tfidf_features import create_tfidf_features
from src.features.word2vec_features import (
    train_word2vec,
    texts_to_vectors
)
from src.evaluation.metrics import calculate_metrics
from src.utils.config import load_config


# ==========================================
# 1. Load configuration
# ==========================================

CONFIG_PATH = PROJECT_ROOT / "configs" / "classic_config.yaml"
config = load_config(CONFIG_PATH)


# ==========================================
# 2. Load data
# ==========================================

train, test = load_data(
    PROJECT_ROOT / "data" / "train.csv",
    PROJECT_ROOT / "data" / "test.csv"
)


# ==========================================
# 3. Prepare text
# ==========================================

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


# ==========================================
# 4. Train / validation split
# ==========================================

X_train_text, X_val_text, y_train, y_val = train_test_split(
    X_text,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 5. TF-IDF + Logistic Regression
# ==========================================

X_train_tfidf, X_val_tfidf, _ = create_tfidf_features(
    X_train_text,
    X_val_text,
    max_features=config["tfidf"]["max_features"],
    ngram_range=tuple(config["tfidf"]["ngram_range"])
)

logistic_model = LogisticRegression(
    C=config["logistic_regression"]["C"],
    max_iter=config["logistic_regression"]["max_iter"],
    random_state=42
)

logistic_model.fit(
    X_train_tfidf,
    y_train
)

logistic_predictions = logistic_model.predict(
    X_val_tfidf
)

logistic_metrics = calculate_metrics(
    y_val,
    logistic_predictions
)

print("\nTF-IDF + Logistic Regression")
print(logistic_metrics)


# ==========================================
# 6. Word2Vec
# ==========================================

word2vec_model = train_word2vec(
    X_train_text,
    vector_size=config["word2vec"]["vector_size"],
    window=config["word2vec"]["window"],
    min_count=config["word2vec"]["min_count"],
    workers=config["word2vec"]["workers"],
    epochs=config["word2vec"]["epochs"]
)

X_train_w2v = texts_to_vectors(
    X_train_text,
    word2vec_model
)

X_val_w2v = texts_to_vectors(
    X_val_text,
    word2vec_model
)


# ==========================================
# 7. Word2Vec + Logistic Regression
# ==========================================

w2v_logistic_model = LogisticRegression(
    C=config["logistic_regression"]["C"],
    max_iter=config["logistic_regression"]["max_iter"],
    random_state=42
)

w2v_logistic_model.fit(
    X_train_w2v,
    y_train
)

w2v_logistic_predictions = w2v_logistic_model.predict(
    X_val_w2v
)

w2v_logistic_metrics = calculate_metrics(
    y_val,
    w2v_logistic_predictions
)

print("\nWord2Vec + Logistic Regression")
print(w2v_logistic_metrics)


# ==========================================
# 8. Word2Vec + Gradient Boosting
# ==========================================

gradient_model = GradientBoostingClassifier(
    n_estimators=config["gradient_boosting"]["n_estimators"],
    random_state=42
)

gradient_model.fit(
    X_train_w2v,
    y_train
)

gradient_predictions = gradient_model.predict(
    X_val_w2v
)

gradient_metrics = calculate_metrics(
    y_val,
    gradient_predictions
)

print("\nWord2Vec + Gradient Boosting")
print(gradient_metrics)


# ==========================================
# 9. Save results
# ==========================================

results = pd.DataFrame([
    {
        "Approach": "Classic NLP",
        "Vectorization": "TF-IDF",
        "Model": "Logistic Regression + Stemming",
        "F1-score": logistic_metrics["f1"]
    },
    {
        "Approach": "Classic NLP",
        "Vectorization": "Word2Vec",
        "Model": "Logistic Regression",
        "F1-score": w2v_logistic_metrics["f1"]
    },
    {
        "Approach": "Classic NLP",
        "Vectorization": "Word2Vec",
        "Model": "Gradient Boosting",
        "F1-score": gradient_metrics["f1"]
    }
])

RESULTS_PATH = PROJECT_ROOT / "results" / "classic_metrics.csv"

results.to_csv(
    RESULTS_PATH,
    index=False
)

print(f"\nResults saved to: {RESULTS_PATH}")
