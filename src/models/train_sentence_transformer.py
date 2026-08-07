import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sentence_transformers import SentenceTransformer

from src.evaluation.metrics import calculate_metrics


# ==========================================
# 1. Load data
# ==========================================

train = pd.read_csv(
    PROJECT_ROOT / "data" / "train.csv"
)


# ==========================================
# 2. Prepare text
# ==========================================

texts = (
    train["text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

y = train["target"]


# ==========================================
# 3. Train / validation split
# ==========================================

X_train_text, X_val_text, y_train, y_val = train_test_split(
    texts,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 4. Load Sentence Transformer
# ==========================================

MODEL_NAME = "all-MiniLM-L6-v2"

embedding_model = SentenceTransformer(
    MODEL_NAME
)


# ==========================================
# 5. Create embeddings
# ==========================================

X_train_embeddings = embedding_model.encode(
    X_train_text.tolist(),
    show_progress_bar=True
)

X_val_embeddings = embedding_model.encode(
    X_val_text.tolist(),
    show_progress_bar=True
)


# ==========================================
# 6. Logistic Regression
# ==========================================

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(
    X_train_embeddings,
    y_train
)


# ==========================================
# 7. Evaluation
# ==========================================

predictions = model.predict(
    X_val_embeddings
)

metrics = calculate_metrics(
    y_val,
    predictions
)

print("\nSentence Transformer + Logistic Regression")
print(f"Model: {MODEL_NAME}")
print(metrics)
