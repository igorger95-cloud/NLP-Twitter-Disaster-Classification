from pathlib import Path

import numpy as np
import pandas as pd

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    DataCollatorWithPadding
)


# ==========================================
# Project root
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ==========================================
# Paths
# ==========================================

MODEL_PATH = PROJECT_ROOT / "models" / "distilbert"

TEST_PATH = PROJECT_ROOT / "data" / "test.csv"

SAMPLE_SUBMISSION_PATH = (
    PROJECT_ROOT / "data" / "sample_submission.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "results" / "submission.csv"
)


# ==========================================
# 1. Load test data
# ==========================================

test = pd.read_csv(TEST_PATH)

print("Test shape:", test.shape)


# ==========================================
# 2. Minimal text preprocessing
# ==========================================

test["text_clean"] = (
    test["text"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# ==========================================
# 3. Load tokenizer
# ==========================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)

# ==========================================
# 4. Convert test data to Dataset
# ==========================================

test_dataset = Dataset.from_pandas(
    test[["text_clean"]],
    preserve_index=False
)


# ==========================================
# 5. Tokenization
# ==========================================

def tokenize_function(examples):

    return tokenizer(
        examples["text_clean"],
        truncation=True,
        max_length=128
    )


test_dataset = test_dataset.map(
    tokenize_function,
    batched=True
)


# Remove original text
test_dataset = test_dataset.remove_columns(
    ["text_clean"]
)


# ==========================================
# 6. Load trained model
# ==========================================

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)


# ==========================================
# 7. Create Trainer for prediction
# ==========================================

trainer = Trainer(
    model=model,
    data_collator=data_collator
)
# ==========================================
# 8. Make predictions
# ==========================================

predictions = trainer.predict(
    test_dataset
)


predicted_labels = np.argmax(
    predictions.predictions,
    axis=-1
)


# ==========================================
# 9. Create submission
# ==========================================

sample_submission = pd.read_csv(
    SAMPLE_SUBMISSION_PATH
)


submission = pd.DataFrame({
    "id": sample_submission["id"],
    "target": predicted_labels
})


# ==========================================
# 10. Save submission
# ==========================================

submission.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\nSubmission created:")
print(submission.head())

print(
    f"\nSaved to: {OUTPUT_PATH}"
)
