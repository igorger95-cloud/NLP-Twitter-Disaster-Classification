import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)


# ==========================================
# Project root
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))


# ==========================================
# Configuration
# ==========================================

MODEL_NAME = "distilbert-base-uncased"

MAX_LENGTH = 128
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
EPOCHS = 3
WEIGHT_DECAY = 0.01


# ==========================================
# 1. Load data
# ==========================================

train = pd.read_csv(
    PROJECT_ROOT / "data" / "train.csv"
)


# ==========================================
# 2. Minimal text preprocessing
# ==========================================

train["text_clean"] = (
    train["text"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# ==========================================
# 3. Train / validation split
# ==========================================

train_df, val_df = train_test_split(
    train[["text_clean", "target"]],
    test_size=0.2,
    random_state=42,
    stratify=train["target"]
)


# ==========================================
# 4. Convert to HuggingFace Dataset
# ==========================================

train_dataset = Dataset.from_pandas(
    train_df,
    preserve_index=False
)

val_dataset = Dataset.from_pandas(
    val_df,
    preserve_index=False
)


train_dataset = train_dataset.rename_column(
    "target",
    "labels"
)

val_dataset = val_dataset.rename_column(
    "target",
    "labels"
)


# ==========================================
# 5. Load tokenizer
# ==========================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ==========================================
# 6. Tokenization
# ==========================================

def tokenize_function(examples):

    return tokenizer(
        examples["text_clean"],
        truncation=True,
        max_length=MAX_LENGTH
    )


train_dataset = train_dataset.map(
    tokenize_function,
    batched=True
)

val_dataset = val_dataset.map(
    tokenize_function,
    batched=True
)


# Remove original text column
train_dataset = train_dataset.remove_columns(
    ["text_clean"]
)

val_dataset = val_dataset.remove_columns(
    ["text_clean"]
)


# ==========================================
# 7. Data collator
# ==========================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# ==========================================
# 8. Load model
# ==========================================

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)


# ==========================================
# 9. Metrics
# ==========================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1
    )

    return {
        "accuracy": accuracy_score(
            labels,
            predictions
        ),

        "precision": precision_score(
            labels,
            predictions
        ),

        "recall": recall_score(
            labels,
            predictions
        ),

        "f1": f1_score(
            labels,
            predictions
        )
    }


# ==========================================
# 10. Training arguments
# ==========================================

training_args = TrainingArguments(

    output_dir=str(
        PROJECT_ROOT / "models" / "distilbert"
    ),

    learning_rate=LEARNING_RATE,

    per_device_train_batch_size=BATCH_SIZE,

    per_device_eval_batch_size=BATCH_SIZE,

    num_train_epochs=EPOCHS,

    weight_decay=WEIGHT_DECAY,

    eval_strategy="epoch",

    save_strategy="epoch",

    load_best_model_at_end=True,

    metric_for_best_model="f1",

    greater_is_better=True,

    fp16=True,

    logging_dir=str(
        PROJECT_ROOT / "models" / "distilbert" / "logs"
    ),

    report_to="none"
)


# ==========================================
# 11. Trainer
# ==========================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset,

    processing_class=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics
)


# ==========================================
# 12. Fine-tuning
# ==========================================

trainer.train()


# ==========================================
# 13. Evaluation
# ==========================================

evaluation_results = trainer.evaluate()

print("\nEvaluation results:")

for metric, value in evaluation_results.items():

    if isinstance(value, float):

        print(
            f"{metric}: {value:.4f}"
        )


# ==========================================
# 14. Save model
# ==========================================

model_path = (
    PROJECT_ROOT
    / "models"
    / "distilbert"
)

trainer.save_model(
    model_path
)

tokenizer.save_pretrained(
    model_path
)

print(
    f"\nModel saved to: {model_path}"
)
