import os

import numpy as np
from datasets import load_dataset, load_metric, set_caching_enabled
from decouple import config
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

# Logging
from transformers.integrations import NeptuneCallback
from transformers.trainer_utils import set_seed

BATCH_SIZE = config("BATCH_SIZE", cast=int, default=1)
MODEL_DIR = config("MODEL_DIR", default="/workspace/model")
DATA_DIR = config("DATA_DIR", default="/workspace/data")
CACHE_DIR = config("CACHE_DIR", default="./cache/huggingface")
LOG_DIR = config("LOG_DIR", default="/workspace/logs")

# Neptune API (loaded as global env. variables from the Training Callback)
NEPTUNE_API_TOKEN = config("NEPTUNE_API_TOKEN")
NEPTUNE_PROJECT = config("NEPTUNE_PROJECT")

# Huggingface transformers specific
MODEL_NAME = "bert-base-uncased"
METRIC_NAME = "accuracy"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def tokenize_fn(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


if __name__ == "__main__":
    # Define model and data
    set_caching_enabled(True)
    set_seed(42)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2, cache_dir=CACHE_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    dataset_raw = load_dataset("imdb", cache_dir=CACHE_DIR)
    metric = load_metric(METRIC_NAME)

    # Prepate the data
    dataset = dataset_raw.map(tokenize_fn, batched=True)
    data_collator = default_data_collator

    # Training on smaller dataset first is a good first check
    small_train_dataset = dataset["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = dataset["test"].shuffle(seed=42).select(range(1000))

    train_dataset = dataset["train"]
    test_dataset, eval_dataset = dataset["test"].train_test_split(shuffle=True, seed=42, test_size=0.4)

    # Train and evaluate the model
    training_args = TrainingArguments(
        f"{MODEL_DIR}/{MODEL_NAME}-finetuned",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=2e-5,
        num_train_epochs=1,
        logging_dir=LOG_DIR,
        warmup_steps=500,
        weight_decay=0.01,
        metric_for_best_model=METRIC_NAME,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        callbacks=[
            NeptuneCallback,
            # TensorBoardCallback(SummaryWriter(log_dir=LOG_DIR)),
        ],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    result_train = trainer.train()
    result_eval = trainer.evaluate()
