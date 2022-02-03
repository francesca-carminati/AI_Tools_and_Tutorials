import argparse
import datetime
import json
import logging
import os
from pathlib import Path
from pprint import PrettyPrinter
from typing import Any, Dict, List, Optional

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from datasets import Dataset, DatasetDict, load_dataset
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from torch import FloatTensor, LongTensor, Tensor
from torch.utils.data import DataLoader
from torchmetrics import F1, Accuracy, MetricCollection, Precision, Recall
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
)

logger = logging.getLogger(__name__)
pprint = PrettyPrinter(indent=4, width=56, compact=True).pprint

# Gives error for running in potentiall parallism in PL else
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Store all predictions
# TODO move to own module later
destroyed_indices = set()
prediction_dict = {
    "dataset": "imdb",
    "split": "train",
    "epochs": [],
    "idx": [],
    "max_epochs": 0,  # will change
}


def get_prediction_dict():
    """Get the predicitons list and return a copy of it."""
    global prediction_dict
    return prediction_dict


def set_prediction_dict(updated_predictions):
    """Update the predicitons list with a new list"""
    global prediction_dict
    prediction_dict = updated_predictions


def set_prediction_dict_value(key: str, value: str):
    """Set a key value pair in"""
    global prediction_dict
    prediction_dict[key] = value


def insert_batch_predictions(current_epoch, batch_pred):
    """Insert the predictions for the current batch into the prediction list.
    Find the correct epoch and location in the list to insert the predictions.
    """
    pred_list = get_prediction_dict()

    if current_epoch not in pred_list["idx"]:
        pred_list["idx"].append(current_epoch)
        pred_list["epochs"].append({"idx": current_epoch, "batches": [batch_pred]})
    else:
        for idx, d in enumerate(pred_list["epochs"]):
            if d["idx"] == current_epoch:
                pred_list["epochs"][idx]["batches"].append(batch_pred)

    # Update the predictions list
    set_prediction_dict(pred_list)


def get_destroyed_indices():
    global destroyed_indices
    return destroyed_indices


def set_destroyed_indices(num_destroyed):
    """Set which indices in the dataset have been destroyed."""
    global destroyed_indices
    destroyed_indices = set(range(0, num_destroyed))


def get_matching_destroyed_indices(indices: List[int]) -> List[int]:
    """Check if data points belong to the destroyed labels."""
    destroyed_indices = get_destroyed_indices()
    return [des_idx in destroyed_indices for des_idx in indices]


def tolist(tensor) -> List:
    """Tensor to python list object."""
    return tensor.cpu().detach().tolist()


def get_dataframe_maching_label(df, label: int):
    return df.where(df["label"] == label).dropna().reset_index(drop=True)


def is_not_empty(_list: List[Any]) -> bool:
    return len(_list) > 0


def destroy_labels(dataset: DatasetDict, num_destroyed_labels: int,) -> DatasetDict:
    """Change positve label (1s) to false labels (0s)."""

    tmp_file = "./destroyed_labels.csv"
    dataset.to_csv(f"{tmp_file}", index=False)
    df = pd.read_csv(f"{tmp_file}")

    pos_csv = get_dataframe_maching_label(df, 1)
    neg_csv = get_dataframe_maching_label(df, 0)

    pos_csv.loc[: num_destroyed_labels - 1, "label"] = 0
    set_destroyed_indices(num_destroyed_labels)

    df = pd.concat([pos_csv, neg_csv], ignore_index=True)
    df["label"] = df["label"].astype(int)
    df.to_csv(f"{tmp_file}", index=False)

    # Load from csv creates a dict with the data split
    dataset_split = "train"
    dataset = load_dataset("csv", data_files=tmp_file)
    os.remove(tmp_file)
    return dataset[dataset_split]


class ImdbCorruptDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer_name: str,
        max_sequence_length: int,
        batch_size: int,
        seed: int,
        corrupt_percentage: int = 0,
        corrupt_num_samples: int = 0,
        num_train_samples: int = 10000,
        num_val_samples: int = -1,  # TODO not used (num_train == num_val)
        num_label_split: int = -1,  # TODO not used
        val_is_test: bool = True,  # TODO fix for other dataset
    ):
        super().__init__()
        self.tokenizer_name = tokenizer_name
        self.max_sequence_length = max_sequence_length
        self.batch_size = batch_size
        self.seed = seed
        self.corrupt_percentage = corrupt_percentage
        self.corrupt_num_samples = corrupt_num_samples
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_label_split = num_label_split
        self.val_is_test = val_is_test

        self.data_loader_cache: Dict[str, DataLoader] = dict()
        self.dataset: Dataset

    # (required) Runs first on only one GPU
    def prepare_data(self):
        """Download files to cache first once per node."""
        AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)
        split = self.get_train_test_split()
        load_dataset(path="imdb", split=split)

    def sync_destroyed_num_samples_and_percentages(self) -> None:
        """Depending on which param sent in.
        Sync how many labels destroyed"""

        # N/2 since we are using equally many of label 0 and label 1.
        N = self.num_train_samples // 2

        if self.corrupt_num_samples > 0:
            self.corrupt_percentage = self.corrupt_num_samples // N
        elif self.corrupt_percentage > 0:
            self.corrupt_num_samples = (N // 100) * self.corrupt_percentage

    def get_train_test_split(self):
        """Return train test split for HF datasets."""
        split = {
            "train": "train",
            "test": "test",
        }

        if self.num_train_samples > 0:
            num_samples = self.num_train_samples // 2
            split = {
                "train": f"train[:{num_samples}]+train[-{num_samples}:]",
                "test": f"test[:{num_samples}]+test[-{num_samples}:]",
            }
        return split

    def get_and_process_dataset(self):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)

        # tokenize the dataset
        def preprocess(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
            encoding = tokenizer(
                example["text"],
                max_length=self.max_sequence_length,
                truncation=True,
                padding="max_length",
                return_attention_mask=True,
                return_token_type_ids=True,
            )
            encoding.update({"idx": idx})
            return encoding

        dataset: DatasetDict = DatasetDict()
        split = self.get_train_test_split()
        dataset = load_dataset(path="imdb", split=split)

        self.sync_destroyed_num_samples_and_percentages()
        num_destroyed = self.corrupt_num_samples

        if num_destroyed > 0:
            dataset["train"] = destroy_labels(dataset["train"], num_destroyed)

        logger.info("Tokenizing the data")
        return dataset.map(preprocess, batched=True, with_indices=True)

    # (required)
    def setup(self, stage: Optional[str] = None):
        dataset = self.get_and_process_dataset()
        dataset.set_format(
            type="torch",
            columns=["input_ids", "token_type_ids", "attention_mask", "label", "idx"],  # To find destroyed labels
        )
        self.dataset = dataset.shuffle(seed=self.seed)

    def _create_dataloader(
        self, dataset_partitions: Dataset, split: str, batch_size: int,
    ):
        if split == "validation" and self.val_is_test:
            split = "test"

        dataset = dataset_partitions[split]
        return DataLoader(dataset, batch_size=batch_size, num_workers=4,)

    # TODO use later
    def get_dataset_cache_split(self, split: str):
        self.data_loader_cache[split] = self._create_dataloader(self.dataset, split=split, batch_size=self.batch_size)
        return self.data_loader_cache[split]

    """
    # (required)
    def train_dataloader(self):
        return self.get_dataset_cache_split(self, split="train")

    # (required)
    def val_dataloader(self):
        return self.get_dataset_cache_split(self, split="validation")

    # (required)
    def test_dataloader(self):
        return self.get_dataset_cache_split(self, split="test")
    """

    def train_dataloader(self):
        self.data_loader_cache["train"] = self._create_dataloader(
            self.dataset, split="train", batch_size=self.batch_size
        )
        return self.data_loader_cache["train"]

    def val_dataloader(self):
        self.data_loader_cache["validation"] = self._create_dataloader(
            self.dataset, split="validation", batch_size=self.batch_size
        )
        return self.data_loader_cache["validation"]

    def test_dataloader(self):
        self.data_loader_cache["test"] = self._create_dataloader(self.dataset, split="test", batch_size=self.batch_size)
        return self.data_loader_cache["test"]


class NLLabelSmoothing(nn.Module):
    """NLL loss with label smoothing."""

    def __init__(self, smoothing=0.0):
        super(NLLabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, probs, target):
        logprobs = F.log_softmax(probs, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class LabelSmoothing(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1, weight=None):
        """Regular CrossEntropy with labels smoothing.
        Expects the target labels to be one-hot encoded.
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.weight = weight
        self.cls = classes
        self.dim = dim

    def forward(self, logit, target):
        assert 0 <= self.smoothing < 1
        pred = logit.log_softmax(dim=self.dim)

        if self.weight is not None:
            pred = pred * self.weight.unsqueeze(0)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class BinaryClassificationModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        lr: float,
        smoothing: float,
        batch_size: int,
        weight_decay: float,
        warmup_steps: int,
        num_workers: int,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.lr = lr
        self.smoothing = smoothing
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.num_workers = num_workers

        self.loss = LabelSmoothing(num_labels, smoothing=smoothing)
        self.train_metrics = MetricCollection(
            [
                Accuracy(),
                Precision(num_classes=self.num_labels),
                Recall(num_classes=self.num_labels),
                F1(num_classes=self.num_labels),
            ]
        )
        self.val_metrics = self.train_metrics.clone()
        self.test_metrics = self.train_metrics.clone()

        self.config: PretrainedConfig
        self.model: PreTrainedModel

        # For debugging
        self.iter_counter = 0

    def prepare_data(self):
        """Set logging level and download model on main process."""
        transformers.utils.logging.set_verbosity_warning()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
        AutoModelForSequenceClassification.from_pretrained(self.model_name)

    def setup(self, stage: Optional[str] = None):
        self.config = AutoConfig.from_pretrained(self.model_name, num_labels=self.num_labels,)

        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=self.config)

        set_prediction_dict_value("max_epochs", self.trainer.max_epochs)
        # set_prediction_dict_value(
        #     "num_training_batches_per_epoch",
        #     self.num_training_steps / self.trainer.max_epochs,
        # )

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids, attention_mask=attention_mask, return_dict=True)
        return output

    def log_predictions(
        self, data_idx_t: LongTensor, labels_t: LongTensor, probs_t: FloatTensor, preds_t: FloatTensor, batch_idx: int,
    ) -> None:
        """Capture and store the results from the training.

        For the logits, predicted labels, softmax values etc.,
        store the values to a global dict.

        These are later saved to a .json file for later access.

        Args:
            data_idx_t - Index for all data samples in the batch
            labels_t - The expected correct labels
            probs_t - The softmax output for each batch
            preds_t - The predicted labels
            batch_idx - Index of the batch samples

        Returns:
            Updates global list with all batch predictions, labels etc.
        """

        # Store which predictions the model predicted correct and not
        match = tolist((preds_t == labels_t))
        indices_correct = [i for i, x in enumerate(match) if x]
        indices_incorrect = [i for i, x in enumerate(match) if not x]

        data_idx: List[int] = tolist(data_idx_t)
        probs: List[float] = tolist(probs_t)
        preds: List[float] = tolist(preds_t)
        labels: List[int] = tolist(labels_t)
        is_corrupted = get_matching_destroyed_indices(data_idx)

        def insert_batch_pred(indices_list: List):
            # Inserts predictions for the current batch
            # Based on the index (predicts softmax correct or not)
            batch_pred: Dict[str, Any]
            batch_pred = {
                "idx": batch_idx,
                "data_idx": [],
                "probs": [],
                "preds": [],
                "labels": [],
                "is_corrupted": [],
            }

            while len(indices_list) > 0:
                index = indices_list.pop(0)
                batch_pred["data_idx"].append(data_idx[index])
                batch_pred["probs"].append(probs[index])
                batch_pred["preds"].append(preds[index])
                batch_pred["labels"].append(labels[index])
                batch_pred["is_corrupted"].append(is_corrupted[index])
            return batch_pred

        batch_pred = {
            "correct": insert_batch_pred(indices_correct),
            "incorrect": insert_batch_pred(indices_incorrect),
        }

        insert_batch_predictions(self.current_epoch, batch_pred)

        return None

    def log_each_step(self, prefix: str, metric_name: str, metric: Tensor, show_progress=False,) -> None:
        """Log the metric value used in each step."""
        if show_progress:
            self.log(
                f"{prefix}_{metric_name}", metric, prog_bar=True, on_step=True, logger=True,
            )
        else:
            self.log(f"{prefix}_{metric_name}", metric, on_step=True, logger=True)

    def step(self, batch, batch_idx, stage):
        input_ids = batch["input_ids"]
        labels = batch["label"]
        attention_mask = batch["attention_mask"]
        output = self(input_ids, attention_mask=attention_mask)

        logits = output["logits"]
        loss = self.loss(logits, labels)

        probs = logits.softmax(dim=-1)
        preds = probs.argmax(dim=-1)
        acc, prec, recall, f1 = eval(f"self.{stage}_metrics(preds, labels)").values()

        # TODO make better
        # f'{loss=}'.split('=')[0]
        self.log_each_step(stage, "loss", loss, show_progress=True)
        self.log_each_step(stage, "acc", acc)
        self.log_each_step(stage, "prec", prec)

        if stage == "train":
            self.iter_counter += 1
            self.log_predictions(batch["idx"], labels, probs, preds, batch_idx)
            self.log("learning_rate", self.optim.param_groups[0]["lr"])

        return loss, acc, prec, recall, f1

    def training_step(self, batch, batch_idx):
        loss, acc, prec, recall, f1 = self.step(batch, batch_idx, stage="train")
        return {"loss": loss, "acc": acc}

    def validation_step(self, batch, batch_idx):
        loss, acc, prec, recall, f1 = self.step(batch, batch_idx, stage="val")
        return {"val_loss": loss, "val_acc": acc}

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_val_acc = torch.stack([x["val_acc"] for x in outputs]).mean()

        self.log("avg_val_loss", avg_val_loss, prog_bar=True)
        self.log("avg_val_acc", avg_val_acc, prog_bar=True)
        return {"avg_val_loss": avg_val_loss, "avg_val_acc": avg_val_acc}

    def on_batch_end(self):
        # This is needed to use the One Cycle learning rate.
        # Without this, the learning rate will only change after every epoch
        if self.optim is not None:
            self.optim.step()

        if self.sched is not None:
            self.sched.step()

    def on_epoch_end(self):
        if self.optim is not None:
            self.optim.step()

        if self.sched is not None:
            self.sched.step()

    def test_step(self, batch, batch_idx):
        loss, acc, prec, recall, f1 = self.step(batch, batch_idx, stage="test")
        return {"test_loss": loss, "test_acc": acc}

    def test_epoch_end(self, outputs):
        avg_test_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        avg_test_acc = torch.stack([x["test_acc"] for x in outputs]).mean()

        tensorboard_logs = {
            "avg_test_loss": avg_test_loss,
            "avg_test_acc": avg_test_acc,
        }
        return {
            "avg_test_acc": avg_test_acc,
            "log": tensorboard_logs,
            "progress_bar": tensorboard_logs,
        }

    def configure_optimizers(self):
        """Return optimizers and schedulers."""
        no_decay = ["bias", "LayerNorm.weight"]
        optim_params = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(optim_params, lr=self.lr, eps=1e-8,)
        # scheduler = optimization.get_linear_schedule_with_warmup(
        #    optimizer,
        #    num_warmup_steps=0,
        #    num_training_steps=total_steps
        # )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-5, total_steps=2000)

        self.sched = scheduler
        self.optim = optimizer
        return [optimizer], [scheduler]

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs


def get_logging_callback(_config):
    loggers = []
    logger_choice = _config["loggers"]

    project_name = _config["experiment_name"]
    corruption = f"p={_config['corrupt_percentage']}"
    label_smoothing = f"smoothing={_config['label_smoothing']}"

    if logger_choice in ["wandb", "all"]:
        wandb_logger = pl.loggers.WandbLogger(project=f"{project_name}", name=f"{corruption}-{label_smoothing}",)
        loggers.append(wandb_logger)
    if logger_choice in ["tb", "tensorboard", "all"]:
        tb_logger = pl_loggers.TensorBoardLogger("logs/")
        loggers.append(tb_logger)
    if logger_choice in ["none", "None", None]:
        loggers = None

    return loggers


def start_training(_config):
    global SEED
    SEED = _config["seed"]
    pl.seed_everything(SEED)

    dataset = ImdbCorruptDataModule(
        tokenizer_name=_config["model_name"],
        max_sequence_length=_config["max_sequence_length"],
        batch_size=_config["gpu_batch_size"],
        corrupt_percentage=_config["corrupt_percentage"],
        corrupt_num_samples=_config["corrupt_num_samples"],
        num_train_samples=_config["num_train_samples"],
        num_val_samples=_config["num_val_samples"],
        val_is_test=True,
        seed=_config["seed"],
    )

    model = BinaryClassificationModel(
        model_name=_config["model_name"],
        num_labels=2,
        lr=_config["lr"],
        smoothing=_config["label_smoothing"],
        batch_size=_config["gpu_batch_size"],
        weight_decay=_config["weight_decay"],
        warmup_steps=_config["warmup_steps"],
        num_workers=_config["num_workers"],
        # gradient_accumulation_steps=_config["gradient_accumulation_steps"], # TODO
    )

    # loggers
    loggers = get_logging_callback(_config)

    # callbacks
    callbacks: List[pl.callbacks.Callback] = []
    lr_monitor = (LearningRateMonitor(logging_interval="step"),)
    early_stopping = EarlyStopping(
        monitor="val_loss", min_delta=0.0, patience=3, verbose=True, mode="max", strict=True,
    )
    if _config["early_stopping"]:
        callbacks.append(early_stopping)
    if _config["lr_monitor"]:
        callbacks.append(lr_monitor)

    if _config["checkpoints_dir"] is not None:
        checkpoint_dir = os.path.join(_config["checkpoints_dir"], _config["experiment_id"])
        logger.info(f"Storing checkpoints in: {checkpoint_dir}")
        checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=checkpoint_dir)
        callbacks.append(checkpoint_callback)
    else:
        logger.warning("No checkpoint directory specified: no checkpoints will be saved")

    trainer = pl.Trainer(
        # profiler=True,
        deterministic=True,
        fast_dev_run=False,
        precision=16 if _config["fp16"] else 32,
        logger=loggers,
        max_epochs=_config["max_epochs"],
        gradient_clip_val=1.0,
        gpus=1 if torch.cuda.is_available() else 0,
        log_every_n_steps=_config["log_every_n_steps"],
        callbacks=callbacks,
    )

    trainer.fit(model=model, datamodule=dataset)


def get_args():
    def is_valid_percentage(p):
        """Check if value in range [0.0, 1.0]."""
        try:
            p = float(p)
        except ValueError:
            raise argparse.ArgumentTypeError("%r not a floating-point literal" % (p,))
        if p < 0.0 or p > 1.0:
            raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]" % (p,))
        return p

    parser = argparse.ArgumentParser()

    # PL trainer args
    parser.add_argument("--seed", default=42, type=int, help="random seed for initialization")
    parser.add_argument(
        "--experiment_name", default="SMS-RH-label-reconstruction-imdb", type=str, help="type (default: %(default)s)",
    )
    parser.add_argument(
        "--loggers", default=None, choices=["wandb", "tb", "all", "none", "None"], help="type (default: %(default)s)",
    )
    parser.add_argument(
        "--log_every_n_steps", default=10, type=int, help="Determine after how many iterations PL logs the results.",
    )
    parser.add_argument(
        "--checkpoints-dir", default=None, type=str, help="The base directory where checkpoints will be written.",
    )

    # General
    parser.add_argument(
        "--model_name", default="bert-base-uncased", type=str, help="type (default: %(default)s)",
    )
    parser.add_argument(
        "--gpu_batch_size", default=os.getenv("BATCH_SIZE", 16), type=int, help="type (default: %(default)s)",
    )
    parser.add_argument(
        "--max_sequence_length",
        default=256,
        type=int,
        help="The maximum total input sequence length after tokenization."
        "Applies truncation and padding to max length.",
    )
    parser.add_argument(
        "--corrupt_percentage",
        type=int,
        default=0,
        help="Percentage of how many of the labels to corrupt from 1 to 0."
        "Overwritten if --corrupt_num_samples is set. "
        "Default: Don't change the labels",
    )
    parser.add_argument(
        "--corrupt_num_samples",
        type=int,
        default=0,
        help="How many of the labels to corrupt from 1 to 0."
        "Overrides the parameters --corrupt_percentage."
        "Defualt: Don't change the labels",
    )
    parser.add_argument(
        "--num_train_samples",
        type=int,
        default=10000,
        help="Defines how many training examples to restrict the dataset to.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data", help="Set folder for where predictions will be saved.",
    )
    parser.add_argument(
        "--num_val_samples", type=int, default=-1, help="UNUSED!!",
    )
    parser.add_argument(
        "--num_label_split", type=int, default=-1, help="UNUSED!!",
    )
    parser.add_argument(
        "--val_is_test",
        action="store_true",
        default=False,
        help="Defines if there is no validation set. " "Uses test set for validation.",
    )

    # Model
    parser.add_argument("--lr", default=2e-5, type=float, help="Initial learning rate for Adam")
    parser.add_argument("--num_workers", default=4, type=int, help="type (default: %(default)s)")
    parser.add_argument("--warmup_steps", default=0, type=int, help="type (default: %(default)s)")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="type (default: %(default)s)")
    parser.add_argument(
        "--max_epochs", default=10, type=int, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--training_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override " "num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before " "performing a backward pass.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex)." "32-bit",
    )
    parser.add_argument(
        "--label_smoothing",
        default=0,
        type=is_valid_percentage,
        help="Amount of label smoothing to be applied in ranges [0.0, 1.0]."
        "Defaults to using no label smoothing (0.0)",
    )
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        default=False,
        help="If training should be stopped when validation loss decreases.",
    )
    parser.add_argument(
        "--lr_monitor",
        action="store_true",
        default=False,
        help="If learning rate should be monitored during training.",
    )
    return parser.parse_args()


def main():
    dataset_name = "imdb"  # TODO move dataset name to argparse
    exp_id = str(datetime.datetime.now().strftime("%y-%m-%d--%H:%M:%S"))

    # Start training
    _config = vars(get_args())
    _config["experiment_id"] = exp_id
    start_training(_config)

    # Write incorrect predictions to json file
    p = _config["corrupt_percentage"]
    p = 0 if p < 0 else p
    p = str(p).zfill(3)
    seed = _config["seed"]
    s = _config["label_smoothing"]
    dataname = f"seed-{seed}-{dataset_name}-p-{p}-smoothing-{s}-{exp_id}"

    output_dir = _config["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    file_path = f"{output_dir}/{dataname}.json"

    pred_list = get_prediction_dict()
    with open(file_path, "w") as f:
        json.dump(pred_list, f)


if __name__ == "__main__":
    main()
