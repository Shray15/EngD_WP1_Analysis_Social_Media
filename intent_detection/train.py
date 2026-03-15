"""
Unified intent-classification fine-tuning script.

Trains one of three Dutch transformer models (BERTje, DeBERTa, RoBERTa) on a
labelled Excel dataset using stratified k-fold cross-validation.  The fold with
the lowest validation loss is selected, evaluated on a held-out test set, and
saved as the final model.

Usage
-----
    python intent_detection/train.py --model bert_dutch
    python intent_detection/train.py --model deberta --data path/to/data.xlsx
    python intent_detection/train.py --model roberta  --seed 0

Supported model keys (see config.MODEL_NAMES / config.MODEL_HYPERPARAMS):
    bert_dutch  — GRoNLP/bert-base-dutch-cased
    deberta     — MoritzLaurer/mDeBERTa-v3-base-mnli-xnli
    roberta     — DTAI-KULeuven/robbert-2023-dutch-large
"""

import argparse
import logging
import math
import os
import shutil
import stat
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Suppress noisy but harmless warnings from transformers and torch.
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# Project-level config (paths, model names, hyperparams).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from intent_utils.intent_train_test_preprocess import preprocess

logger = logging.getLogger(__name__)

# Column names used in the Excel training file.
TEXT_COL = config.COLUMN_NAMES["synthetic_data"]  # "Synthetic Data"
LABEL_COL = config.COLUMN_NAMES["intent"]          # "Intent"

# Special tokens added to every tokenizer.
SPECIALS = ["<PERSON>", "<ORG>"]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_data(path: str) -> tuple:
    """Load and preprocess an Excel training file.

    Parameters
    ----------
    path:
        Path to the ``.xlsx`` file containing ``TEXT_COL`` and ``LABEL_COL``.

    Returns
    -------
    texts : list[str]
        Preprocessed text strings.
    labels : numpy.ndarray
        Integer-encoded label array.
    encoder : LabelEncoder
        Fitted encoder (used to map integers back to class names).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training data not found: {path}")

    data = pd.read_excel(path)
    if TEXT_COL not in data.columns or LABEL_COL not in data.columns:
        raise KeyError(
            f"Expected columns '{TEXT_COL}' and '{LABEL_COL}'. "
            f"Found: {list(data.columns)}"
        )

    data = data.dropna(subset=[TEXT_COL, LABEL_COL]).reset_index(drop=True)
    logger.info("Loaded %d rows from %s", len(data), path)

    data["content_preprocessed"] = data[TEXT_COL].apply(preprocess)

    encoder = LabelEncoder()
    labels = encoder.fit_transform(data[LABEL_COL])
    texts = data["content_preprocessed"].tolist()

    logger.info("Classes: %s", encoder.classes_.tolist())
    return texts, labels, encoder


def make_tokenizer(model_name: str) -> tuple:
    """Load a tokenizer and register domain-specific special tokens.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier.

    Returns
    -------
    tokenizer : PreTrainedTokenizerFast
    num_added : int
        Number of special tokens actually added (may be 0 if already present).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": SPECIALS}
    )
    logger.info(
        "Tokenizer loaded (%s). Special tokens added: %d", model_name, num_added
    )
    return tokenizer, num_added


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HFDictDataset(torch.utils.data.Dataset):
    """A minimal HuggingFace-compatible dataset wrapping tokenizer output.

    Parameters
    ----------
    encodings:
        Dict mapping token field names to lists of integer sequences,
        as returned by ``tokenizer(..., return_tensors=None)``.
    labels:
        Integer label array aligned with ``encodings``.
    """

    def __init__(self, encodings: dict, labels: np.ndarray) -> None:
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> dict:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self) -> int:
        return len(self.labels)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def tokenize_batch(tokenizer, text_list: list, max_length: int) -> dict:
    """Tokenise a list of texts without padding (padding done by collator).

    Parameters
    ----------
    tokenizer:
        A HuggingFace tokenizer.
    text_list:
        Raw (or preprocessed) strings to tokenise.
    max_length:
        Maximum sequence length; longer sequences are truncated.

    Returns
    -------
    dict
        Tokenizer output suitable for ``HFDictDataset``.
    """
    return tokenizer(
        text_list,
        padding=False,
        truncation=True,
        max_length=max_length,
        return_tensors=None,
    )


def compute_steps_per_epoch(
    num_examples: int, per_device_bs: int, grad_accum: int
) -> int:
    """Compute the number of optimiser steps per epoch.

    Parameters
    ----------
    num_examples:
        Training set size.
    per_device_bs:
        Per-device batch size.
    grad_accum:
        Gradient accumulation steps.

    Returns
    -------
    int
        Optimiser steps per epoch (minimum 1).
    """
    train_batches = math.ceil(num_examples / per_device_bs)
    return max(1, math.ceil(train_batches / grad_accum))


def compute_metrics(eval_pred) -> dict:
    """Compute classification metrics from HuggingFace ``EvalPrediction``.

    Returns weighted and macro precision/recall/F1 plus accuracy.
    Model selection uses ``eval_loss`` (not these metrics), but they are
    logged for human inspection.
    """
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    _, _, f1_m, _ = precision_recall_fscore_support(
        labels, preds, average="macro"
    )
    return {
        "accuracy": acc,
        "precision_weighted": prec_w,
        "recall_weighted": rec_w,
        "f1_weighted": f1_w,
        "macro_f1": f1_m,
    }


def safe_rmtree(path: str) -> None:
    """Remove a directory tree, retrying on Windows permission errors.

    On Windows, files in a directory can be locked briefly after a process
    releases them.  This function waits 100 ms and retries with a write-
    permission fix before giving up.

    Parameters
    ----------
    path:
        Directory to remove.  A no-op if the path does not exist.
    """
    if not os.path.exists(path):
        return

    def _on_error(func, p, _exc_info):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except Exception:
            pass

    time.sleep(0.1)
    shutil.rmtree(path, ignore_errors=False, onerror=_on_error)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def run_cv(
    model_name: str,
    hparams: dict,
    train_texts: list,
    train_labels: np.ndarray,
    tokenizer,
    num_added: int,
    id2label: dict,
    label2id: dict,
    num_labels: int,
    output_dir: str,
    seed: int,
    max_length: int,
    n_splits: int,
) -> tuple:
    """Run stratified k-fold cross-validation and return the best model dir.

    The fold with the lowest validation loss is kept; all other temporary
    checkpoints are cleaned up.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier.
    hparams:
        Per-model hyperparameters (from ``config.MODEL_HYPERPARAMS``).
    train_texts, train_labels:
        Training split (held-out test data excluded).
    tokenizer:
        Tokenizer (already has special tokens added).
    num_added:
        Number of special tokens added (used for embedding resize).
    id2label, label2id:
        Label index maps passed to the model config.
    num_labels:
        Number of intent classes.
    output_dir:
        Base directory for CV run artefacts.
    seed:
        Random seed for reproducibility.
    max_length:
        Tokenisation max length.
    n_splits:
        Number of CV folds.

    Returns
    -------
    best_model_dir : str
        Path to the saved best-fold model.
    fold_results : list[dict]
        Per-fold evaluation metrics.
    """
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    X = np.array(train_texts, dtype=object)
    y = np.array(train_labels)

    best_val_loss = float("inf")
    best_fold = None
    best_model_tmpdir = None
    fold_results = []

    per_device_bs = hparams["per_device_train_batch_size"]
    grad_accum = hparams["gradient_accumulation_steps"]

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        logger.info("===== Fold %d/%d =====", fold, n_splits)

        tr_texts_fold = X[tr_idx].tolist()
        tr_labels_fold = y[tr_idx]
        val_texts_fold = X[val_idx].tolist()
        val_labels_fold = y[val_idx]

        tr_enc = tokenize_batch(tokenizer, tr_texts_fold, max_length)
        val_enc = tokenize_batch(tokenizer, val_texts_fold, max_length)
        tr_dataset = HFDictDataset(tr_enc, tr_labels_fold)
        val_dataset = HFDictDataset(val_enc, val_labels_fold)

        # Fresh model per fold to avoid information leakage.
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
        if num_added > 0:
            model.resize_token_embeddings(len(tokenizer))
        model.gradient_checkpointing_enable()

        optim_steps = compute_steps_per_epoch(
            len(tr_dataset), per_device_bs, grad_accum
        )
        eval_every = max(50, optim_steps // 4)

        training_args = TrainingArguments(
            output_dir=os.path.join(output_dir, f"fold_{fold}"),
            learning_rate=hparams["learning_rate"],
            num_train_epochs=hparams["num_train_epochs"],
            per_device_train_batch_size=per_device_bs,
            gradient_accumulation_steps=grad_accum,
            per_device_eval_batch_size=2,
            evaluation_strategy="steps",
            eval_steps=eval_every,
            save_strategy="steps",
            save_steps=eval_every,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            warmup_ratio=0.06,
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            label_smoothing_factor=0.05,
            fp16=False,
            bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
            group_by_length=True,
            dataloader_num_workers=0,
            report_to=[],
            disable_tqdm=False,
            save_safetensors=True,
            seed=seed,
            eval_accumulation_steps=16,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tr_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        trainer.train()
        eval_metrics = trainer.evaluate()
        logger.info("Fold %d eval: %s", fold, eval_metrics)

        fold_val_loss = eval_metrics.get("eval_loss", float("inf"))
        fold_results.append({"fold": fold, **eval_metrics})

        if fold_val_loss < best_val_loss:
            best_val_loss = fold_val_loss
            best_fold = fold
            new_tmp_dir = os.path.join(output_dir, f"best_fold_model_tmp_{fold}")
            if os.path.exists(new_tmp_dir):
                safe_rmtree(new_tmp_dir)
            trainer.save_model(new_tmp_dir)
            tokenizer.save_pretrained(new_tmp_dir)
            if (
                best_model_tmpdir
                and best_model_tmpdir != new_tmp_dir
                and os.path.exists(best_model_tmpdir)
            ):
                safe_rmtree(best_model_tmpdir)
            best_model_tmpdir = new_tmp_dir

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info(
        "Best fold by eval_loss: Fold %d (eval_loss=%.6f)", best_fold, best_val_loss
    )
    return best_model_tmpdir, fold_results


# ---------------------------------------------------------------------------
# Test-set evaluation
# ---------------------------------------------------------------------------

def evaluate_on_test(
    best_model_dir: str,
    test_texts: list,
    test_labels: np.ndarray,
    label_encoder: LabelEncoder,
    output_dir: str,
    max_length: int,
) -> dict:
    """Load the best-fold model and evaluate it on the held-out test set.

    Parameters
    ----------
    best_model_dir:
        Directory produced by ``run_cv`` containing the best fold model.
    test_texts, test_labels:
        Held-out evaluation data (never used during CV).
    label_encoder:
        Fitted encoder used to map integer labels back to class names.
    output_dir:
        Directory for temporary trainer artefacts.
    max_length:
        Tokenisation max length.

    Returns
    -------
    dict
        Mapping of metric name to value.
    """
    best_model = AutoModelForSequenceClassification.from_pretrained(best_model_dir)
    best_tokenizer = AutoTokenizer.from_pretrained(best_model_dir)
    test_collator = DataCollatorWithPadding(tokenizer=best_tokenizer)

    test_enc = best_tokenizer(
        test_texts,
        padding=False,
        truncation=True,
        max_length=max_length,
        return_tensors=None,
    )
    test_dataset = HFDictDataset(test_enc, test_labels)

    test_trainer = Trainer(
        model=best_model,
        args=TrainingArguments(
            output_dir=os.path.join(output_dir, "final_eval"),
            per_device_eval_batch_size=2,
            eval_accumulation_steps=16,
            dataloader_num_workers=0,
            report_to=[],
            disable_tqdm=False,
        ),
        data_collator=test_collator,
        tokenizer=best_tokenizer,
        compute_metrics=compute_metrics,
    )

    logger.info("Evaluating selected model on held-out test set...")
    test_raw = test_trainer.predict(test_dataset)
    test_logits = (
        test_raw.predictions
        if not isinstance(test_raw.predictions, tuple)
        else test_raw.predictions[0]
    )
    test_preds = np.argmax(test_logits, axis=-1)
    test_labels_np = np.array(test_labels)

    acc = accuracy_score(test_labels_np, test_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        test_labels_np, test_preds, average="weighted"
    )

    logger.info("=== TEST Metrics (weighted) ===")
    logger.info("accuracy:  %.6f", acc)
    logger.info("precision: %.6f", prec)
    logger.info("recall:    %.6f", rec)
    logger.info("f1:        %.6f", f1)
    logger.info(
        "\n%s",
        classification_report(
            test_labels_np, test_preds,
            target_names=label_encoder.classes_.tolist(),
            digits=4,
        ),
    )
    logger.info("Confusion matrix:\n%s", confusion_matrix(test_labels_np, test_preds))

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1_weighted": f1}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune a Dutch transformer for intent classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=list(config.MODEL_NAMES.keys()),
        required=True,
        help="Model key (maps to config.MODEL_NAMES and config.MODEL_HYPERPARAMS).",
    )
    parser.add_argument(
        "--data",
        default=str(config.DEFAULT_DATA_PATHS["synthetic_data"]),
        help="Path to the training Excel file (.xlsx).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.MODEL_CONFIG["seed"],
        help="Random seed.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=config.CV_CONFIG["n_splits"],
        help="Number of stratified CV folds.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum tokenisation length.",
    )
    return parser.parse_args()


def main() -> None:
    """Orchestrate data loading, CV training, test evaluation, and model saving."""
    args = parse_args()

    model_name = config.MODEL_NAMES[args.model]
    hparams = config.MODEL_HYPERPARAMS[args.model]
    final_save_dir = str(config.MODEL_SAVE_PATHS[hparams["final_save_key"]])
    output_dir = str(config.CHECKPOINTS_DIR / hparams["output_dir_prefix"])

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(final_save_dir, exist_ok=True)
    set_seed(args.seed)

    logger.info("Model:      %s (%s)", args.model, model_name)
    logger.info("Data:       %s", args.data)
    logger.info("Output dir: %s", output_dir)
    logger.info("Final save: %s", final_save_dir)

    # 1. Load data
    texts_all, y_all, label_encoder = load_data(args.data)
    num_labels = len(label_encoder.classes_)
    id2label = {i: lab for i, lab in enumerate(label_encoder.classes_.tolist())}
    label2id = {v: k for k, v in id2label.items()}

    # 2. Hold out 20 % as a test set (never used for fold selection)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts_all, y_all,
        test_size=0.2,
        stratify=y_all,
        random_state=args.seed,
    )

    # 3. Tokenizer
    tokenizer, num_added = make_tokenizer(model_name)

    # 4. Cross-validation
    best_model_dir, fold_results = run_cv(
        model_name=model_name,
        hparams=hparams,
        train_texts=train_texts,
        train_labels=train_labels,
        tokenizer=tokenizer,
        num_added=num_added,
        id2label=id2label,
        label2id=label2id,
        num_labels=num_labels,
        output_dir=output_dir,
        seed=args.seed,
        max_length=args.max_length,
        n_splits=args.n_splits,
    )

    # Save fold results CSV for later comparison.
    fold_csv = os.path.join(output_dir, "fold_results.csv")
    pd.DataFrame(fold_results).to_csv(fold_csv, index=False)
    logger.info("Fold results saved to %s", fold_csv)

    if best_model_dir is None or not os.path.exists(best_model_dir):
        raise RuntimeError(
            "No best model directory found — did training or evaluation fail?"
        )

    # 5. Test-set evaluation
    evaluate_on_test(
        best_model_dir=best_model_dir,
        test_texts=test_texts,
        test_labels=test_labels,
        label_encoder=label_encoder,
        output_dir=output_dir,
        max_length=args.max_length,
    )

    # 6. Persist the best model as the final artefact.
    best_model = AutoModelForSequenceClassification.from_pretrained(best_model_dir)
    best_model.config.id2label = id2label
    best_model.config.label2id = label2id
    best_tokenizer = AutoTokenizer.from_pretrained(best_model_dir)

    best_model.save_pretrained(final_save_dir)
    best_tokenizer.save_pretrained(final_save_dir)
    logger.info("Final model saved to: %s", final_save_dir)


if __name__ == "__main__":
    main()
