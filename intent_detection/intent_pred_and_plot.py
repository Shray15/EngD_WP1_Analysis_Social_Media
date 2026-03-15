"""
Intent prediction and distribution plotting using three fine-tuned models.

Loads BERTje, RoBERTa, and DeBERTa models saved by ``intent_detection/train.py``,
runs batched inference on a CSV of comments, applies majority voting across the
three models, and saves predictions alongside a bar-chart of the final label
distribution.

Paths are resolved from ``config.py``; update ``config.DEFAULT_DATA_PATHS`` and
``config.MODEL_SAVE_PATHS`` for your environment instead of editing this file.

Usage
-----
    python intent_detection/intent_pred_and_plot.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config
from intent_utils.intent_train_test_preprocess import preprocess

logger = logging.getLogger(__name__)

# Intent classes must match the order used during training.
LABELS = ["Appreciation", "Criticism", "Inquiry", "Statement"]

# Map internal model names to config.MODEL_SAVE_PATHS keys.
_MODEL_KEY_MAP = {
    "GRONLP_new3_CV": "bert_dutch",
    "robert_new3_CV": "roberta",
    "debertaV3_new3_CV": "deberta",
}

# Short tag used as a column prefix in the combined output DataFrame.
_TAG_MAP = {
    "GRONLP_new3_CV": "GRO_NLP",
    "robert_new3_CV": "roberta",
    "debertaV3_new3_CV": "debertaV3",
}


def run_model_and_collect(
    model_dir: str,
    tok_dir: str,
    model_tag: str,
    texts: list,
    labels: list,
    original_comments: "pd.Series",
    batch_size: int = 32,
    max_length: int = 512,
) -> pd.DataFrame:
    """Run batched inference with one model and collect per-label logits and probs.

    Parameters
    ----------
    model_dir:
        Directory containing a saved ``AutoModelForSequenceClassification``.
    tok_dir:
        Directory containing the saved tokenizer (often the same as ``model_dir``).
    model_tag:
        Short identifier used as a prefix for output columns (e.g. ``"GRO_NLP"``).
    texts:
        Pre-processed text strings to classify.
    labels:
        Ordered list of intent class names.
    original_comments:
        Raw (un-preprocessed) comments to include in the output DataFrame.
    batch_size:
        Number of texts per inference batch.
    max_length:
        Tokenisation max length.

    Returns
    -------
    pd.DataFrame
        One row per input text with columns: ``Comments``, ``Labels``, and
        per-label logit/probability columns prefixed by ``model_tag``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=False)
    model.eval()

    predicted_labels = []
    per_label_logits = {lab: [] for lab in labels}
    per_label_probs = {lab: [] for lab in labels}

    for i in tqdm(range(0, len(texts), batch_size), desc=f"Predicting ({model_tag})"):
        batch_texts = texts[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.inference_mode():
            logits_t = model(**enc).logits.float()
            probs_t = torch.softmax(logits_t, dim=-1)

        logits_np = logits_t.cpu().numpy()
        probs_np = probs_t.cpu().numpy()
        preds = probs_np.argmax(axis=1)

        predicted_labels.extend([labels[p] for p in preds])
        for row_idx in range(len(batch_texts)):
            for j, lab in enumerate(labels):
                per_label_logits[lab].append(float(logits_np[row_idx][j]))
                per_label_probs[lab].append(float(probs_np[row_idx][j]))

    df = pd.DataFrame({"Comments": original_comments.reset_index(drop=True)})
    df["Labels"] = predicted_labels
    for lab in labels:
        df[f"{model_tag}_logit_{lab}"] = per_label_logits[lab]
        df[f"{model_tag}_prob_{lab}"] = per_label_probs[lab]

    return df


def majority_vote(row: pd.Series, model_cols: list) -> str:
    """Return the majority-voted label across ``model_cols`` in ``row``.

    Parameters
    ----------
    row:
        A DataFrame row containing one column per model.
    model_cols:
        Column names whose values are intent label strings.

    Returns
    -------
    str
        The winning label, or ``'undefined'`` on a three-way tie.
    """
    counts = Counter(row[col] for col in model_cols)
    most_common = counts.most_common()
    if most_common[0][1] >= 2:
        return most_common[0][0]
    return "undefined"


def main() -> None:
    """Load models, run inference, apply majority voting, save results and plot."""
    # --- Input data ---
    data_path = str(config.DEFAULT_DATA_PATHS["synthetic_data"])
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Input data not found: {data_path}\n"
            "Update config.DEFAULT_DATA_PATHS['synthetic_data'] for your environment."
        )

    logger.info("Loading data from %s", data_path)
    data = pd.read_csv(data_path)
    preprocessed_texts = [preprocess(t) for t in data["generated_comment"]]
    original_comments = data["generated_comment"]

    # --- Run all three models ---
    model_dfs = {}
    for model_internal_name, config_key in _MODEL_KEY_MAP.items():
        model_dir = str(config.MODEL_SAVE_PATHS[config_key])
        tag = _TAG_MAP[model_internal_name]
        logger.info("Running model: %s (%s)", tag, model_dir)
        model_dfs[tag] = run_model_and_collect(
            model_dir=model_dir,
            tok_dir=model_dir,
            model_tag=tag,
            texts=preprocessed_texts,
            labels=LABELS,
            original_comments=original_comments,
        )

    gro_df = model_dfs["GRO_NLP"]
    roberta_df = model_dfs["roberta"]
    deberta_df = model_dfs["debertaV3"]

    # --- Combine predictions ---
    combined = pd.DataFrame({"Comments": gro_df["Comments"]})
    combined["GRO_NLP"] = gro_df["Labels"]
    combined["roberta"] = roberta_df["Labels"]
    combined["debertaV3"] = deberta_df["Labels"]

    for df_src in [gro_df, roberta_df, deberta_df]:
        extra_cols = [c for c in df_src.columns if c not in {"Comments", "Labels"}]
        combined = combined.join(df_src[extra_cols])

    # --- Majority voting ---
    model_cols = ["GRO_NLP", "roberta", "debertaV3"]
    combined["final_label_three_models"] = combined.apply(
        majority_vote, axis=1, model_cols=model_cols
    )
    # Fall back to roberta on three-way ties.
    combined["final_label_three_models_final"] = combined.apply(
        lambda row: (
            row["roberta"]
            if row["final_label_three_models"] == "undefined"
            else row["final_label_three_models"]
        ),
        axis=1,
    )
    # Comments starting with <PERSON> are forwarded messages, not intent-bearing.
    combined.loc[
        combined["Comments"].str.startswith("<PERSON>"),
        "final_label_three_models_final",
    ] = "Forward"

    # --- Save predictions ---
    out_path = str(config.RESULTS_DIR / "intent_predictions.csv")
    combined.to_csv(out_path, index=False)
    logger.info("Predictions saved to %s", out_path)

    # --- Plot distribution ---
    label_counts = combined["final_label_three_models_final"].value_counts()
    desired_order = label_counts.index.tolist()

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        x=desired_order,
        y=[label_counts[lab] for lab in desired_order],
        palette="viridis",
    )
    for bar, count in zip(ax.patches, [label_counts[lab] for lab in desired_order]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            str(count),
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    plt.title("Intent Distribution — Final Label (Three Models)")
    plt.xlabel("Intent")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
