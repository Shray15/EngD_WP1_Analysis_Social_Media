import pandas as pd

data = pd.read_excel(r"C:\Users\20245179\OneDrive - TU Eindhoven\LLM_EngD_project\Intent recognition Comments\Synthetic_data\New_synthethic_data_generation.xlsx")

data.dropna(subset=['Synthetic Data'], inplace=True)
# data.drop_duplicates(subset=['Intent'], inplace=True)
data.reset_index(drop=True, inplace=True)
print("Number of rows after dropping NaN and duplicates:", len(data)    )
# Check the distribution of classes
print(data["Intent"].value_counts())
data.reset_index(drop=True, inplace=True)

# %% ===================== True Stratified K-Fold CV (best-by eval_loss) =====================
import os
import re
import time
import stat
import math
import shutil
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
)

# ---------------------------
# 0) Config
# ---------------------------
MODEL_NAME     = "DTAI-KULeuven/robbert-2023-dutch-large"
DATA_EXCEL      = r"PATH_TO_SYNTHETIC_DATASET"  # <- set this to  dataset # <- set this to your dataset
TEXT_COL       = "Synthetic Data"
LABEL_COL      = "Intent"

MAX_LENGTH     = 512
SEED           = 42
N_SPLITS       = 4

OUTPUT_DIR     = "./cv_runs_robbert_eval_loss"
FINAL_SAVE_DIR = r"MODEL_SAVE_PATH\intent_fine_tuned_4_intents_robert_new3_CV"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FINAL_SAVE_DIR, exist_ok=True)
set_seed(SEED)

from intent_utils.intent_train_test_preprocess import preprocess
# ---------------------------
# 2) Load data
# ---------------------------
if not os.path.exists(DATA_EXCEL):
    raise FileNotFoundError(f"DATA_EXCEL not found: {DATA_EXCEL}")

data = pd.read_excel(DATA_EXCEL)
if TEXT_COL not in data.columns or LABEL_COL not in data.columns:
    raise KeyError(f"Expected columns '{TEXT_COL}' and '{LABEL_COL}'. Found: {list(data.columns)}")

data = data.dropna(subset=[TEXT_COL, LABEL_COL]).reset_index(drop=True)
data["content_preprocessed"] = data[TEXT_COL].apply(preprocess)

# ---------------------------
# 3) Labels
# ---------------------------
le = LabelEncoder()
y_all = le.fit_transform(data[LABEL_COL])
texts_all = data["content_preprocessed"].tolist()
num_labels = len(le.classes_)
id2label = {i: lab for i, lab in enumerate(le.classes_.tolist())}
label2id = {v: k for k, v in id2label.items()}

# ---------------------------
# 4) Held-out TEST set (never used for model selection)
# ---------------------------
train_texts_all, test_texts, train_labels_all, test_labels = train_test_split(
    texts_all, y_all, test_size=0.2, stratify=y_all, random_state=SEED
)

# ---------------------------
# 5) Tokenizer (+ special tokens) & collator
# ---------------------------
SPECIALS = ["<PERSON>", "<ORG>"]
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
num_added = tokenizer.add_special_tokens({"additional_special_tokens": SPECIALS})
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def tokenize_batch(text_list):
    return tokenizer(
        text_list,
        padding=False,         # dynamic padding via collator
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors=None
    )
def compute_steps_per_epoch(num_examples: int, per_device_bs: int, grad_accum: int) -> int:
    train_batches = math.ceil(num_examples / per_device_bs)
    return max(1, math.ceil(train_batches / grad_accum))

class HFDictDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    def __len__(self):
        return len(self.labels)

def new_fold_model():
    """Fresh model for a fold, resized for special tokens, with label maps."""
    m = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    if num_added > 0:
        m.resize_token_embeddings(len(tokenizer))
    # Memory saver for large model
    m.gradient_checkpointing_enable()
    return m

# ---------------------------
# 6) Metrics (reporting only; selection uses eval_loss)
# ---------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(labels, preds, average="macro")
    return {
        "accuracy": acc,
        "precision_weighted": prec_w,
        "recall_weighted": rec_w,
        "f1_weighted": f1_w,
        "macro_f1": f1_m,
    }

# ---------------------------
# 7) Windows-safe directory removal
# ---------------------------
def windows_safe_rmtree(path):
    if not os.path.exists(path):
        return
    def _onerror(func, p, exc_info):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except Exception:
            pass
    time.sleep(0.1)
    shutil.rmtree(path, ignore_errors=False, onerror=_onerror)

# ---------------------------
# 8) K-Fold CV on the training portion only
# ---------------------------
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
best_val_loss = float("inf")
best_fold = None
best_model_tmpdir = None
fold_results = []

X = np.array(train_texts_all, dtype=object)
y = np.array(train_labels_all)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
    print(f"\n===== Fold {fold}/{N_SPLITS} =====")
    tr_texts, tr_labels = X[tr_idx].tolist(), y[tr_idx]
    val_texts, val_labels = X[val_idx].tolist(), y[val_idx]

    tr_enc = tokenize_batch(tr_texts)
    val_enc = tokenize_batch(val_texts)
    tr_dataset = HFDictDataset(tr_enc, tr_labels)
    val_dataset = HFDictDataset(val_enc, val_labels)

    # Fresh model for this fold
    model = new_fold_model()

    # Compute eval/save frequency ~ every 1/4 epoch (true optimizer steps)
    per_device_bs = 1
    grad_accum = 16
    optim_steps = compute_steps_per_epoch(len(tr_dataset), per_device_bs, grad_accum)
    eval_every  = max(50, optim_steps // 4)

    training_args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, f"fold_{fold}"),
        learning_rate=2e-5,
        num_train_epochs=3,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        per_device_eval_batch_size=2,
        evaluation_strategy="steps",
        eval_steps=eval_every,
        save_strategy="steps",
        save_steps=eval_every,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",   # <- select best by eval_loss
        greater_is_better=False,            # <- lower eval_loss is better
        warmup_ratio=0.06,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        label_smoothing_factor=0.05,
        fp16=False,  # set bf16=True if your GPU supports it
        bf16=False,
        group_by_length=True,
        dataloader_num_workers=0,
        report_to=[],
        disable_tqdm=False,
        save_safetensors=True,
        seed=SEED,
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
    eval_metrics = trainer.evaluate()  # contains eval_loss + metrics
    print(f"Fold {fold} eval: {eval_metrics}")

    fold_val_loss = eval_metrics.get("eval_loss", np.inf)
    fold_results.append({"fold": fold, **eval_metrics})

    # Keep the lowest eval_loss fold
    if fold_val_loss < best_val_loss:
        best_val_loss = fold_val_loss
        best_fold = fold

        new_tmp_dir = os.path.join(OUTPUT_DIR, f"best_fold_model_tmp_{fold}")
        if os.path.exists(new_tmp_dir):
            windows_safe_rmtree(new_tmp_dir)
        trainer.save_model(new_tmp_dir)
        tokenizer.save_pretrained(new_tmp_dir)

        if best_model_tmpdir and best_model_tmpdir != new_tmp_dir and os.path.exists(best_model_tmpdir):
            windows_safe_rmtree(best_model_tmpdir)
        best_model_tmpdir = new_tmp_dir

    # Optional: free memory between folds
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"\nBest fold by eval_loss: Fold {best_fold} (eval_loss={best_val_loss:.6f})")

# Save fold results
pd.DataFrame(fold_results).to_csv(os.path.join(OUTPUT_DIR, "fold_results.csv"), index=False)

# ---------------------------
# 9) Evaluate the SELECTED model on the held-out TEST set
# ---------------------------
if best_model_tmpdir is None or not os.path.exists(best_model_tmpdir):
    raise RuntimeError("No best model directory found. Did training/evaluation fail?")

best_model = AutoModelForSequenceClassification.from_pretrained(best_model_tmpdir)
best_tokenizer = AutoTokenizer.from_pretrained(best_model_tmpdir)
test_collator = DataCollatorWithPadding(tokenizer=best_tokenizer)

test_enc = best_tokenizer(
    test_texts, padding=False, truncation=True, max_length=MAX_LENGTH, return_tensors=None
)
test_dataset = HFDictDataset(test_enc, test_labels)

test_trainer = Trainer(
    model=best_model,
    args=TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "final_eval"),
        per_device_eval_batch_size=2,
        eval_accumulation_steps=16,
        dataloader_num_workers=0,
        report_to=[],
        disable_tqdm=False
    ),
    data_collator=test_collator,
    tokenizer=best_tokenizer,
    compute_metrics=compute_metrics
)

print("\nEvaluating selected model on HELD-OUT TEST set...")
test_raw = test_trainer.predict(test_dataset)
test_logits = test_raw.predictions if not isinstance(test_raw.predictions, tuple) else test_raw.predictions[0]
test_preds = np.argmax(test_logits, axis=-1)
test_labels_np = np.array(test_labels)

test_acc = accuracy_score(test_labels_np, test_preds)
test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(
    test_labels_np, test_preds, average="weighted"
)

print("\n=== TEST Metrics (weighted) for SELECTED FOLD MODEL ===")
print(f"accuracy:  {test_acc:.6f}")
print(f"precision: {test_prec:.6f}")
print(f"recall:    {test_rec:.6f}")
print(f"f1:        {test_f1:.6f}")

print("\n=== Per-class Report (held-out test) ===")
print(classification_report(test_labels_np, test_preds, target_names=le.classes_.tolist(), digits=4))

print("\nConfusion Matrix (held-out test):")
print(confusion_matrix(test_labels_np, test_preds))

# ---------------------------
# 10) Save the selected model as your final model
# ---------------------------
# keep label maps in config for nice decoding later
best_model.config.id2label = id2label
best_model.config.label2id = label2id

best_model.save_pretrained(FINAL_SAVE_DIR)
best_tokenizer.save_pretrained(FINAL_SAVE_DIR)
print(f"\nFinal model (best fold by eval_loss) saved to: {FINAL_SAVE_DIR}")
