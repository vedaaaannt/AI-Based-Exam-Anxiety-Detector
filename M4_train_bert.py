# Run this file in Google Colab with GPU runtime:
# Runtime → Change runtime type → T4 GPU
# !pip install transformers datasets scikit-learn torch pandas numpy --quiet

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# ── Activity 4.1 — BERT Model Selection ──────────────────────
# bert-base-uncased chosen for:
#   - Bidirectional context understanding
#   - Pre-trained on large English corpus (110M params)
#   - Uncased normalises student text capitalisation
#   - Best balance of speed and accuracy for classification

# ── Activity 4.3 — Config ────────────────────────────────────
# All training hyperparameters in one place
class Config:
    MODEL_NAME   = "bert-base-uncased"
    MAX_LEN      = 128       # max token length per input
    BATCH_SIZE   = 16        # samples per training step
    EPOCHS       = 5         # number of full passes over data
    LR           = 2e-5      # learning rate for AdamW
    WARMUP_STEPS = 100       # steps before full LR kicks in
    NUM_LABELS   = 3         # Low / Moderate / High
    SEED         = 42
    DATA_DIR     = "data/processed"   # update to Drive path in Colab
    SAVE_DIR     = "model/saved_model"
    DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()
torch.manual_seed(cfg.SEED)
np.random.seed(cfg.SEED)
print(f"Device: {cfg.DEVICE}")

LABEL2ID = {"Low Anxiety": 0, "Moderate Anxiety": 1, "High Anxiety": 2}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}


# ── Activity 4.2 — Upload to Colab ───────────────────────────
# Option A — Mount Google Drive:
#   from google.colab import drive
#   drive.mount('/content/drive')
#
# Option B — Direct file upload:
#   from google.colab import files
#   uploaded = files.upload()


# ── Activity 4.4 — Tokenization ──────────────────────────────
# Load BERT tokenizer and build PyTorch dataset
tokenizer = BertTokenizer.from_pretrained(cfg.MODEL_NAME)

class AnxietyDataset(Dataset):
    def __init__(self, texts, labels):
        # Tokenize all texts: adds [CLS], [SEP], pads to MAX_LEN
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=cfg.MAX_LEN,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }

# Show tokenization example
sample = "I'm terrified about the exam and cannot stop shaking."
enc    = tokenizer(sample, return_tensors="pt", max_length=cfg.MAX_LEN, truncation=True, padding="max_length")
tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
active = [t for t in tokens if t != "[PAD]"]
print(f"Sample tokens: {active}\n")


# Load CSV splits into dataset objects
def load_split(filename):
    path = os.path.join(cfg.DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found — run M3 first and upload to Colab")
    df = pd.read_csv(path)
    return df["text"].tolist(), df["label_id"].tolist()


# ── Activity 4.5 — Model Training ────────────────────────────
def train():
    X_train, y_train = load_split("train.csv")
    X_val,   y_val   = load_split("val.csv")
    print(f"Train: {len(X_train)} | Val: {len(X_val)}\n")

    # Wrap in Dataset and DataLoader
    train_dl = DataLoader(AnxietyDataset(X_train, y_train), batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(AnxietyDataset(X_val, y_val),     batch_size=cfg.BATCH_SIZE, shuffle=False)

    # Load pre-trained BERT with classification head (3 outputs)
    model = BertForSequenceClassification.from_pretrained(
        cfg.MODEL_NAME, num_labels=cfg.NUM_LABELS,
        id2label=ID2LABEL, label2id=LABEL2ID,
    )
    model.to(cfg.DEVICE)

    optimizer   = AdamW(model.parameters(), lr=cfg.LR, eps=1e-8)
    total_steps = len(train_dl) * cfg.EPOCHS
    scheduler   = get_linear_schedule_with_warmup(optimizer, cfg.WARMUP_STEPS, total_steps)

    best_val_acc = 0.0

    print(f"{'Epoch':<8} {'TrainLoss':<12} {'TrainAcc':<12} {'ValLoss':<12} {'ValAcc'}")
    print("─" * 56)

    for epoch in range(1, cfg.EPOCHS + 1):

        # Training pass
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for batch in train_dl:
            ids    = batch["input_ids"].to(cfg.DEVICE)
            mask   = batch["attention_mask"].to(cfg.DEVICE)
            labels = batch["labels"].to(cfg.DEVICE)

            optimizer.zero_grad()
            out  = model(input_ids=ids, attention_mask=mask, labels=labels)
            out.loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)  # prevent exploding gradients
            optimizer.step()
            scheduler.step()

            t_loss    += out.loss.item()
            t_correct += (torch.argmax(out.logits, dim=1) == labels).sum().item()
            t_total   += labels.size(0)

        train_loss = t_loss / len(train_dl)
        train_acc  = t_correct / t_total

        # Validation pass
        model.eval()
        v_loss, all_preds, all_labels = 0.0, [], []
        with torch.no_grad():
            for batch in val_dl:
                ids    = batch["input_ids"].to(cfg.DEVICE)
                mask   = batch["attention_mask"].to(cfg.DEVICE)
                labels = batch["labels"].to(cfg.DEVICE)
                out    = model(input_ids=ids, attention_mask=mask, labels=labels)
                v_loss += out.loss.item()
                all_preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(all_labels, all_preds)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(cfg.SAVE_DIR, exist_ok=True)
            model.save_pretrained(cfg.SAVE_DIR)
            tokenizer.save_pretrained(cfg.SAVE_DIR)
            saved = "✅"
        else:
            saved = ""

        print(f"{epoch:<8} {train_loss:<12.4f} {train_acc:<12.4f} {v_loss/len(val_dl):<12.4f} {val_acc:.4f}  {saved}")

    return model, all_preds, all_labels


# ── Activity 4.6 — Evaluation & Save Model ───────────────────
def evaluate(model, val_preds, val_labels):
    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds, target_names=list(LABEL2ID.keys()), digits=4))

    print("Confusion Matrix:")
    cm     = confusion_matrix(val_labels, val_preds)
    labels = list(LABEL2ID.keys())
    print("".join(f"{l[:8]:>12}" for l in labels))
    for i, row in enumerate(cm):
        print(f"{labels[i]:<20}" + "".join(f"{v:>12}" for v in row))

    # Save label map alongside model weights
    with open(os.path.join(cfg.SAVE_DIR, "label_map.json"), "w") as f:
        json.dump({"LABEL2ID": LABEL2ID, "ID2LABEL": ID2LABEL}, f, indent=2)

    print(f"\nModel saved → {cfg.SAVE_DIR}/")

    # Download from Colab after training:
    # import shutil
    # shutil.make_archive('saved_model', 'zip', cfg.SAVE_DIR)
    # from google.colab import files
    # files.download('saved_model.zip')


if __name__ == "__main__":
    print("MILESTONE 4 - BERT Model Training\n")
    model, val_preds, val_labels = train()
    evaluate(model, val_preds, val_labels)
    print("\nMILESTONE 4 COMPLETE")
