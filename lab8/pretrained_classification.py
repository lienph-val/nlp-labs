"""
12 + 13. Core implementation: Pretrained Transformer + [CLS] classification head
    Input text -> Tokenizer -> input_ids + attention_mask
               -> Pretrained Transformer -> contextual representations
               -> [CLS] representation -> Linear -> logits -> Softmax -> prediction
"""

import argparse
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = SCRIPT_DIR.parent / "data" / "IMDB Dataset.csv"
LABEL_MAP = {"negative": 0, "positive": 1}
DEFAULT_MODEL = "bert-base-uncased"

# Utils

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def hardware_name(device: torch.device) -> str:
    if device.type == "cuda":
        return f"GPU: {torch.cuda.get_device_name(0)}"
    return f"CPU: {platform.processor() or platform.machine()}"


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# 12. Data + Tokenization
class TextDataset(Dataset):
    """Dataset đơn giản: danh sách text + label."""

    def __init__(self, texts, labels):
        self.texts = list(texts)
        self.labels = list(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return {"text": self.texts[i], "label": self.labels[i]}

    def head(self, n):
        return TextDataset(self.texts[:n], self.labels[:n])


def resolve_data_path(data_path=None) -> Path:
    candidates = [Path(data_path)] if data_path else []
    candidates += [
        DEFAULT_DATA_PATH,
        Path.cwd() / "data" / "IMDB Dataset.csv",
        Path.cwd().parent / "data" / "IMDB Dataset.csv",
        Path.cwd() / "IMDB Dataset.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Không tìm thấy 'IMDB Dataset.csv'. Đã thử:\n  "
        + "\n  ".join(str(c) for c in candidates)
        + "\nTruyền data_path=... để chỉ định đường dẫn."
    )


def load_imdb_splits(data_path=None, seed=42, train_fraction=1.0,
                     max_train_samples=None, max_val_samples=None,
                     max_test_samples=None):
    # Chia stratified 80/10/10
    path = resolve_data_path(data_path)
    df = pd.read_csv(path)
    df["label"] = df["sentiment"].str.lower().map(LABEL_MAP)
    df = df.dropna(subset=["review", "label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    train_df, rest_df = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        rest_df, test_size=0.5, random_state=seed, stratify=rest_df["label"]
    )

    def to_ds(d):
        return TextDataset(d["review"].tolist(), d["label"].tolist())

    train_ds, val_ds, test_ds = to_ds(train_df), to_ds(val_df), to_ds(test_df)

    if max_train_samples:
        train_ds = train_ds.head(max_train_samples)
    if train_fraction < 1.0:
        train_ds = train_ds.head(int(len(train_ds) * train_fraction))
    if max_val_samples:
        val_ds = val_ds.head(max_val_samples)
    if max_test_samples:
        test_ds = test_ds.head(max_test_samples)
    return train_ds, val_ds, test_ds


def make_loader(dataset, tokenizer, max_length=128, batch_size=16, shuffle=False):
    """DataLoader: tokenize theo batch với padding + truncation, trả về tensor PyTorch."""

    def collate(batch):
        texts = [b["text"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        tokens = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )  # keys: input_ids, attention_mask (và token_type_ids với BERT)
        tokens["labels"] = labels
        return tokens

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


# 13. Classification head
class PretrainedClassifier(nn.Module):
    """Pretrained encoder + [CLS] + Linear.
    Classification head (Dropout + Linear) là thành phần task-specific, khởi tạo
    ngẫu nhiên; encoder được nạp trọng số pretrained (theta_pretrained).
    freeze_encoder=True -> chỉ train head (feature extraction).
    """

    def __init__(self, model_name=DEFAULT_MODEL, num_labels=2, dropout=0.1,
                 freeze_encoder=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)
        self.set_encoder_trainable(not freeze_encoder)

    def set_encoder_trainable(self, trainable: bool):
        for p in self.encoder.parameters():
            p.requires_grad = trainable

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(**kwargs)
        # last_hidden_state: (batch, seq_len, hidden); vị trí 0 là [CLS]
        cls_repr = out.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls_repr))  # logits (chưa softmax)

    @torch.no_grad()
    def predict_proba(self, input_ids, attention_mask, token_type_ids=None):
        self.eval()
        return torch.softmax(self(input_ids, attention_mask, token_type_ids), dim=-1)


# Train / Evaluate
def _batch_to_device(batch, device):
    return (
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["token_type_ids"].to(device) if "token_type_ids" in batch else None,
        batch["labels"].to(device),
    )


def train_one_epoch(model, loader, optimizer, scheduler, device, loss_fn,
                    log_every=50):
    model.train()
    total = 0.0
    for step, batch in enumerate(loader, 1):
        ids, mask, tti, labels = _batch_to_device(batch, device)
        optimizer.zero_grad()
        loss = loss_fn(model(ids, mask, tti), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total += loss.item()
        if log_every and step % log_every == 0:
            print(f"  step {step}/{len(loader)} - loss {total / step:.4f}")
    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    """Trả về (metrics, labels, preds, probs, confidence)."""
    model.eval()
    all_labels, all_probs = [], []
    for batch in loader:
        ids, mask, tti, labels = _batch_to_device(batch, device)
        probs = torch.softmax(model(ids, mask, tti), dim=-1)  # softmax(logits)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    labels = np.concatenate(all_labels)
    probs = np.concatenate(all_probs)
    preds = probs.argmax(axis=1)
    confidence = probs.max(axis=1)  # confidence = probabilities.max()

    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }
    return metrics, labels, preds, probs, confidence


def fit(model, train_loader, val_loader, device, epochs=2, lr=2e-5,
        warmup_ratio=0.1, log_every=50):
    """Train model; trả về (history, train_time_seconds). Optimizer: AdamW."""
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(warmup_ratio * total_steps), total_steps
    )
    loss_fn = nn.CrossEntropyLoss()

    history = []
    start = time.time()
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler,
                                     device, loss_fn, log_every)
        val_metrics, *_ = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        **{f"val_{k}": v for k, v in val_metrics.items()}})
        print(f"  train_loss={train_loss:.4f} | val_acc={val_metrics['accuracy']:.4f} "
              f"| val_f1={val_metrics['f1']:.4f}")
    return history, time.time() - start

def main():
    p = argparse.ArgumentParser(description="Smoke test cho core implementation")
    p.add_argument("--model_name", default=DEFAULT_MODEL)
    p.add_argument("--data_path", default=None)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--freeze_encoder", action="store_true")
    p.add_argument("--max_train_samples", type=int, default=200)
    p.add_argument("--max_val_samples", type=int, default=100)
    p.add_argument("--max_test_samples", type=int, default=100)
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    print("Device:", hardware_name(device))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds, val_ds, test_ds = load_imdb_splits(
        args.data_path, args.seed, 1.0,
        args.max_train_samples, args.max_val_samples, args.max_test_samples,
    )
    print(f"Train/Val/Test = {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    train_loader = make_loader(train_ds, tokenizer, args.max_length, args.batch_size, True)
    val_loader = make_loader(val_ds, tokenizer, args.max_length, args.batch_size)
    test_loader = make_loader(test_ds, tokenizer, args.max_length, args.batch_size)

    # Kiểm tra shape một batch
    batch = next(iter(train_loader))
    print("Batch keys:", list(batch.keys()))
    print("input_ids:", tuple(batch["input_ids"].shape),
          "| attention_mask:", tuple(batch["attention_mask"].shape))

    model = PretrainedClassifier(args.model_name, freeze_encoder=args.freeze_encoder)
    total, trainable = count_parameters(model)
    print(f"Params: total={total:,} | trainable={trainable:,}")

    model.to(device)
    with torch.no_grad():
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                       batch["token_type_ids"].to(device) if "token_type_ids" in batch else None)
    print("logits shape:", tuple(logits.shape))

    history, secs = fit(model, train_loader, val_loader, device,
                        epochs=args.epochs, lr=args.lr)
    metrics, *_ = evaluate(model, test_loader, device)
    print(f"Train time: {secs:.1f}s")
    print("Test:", {k: round(float(v), 4) for k, v in metrics.items()})


if __name__ == "__main__":
    main()

"""
Kết quả chạy thử:
Device: CPU: Intel64 Family 6 Model 154 Stepping 4, GenuineIntel
Train/Val/Test = 200/100/100
Batch keys: ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
input_ids: (16, 128) | attention_mask: (16, 128)
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
D:\programPC\Miniconda\Lib\site-packages\huggingface_hub\file_download.py:149: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\Users\PC\.cache\huggingface\hub. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.
To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. In order to activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development
  warnings.warn(message)
D:\programPC\Miniconda\Lib\site-packages\huggingface_hub\file_download.py:149: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\Users\PC\.cache\huggingface\hub\models--bert-base-uncased. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.
To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. In order to activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development
  warnings.warn(message)

Loading weights:   0%|          | 0/199 [00:00<?, ?it/s]
Loading weights:   5%|▌         | 10/199 [00:00<00:02, 87.75it/s]
Loading weights:  10%|▉         | 19/199 [00:00<00:03, 48.37it/s]
Loading weights:  13%|█▎        | 25/199 [00:00<00:04, 41.40it/s]
Loading weights:  23%|██▎       | 46/199 [00:00<00:02, 69.60it/s]
Loading weights:  28%|██▊       | 56/199 [00:00<00:01, 72.34it/s]
Loading weights:  34%|███▍      | 68/199 [00:00<00:01, 82.00it/s]
Loading weights: 100%|██████████| 199/199 [00:01<00:00, 198.77it/s]
[transformers] [1mBertModel LOAD REPORT[0m from: bert-base-uncased
Key                                        | Status     |  | 
-------------------------------------------+------------+--+-
cls.predictions.transform.dense.weight     | UNEXPECTED |  | 
cls.seq_relationship.bias                  | UNEXPECTED |  | 
cls.predictions.bias                       | UNEXPECTED |  | 
cls.predictions.transform.LayerNorm.bias   | UNEXPECTED |  | 
cls.predictions.transform.dense.bias       | UNEXPECTED |  | 
cls.predictions.transform.LayerNorm.weight | UNEXPECTED |  | 
cls.seq_relationship.weight                | UNEXPECTED |  | 

Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
Params: total=109,483,778 | trainable=109,483,778
logits shape: (16, 2)
Epoch 1/1
  train_loss=0.6929 | val_acc=0.7100 | val_f1=0.7010
Train time: 266.8s
Test: {'accuracy': 0.63, 'precision': 0.6444, 'recall': 0.58, 'f1': 0.6105}
"""
