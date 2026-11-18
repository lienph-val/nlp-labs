import argparse
import os
import random
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "conll2003")

LABEL_NAMES = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_NAMES)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}
ENTITY_TYPES = ("PER", "ORG", "LOC", "MISC")
IGNORE_INDEX = -100  # CrossEntropyLoss(ignore_index=-100): token nay khong dong gop vao loss

def load_split(split):
    """Doc 1 file parquet -> (list_words, list_labels_str); moi cau la list cac chuoi."""
    path = os.path.normpath(os.path.join(DATA_DIR, f"conll2003-{split}.parquet"))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Khong tim thay {path}")
    df = pd.read_parquet(path)
    words = [[str(t) for t in toks] for toks in df["tokens"]]
    labels = [[LABEL_NAMES[t] if not isinstance(t, str) else t for t in tags] for tags in df["ner_tags"]]
    return words, labels

# 2. Tokenize + Label alignment  (word_ids -> token_labels)
def align_labels(word_ids, word_labels, strategy="A"):
    token_labels, prev = [], None
    for wid in word_ids:
        if wid is None:
            token_labels.append(IGNORE_INDEX)
        elif wid != prev:  # subword dau tien cua tu
            token_labels.append(LABEL2ID[word_labels[wid]])
        else:  # subword tiep theo cua cung 1 tu
            if strategy == "A":
                token_labels.append(IGNORE_INDEX)
            else:
                lab = word_labels[wid]
                if lab.startswith("B-"):
                    lab = "I-" + lab[2:]
                token_labels.append(LABEL2ID[lab])
        prev = wid
    return token_labels


def encode_sentence(tokenizer, words, word_labels, strategy="A", max_length=128):
    """1 cau -> dict(input_ids, attention_mask, word_ids, labels) chua padding."""
    enc = tokenizer(words, is_split_into_words=True, truncation=True, max_length=max_length)
    word_ids = enc.word_ids()  # can fast tokenizer
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "word_ids": word_ids,
        "labels": align_labels(word_ids, word_labels, strategy),
    }


class NERDataset(torch.utils.data.Dataset):
    def __init__(self, words, labels, tokenizer, strategy="A", max_length=128):
        self.items = [encode_sentence(tokenizer, w, l, strategy, max_length) for w, l in zip(words, labels)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def make_collate(pad_token_id):
    """Pad dong theo batch dai nhat: input_ids -> pad_id, attention_mask -> 0, labels -> -100."""
    def collate(batch):
        T = max(len(x["input_ids"]) for x in batch)
        ids = torch.full((len(batch), T), pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(batch), T), dtype=torch.long)
        labs = torch.full((len(batch), T), IGNORE_INDEX, dtype=torch.long)
        for i, x in enumerate(batch):
            n = len(x["input_ids"])
            ids[i, :n] = torch.tensor(x["input_ids"])
            mask[i, :n] = torch.tensor(x["attention_mask"])
            labs[i, :n] = torch.tensor(x["labels"])
        return {"input_ids": ids, "attention_mask": mask, "labels": labs,
                "word_ids": [x["word_ids"] for x in batch]}
    return collate

# 3. Model: Pretrained Transformer + Linear head  (Z = H W + b)
def build_model(model_name):
    return AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(LABEL_NAMES), id2label=ID2LABEL, label2id=LABEL2ID)


@torch.no_grad()
def inspect_pipeline(tokenizer, model, words, word_labels, strategy, device, title=""):
    """In ra TUNG BUOC: tokens, input_ids, attention_mask, word_ids, labels, logits, predictions + shapes."""
    model.eval()
    item = encode_sentence(tokenizer, words, word_labels, strategy)
    ids = torch.tensor([item["input_ids"]], device=device)
    mask = torch.tensor([item["attention_mask"]], device=device)

    out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
    H = out.hidden_states[-1]  # B x T x d
    Z = out.logits             # B x T x C
    pred = Z.argmax(-1)[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(item["input_ids"])

    print(f"\n----- INSPECT {title} (strategy {strategy}) -----")
    print("words      :", words)
    print("word labels:", word_labels)
    print(f"shapes     : H {tuple(H.shape)} = B x T x d | Z (logits) {tuple(Z.shape)} = B x T x C | "
          f"Y (labels) (1, {len(item['labels'])})")
    print(f"{'idx':>3} {'token':<14} {'input_id':>8} {'mask':>4} {'word_id':>7} {'label':<8} {'pred':<8} logits[0:3]")
    for i, tok in enumerate(tokens):
        lab = item["labels"][i]
        lab_s = "IGNORE" if lab == IGNORE_INDEX else ID2LABEL[lab]
        wid = "None" if item["word_ids"][i] is None else item["word_ids"][i]
        lg = ", ".join(f"{v:.2f}" for v in Z[0, i, :3].tolist())
        print(f"{i:>3} {tok:<14} {item['input_ids'][i]:>8} {item['attention_mask'][i]:>4} {str(wid):>7} "
              f"{lab_s:<8} {ID2LABEL[pred[i]]:<8} [{lg}, ...]")

# 4. Training (loss bo qua token IGNORE = -100)
def train(model, loader, device, epochs=3, lr=5e-5):
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = epochs * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: max(0.0, 1 - s / max(1, total)))
    model.train()
    step = 0
    for ep in range(epochs):
        t0, run = time.time(), 0.0
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            # Khi truyen labels, model tinh CrossEntropyLoss(ignore_index=-100) tren logits (B*T, C)
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad()
            run += loss.item()
            step += 1
            if step % 100 == 0:
                print(f"  [ep {ep + 1}] step {step}/{total} loss={run / 100:.4f}")
                run = 0.0
        print(f"[train] epoch {ep + 1}/{epochs} xong ({time.time() - t0:.0f}s)")

# 5. Prediction -> nhan theo TU + danh gia entity-level
@torch.no_grad()
def predict_words(model, loader, device, words_list):
    """Logits -> argmax -> lay du doan o subword DAU TIEN cua moi tu => BIO label theo tu.
    Tu bi cat boi truncation khong co subword nao -> mac dinh 'O'."""
    model.eval()
    all_pred, k = [], 0
    for batch in loader:
        logits = model(input_ids=batch["input_ids"].to(device),
                       attention_mask=batch["attention_mask"].to(device)).logits
        pred_ids = logits.argmax(-1).cpu().tolist()
        for b, word_ids in enumerate(batch["word_ids"]):
            n_words = len(words_list[k])
            word_pred = ["O"] * n_words
            seen = set()
            for t, wid in enumerate(word_ids):
                if wid is not None and wid not in seen:
                    seen.add(wid)
                    word_pred[wid] = ID2LABEL[pred_ids[b][t]]
            all_pred.append(word_pred)
            k += 1
    return all_pred


def get_entities(tags):
    """BIO -> [(type, start, end_inclusive)]. 'I-X' mo dau khong sau B-X/I-X cung loai = entity moi."""
    ents, cur_type, start = [], None, None
    for i, tag in enumerate(list(tags) + ["O"]):
        prefix, typ = ("O", None) if tag == "O" else tag.split("-", 1)
        starts_new = prefix == "B" or (prefix == "I" and typ != cur_type)
        if cur_type is not None and (prefix == "O" or starts_new):
            ents.append((cur_type, start, i - 1))
            cur_type = None
        if prefix != "O" and cur_type is None:
            cur_type, start = typ, i
    return ents


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def entity_level_metrics(gold_sents, pred_sents):
    """Entity dung <=> khop ca ranh gioi lan loai. Tra ve dict type/ALL -> (P, R, F1, support)."""
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    for sid, (g, p) in enumerate(zip(gold_sents, pred_sents)):
        gset = {(sid,) + e for e in get_entities(g)}
        pset = {(sid,) + e for e in get_entities(p)}
        for e in gset & pset:
            tp[e[1]] += 1
        for e in pset - gset:
            fp[e[1]] += 1
        for e in gset - pset:
            fn[e[1]] += 1
    out = {t: (*prf(tp[t], fp[t], fn[t]), tp[t] + fn[t]) for t in ENTITY_TYPES}
    TP, FP, FN = sum(tp.values()), sum(fp.values()), sum(fn.values())
    out["ALL"] = (*prf(TP, FP, FN), TP + FN)
    return out


def report(split, gold, pred):
    res = entity_level_metrics(gold, pred)
    flat_g = [l for s in gold for l in s]
    flat_p = [l for s in pred for l in s]
    acc = sum(a == b for a, b in zip(flat_g, flat_p)) / len(flat_g)
    print(f"\n=== {split.upper()} (entity-level) ===")
    p, r, f, s = res["ALL"]
    print(f"micro: P={p:.4f} R={r:.4f} F1={f:.4f} (n={s}) | word-level acc (gom O)={acc:.4f}")
    for t in ENTITY_TYPES:
        p, r, f, s = res[t]
        print(f"  {t:<5} P={p:.4f} R={r:.4f} F1={f:.4f} (n={s})")
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="distilbert-base-cased",
                    help="model cased co fast tokenizer (vd bert-base-cased, roberta-base can add_prefix_space)")
    ap.add_argument("--strategy", choices=["A", "B"], default="A",
                    help="A: subword dau giu label, con lai IGNORE | B: lan truyen label")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--max_train_samples", type=int, default=None, help="cat bot train de chay nhanh tren CPU")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device} | model={args.model_name} | strategy={args.strategy}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    assert tokenizer.is_fast, "Can fast tokenizer de co word_ids()"
    model = build_model(args.model_name).to(device)

    # Load du lieu 
    tr_w, tr_l = load_split("train")
    va_w, va_l = load_split("validation")
    te_w, te_l = load_split("test")
    if args.max_train_samples:
        tr_w, tr_l = tr_w[:args.max_train_samples], tr_l[:args.max_train_samples]
    print(f"[load] train={len(tr_w)} validation={len(va_w)} test={len(te_w)} cau")

    # Nhin thay pipeline truoc khi train (head chua train nen pred la ngau nhien) 
    demo_words = ["Barack", "Obama", "visited", "Paris", "."]
    demo_labels = ["B-PER", "I-PER", "O", "B-LOC", "O"]
    inspect_pipeline(tokenizer, model, demo_words, demo_labels, args.strategy, device, "cau demo (truoc train)")
    # Cau co tu bi tach nhieu subword de thay ro alignment
    for w, l in zip(tr_w, tr_l):
        enc = tokenizer(w, is_split_into_words=True, truncation=True, max_length=args.max_length)
        if len(enc["input_ids"]) - 2 > len(w) + 2 and any(x != "O" for x in l):
            inspect_pipeline(tokenizer, model, w, l, args.strategy, device, "cau train co subword split (truoc train)")
            break

    collate = make_collate(tokenizer.pad_token_id)
    mk = lambda w, l: NERDataset(w, l, tokenizer, args.strategy, args.max_length)
    train_loader = DataLoader(mk(tr_w, tr_l), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(mk(va_w, va_l), batch_size=64, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(mk(te_w, te_l), batch_size=64, shuffle=False, collate_fn=collate)

    # Train
    train(model, train_loader, device, args.epochs, args.lr)

    # Danh gia entity-level
    report("validation", va_l, predict_words(model, val_loader, device, va_w))
    report("test", te_l, predict_words(model, test_loader, device, te_w))

    inspect_pipeline(tokenizer, model, demo_words, demo_labels, args.strategy, device, "cau demo (sau train)")


if __name__ == "__main__":
    main()
