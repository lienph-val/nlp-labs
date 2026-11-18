
import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import Counter, OrderedDict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))        # .../nlp/lab9
DEFAULT_MODEL_DIR = "ner_model"
FALLBACK_MODEL = "dslim/bert-base-NER"
MAX_LEN = 512

# Option A - News. CoNLL-2003 có PER/ORG/LOC/MISC; IE chỉ giữ 3 loại dưới đây (MISC bị bỏ).
TYPE_MAP = {"PER": "PERSON", "ORG": "ORGANIZATION", "LOC": "LOCATION"}
TARGET_TYPES = ["PERSON", "ORGANIZATION", "LOCATION"]

GOLD_ANNOTATIONS = {
}

RESULTS_CSV = "results.csv" # kết quả đánh giá IE (entity-level)

# Dữ liệu CoNLL-2003 local (chạy từ thư mục lab9)
DEFAULT_CONLL = os.path.join(BASE_DIR, "..", "data", "conll2003", "conll2003-test.parquet")  # .../nlp/data/conll2003
CONLL_NAMES = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

DEMO_TEXT = (
    "Barack Obama visited Paris on Tuesday. "
    "Apple and Google announced new offices in Singapore, Reuters reported. "
    "Maria Garcia from the University of California Berkeley will speak in Hanoi."
)

# 30 câu tin tức ngắn để đánh giá (có thể thay bằng file riêng qua --sentences).
# Chủ ý có: entity nhiều từ, từ đa nghĩa (Washington, Jordan, Liverpool), tên người lạ, tổ chức dạng viết tắt...
SAMPLE_SENTENCES = [
    "Barack Obama visited Paris on Tuesday.",
    "Tim Cook said Apple will open a new store in Singapore.",
    "The United Nations Security Council met in New York to discuss the crisis.",
    "Vingroup and Toyota signed an agreement in Tokyo.",
    "Sundar Pichai spoke at a Google conference in Mountain View.",
    "Police in Berlin arrested two men on Friday, Reuters reported.",
    "Washington announced new trade rules, while Beijing promised a response.",
    "Maria Garcia, a professor at the University of California Berkeley, won the award.",
    "The World Health Organization warned that cases are rising in Africa.",
    "Liverpool defeated Chelsea at Anfield on Sunday.",
    "Elon Musk founded SpaceX in California.",
    "Officials from the European Union met in Brussels on Thursday.",
    "Nguyen Van An joined FPT Corporation in Hanoi last year.",
    "Michael Jordan opened a restaurant in Chicago.",
    "Jordan hosted a regional meeting in Amman.",
    "Amazon shares rose after the company reported earnings in Seattle.",
    "The Mekong River flows through Laos, Cambodia and Vietnam.",
    "Dr. Anna Schmidt of Siemens presented the results in Munich.",
    "Toyota and Honda plan to expand production in Thailand.",
    "Taylor Swift performed in London before thousands of fans.",
    "The Bank of England kept interest rates unchanged on Thursday.",
    "Students from Hanoi University of Science traveled to Seoul for the competition.",
    "Mount Everest lies between Nepal and China.",
    "Jeff Bezos and Blue Origin announced a launch from Texas.",
    "The Red Cross sent aid to Jakarta after the flood.",
    "Oracle hired Priya Sharma as its new director in Bangalore.",
    "Argentina beat France in the World Cup final in Qatar.",
    "Reuters reported that Nokia will cut jobs in Finland.",
    "Cristiano Ronaldo joined Al Nassr in Riyadh.",
    "The Hague hosted a meeting of the International Court of Justice.",
]

# 1) Tiền xử lý: câu và word (kèm offset ký tự để khôi phục đúng surface form)
WORD_RE = re.compile(r"\w+(?:[-'’]\w+)*|[^\w\s]")


def split_sentences(text):
    # Tách câu đơn giản theo . ! ? + khoảng trắng + chữ hoa. Có thể sai với viết tắt (Mr., Dr., U.S.).
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"“])", text.strip())
    return [p for p in parts if p]


def tokenize_words(sentence):
    # Trả về list[(word, start, end)] — tương tự cách CoNLL tách dấu câu thành token riêng.
    return [(m.group(), m.start(), m.end()) for m in WORD_RE.finditer(sentence)]

# 2) NER: tokenizer -> Transformer -> logits -> nhãn BIO mức word (Strategy A: subword đầu tiên)
def find_model_dir():
    # Tìm model đã fine-tune: ./ner_model, ./lab09/ner_model, hoặc /kaggle/input/**/ner_model
    for cand in [os.path.join(BASE_DIR, DEFAULT_MODEL_DIR), DEFAULT_MODEL_DIR,
                 os.path.join("lab09", DEFAULT_MODEL_DIR)]:
        if os.path.isfile(os.path.join(cand, "config.json")):
            return cand
    hits = glob.glob(f"/kaggle/input/**/{DEFAULT_MODEL_DIR}/config.json", recursive=True)
    return os.path.dirname(hits[0]) if hits else FALLBACK_MODEL


class NERTagger:
    def __init__(self, model_path=None, device=None):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        if model_path is None:
            model_path = find_model_dir()
        print(f"[NER] loading model: {model_path}", file=sys.stderr)

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path).to(self.device).eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

    def tag_words(self, words):
        # words: list[str] -> list[str] nhãn BIO, cùng độ dài với words
        if not words:
            return []
        enc = self.tokenizer(words, is_split_into_words=True, truncation=True,
                             max_length=MAX_LEN, return_tensors="pt")
        word_ids = enc.word_ids(batch_index=0) # token -> index word (None cho token đặc biệt)
        with self.torch.no_grad():
            logits = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits[0]   # (T, C)
        pred_ids = logits.argmax(-1).tolist()

        labels, prev = ["O"] * len(words), None
        for pos, w in enumerate(word_ids):
            if w is not None and w != prev:  # chỉ lấy subword đầu tiên của mỗi word
                labels[w] = self.id2label[pred_ids[pos]]
            prev = w
        return labels

# 3) BIO -> entity spans -> entity có type chuẩn hoá
def bio_to_spans(labels):
    # Trả về list[(type, start_word, end_word)] (end inclusive).
    # I-X không có B-X đứng trước (hoặc khác type) được coi là bắt đầu entity mới.
    spans, cur = [], None
    for i, lab in enumerate(labels):
        if lab == "O":
            if cur:
                spans.append(tuple(cur)); cur = None
            continue
        prefix, etype = lab.split("-", 1)
        if prefix == "B" or cur is None or cur[0] != etype:
            if cur:
                spans.append(tuple(cur))
            cur = [etype, i, i]
        else:
            cur[2] = i
    if cur:
        spans.append(tuple(cur))
    return spans


def extract_from_sentence(sentence, tagger, sentence_id=0):
    # Một câu -> list entity: {"text", "type", "start", "end", "sentence_id"} (offset ký tự trong câu)
    words = tokenize_words(sentence)
    labels = tagger.tag_words([w for w, _, _ in words])
    entities = []
    for etype, s, e in bio_to_spans(labels):
        if etype not in TYPE_MAP:  # bỏ MISC (không thuộc Option A)
            continue
        start, end = words[s][1], words[e][2]
        entities.append({"text": sentence[start:end], "type": TYPE_MAP[etype],
                         "start": start, "end": end, "sentence_id": sentence_id})
    return entities


def extract_from_text(text, tagger):
    # Raw text -> (list entity theo câu, danh sách câu)
    sentences = split_sentences(text)
    entities = []
    for sid, sent in enumerate(sentences):
        entities.extend(extract_from_sentence(sent, tagger, sid))
    return entities, sentences


def to_structured(entities):
    # Danh sách entity -> {"PERSON": [...], "ORGANIZATION": [...], "LOCATION": [...]} (unique, giữ thứ tự xuất hiện)
    out = OrderedDict((t, []) for t in TARGET_TYPES)
    for ent in entities:
        if ent["text"] not in out[ent["type"]]:
            out[ent["type"]].append(ent["text"])
    return out

# Mục 27
def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def make_gold_template(sentences, path):
    data = [{"text": s, "annotated": False, "entities": []} for s in sentences]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[gold] Đã tạo template: {path}")
    print("       Hãy TỰ gán entity cho từng câu rồi đặt \"annotated\": true. Ví dụ:")
    print('       {"text": "Barack Obama visited Paris.", "annotated": true,')
    print('        "entities": [{"text": "Barack Obama", "type": "PERSON"}, {"text": "Paris", "type": "LOCATION"}]}')
    print(f"       type hợp lệ: {', '.join(TARGET_TYPES)}. Câu không có entity: để entities = [] và annotated = true.")
    print("       (Kaggle: có thể thay vì sửa JSON, điền thẳng vào dict GOLD_ANNOTATIONS đầu file .py rồi chạy lại --evaluate)")


def load_gold(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    gold, skipped = [], 0
    for item in data:
        if not item.get("annotated", False):
            skipped += 1
            continue
        for e in item["entities"]:
            if e["type"] not in TARGET_TYPES:
                raise ValueError(f"type không hợp lệ '{e['type']}' trong câu: {item['text']}")
        gold.append(item)
    if skipped:
        print(f"[gold] Bỏ qua {skipped} câu chưa annotate (annotated = false).")
    return gold


def gold_from_dict(d):
    items = []
    for text, ents in d.items():
        for _, t in ents:
            if t not in TARGET_TYPES:
                raise ValueError(f"type không hợp lệ '{t}' trong câu: {text}")
        items.append({"text": text, "annotated": True, "entities": [{"text": x, "type": t} for x, t in ents]})
    return items


def evaluate(gold_items, tagger, out_prefix="ie"):
    tp, fp, fn = Counter(), Counter(), Counter()
    error_rows = []
    for sid, item in enumerate(gold_items):
        g = Counter((e["type"], e["text"]) for e in item["entities"])
        p = Counter((e["type"], e["text"]) for e in extract_from_sentence(item["text"], tagger, sid))
        for (t, txt), c in (g & p).items(): tp[t] += c
        for (t, txt), c in (p - g).items():
            fp[t] += c; error_rows.append([sid, item["text"], "FP (predicted, not in gold)", t, txt])
        for (t, txt), c in (g - p).items():
            fn[t] += c; error_rows.append([sid, item["text"], "FN (in gold, not predicted)", t, txt])

    rows = []
    for t in TARGET_TYPES:
        P, R, F = prf(tp[t], fp[t], fn[t])
        rows.append([t, tp[t], fp[t], fn[t], P, R, F, tp[t] + fn[t]])
    TP, FP, FN = sum(tp.values()), sum(fp.values()), sum(fn.values())
    P, R, F = prf(TP, FP, FN)
    rows.append(["ALL (micro)", TP, FP, FN, P, R, F, TP + FN])

    print(f"\nEntity-level IE evaluation — {len(gold_items)} câu")
    print(f"{'Type':<14}{'TP':>5}{'FP':>5}{'FN':>5}{'Precision':>11}{'Recall':>9}{'F1':>8}{'Support':>9}")
    for r in rows:
        print(f"{r[0]:<14}{r[1]:>5}{r[2]:>5}{r[3]:>5}{r[4]:>11.4f}{r[5]:>9.4f}{r[6]:>8.4f}{r[7]:>9}")

    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Type", "TP", "FP", "FN", "Precision", "Recall", "F1", "Support"])
        w.writerows([[r[0], r[1], r[2], r[3], round(r[4], 4), round(r[5], 4), round(r[6], 4), r[7]] for r in rows])
    with open(f"{out_prefix}_errors.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sent_id", "text", "error", "type", "entity"])
        w.writerows(error_rows)
    print(f"\nĐã ghi {RESULTS_CSV} và {out_prefix}_errors.csv ({len(error_rows)} lỗi để tự phân tích).")
    return rows


# Đọc CoNLL-2003 (parquet) -> gold + text, chạy mục 25-27 một lượt
def load_conll_sample(path, n=30, seed=42, min_len=5, max_len=40):
    # Lấy n câu ngẫu nhiên (có ít nhất 1 PER/ORG/LOC). Gold lấy từ ner_tags có sẵn; bỏ MISC.
    import random
    import pandas as pd

    df = pd.read_parquet(path)
    cands = []
    for toks, tags in zip(df["tokens"], df["ner_tags"]):
        words = [str(w) for w in toks]
        if not (min_len <= len(words) <= max_len) or words[0] == "-DOCSTART-":
            continue
        labels = [t if isinstance(t, str) else CONLL_NAMES[int(t)] for t in tags]
        spans = [sp for sp in bio_to_spans(labels) if sp[0] in TYPE_MAP]
        if spans:
            cands.append((words, spans))

    random.Random(seed).shuffle(cands)
    items = []
    for words, spans in cands[:n]:
        sentence = " ".join(words)  # nối bằng 1 khoảng trắng -> offset ký tự tính được chính xác
        offs, pos = [], 0
        for w in words:
            offs.append((pos, pos + len(w)))
            pos += len(w) + 1
        ents = [{"text": sentence[offs[s][0]:offs[e][1]], "type": TYPE_MAP[t]} for t, s, e in spans]
        items.append({"text": sentence, "annotated": True, "entities": ents})
    return items


def run_conll(args, tagger):
    gold = load_conll_sample(args.conll, args.n, args.seed)
    n_ent = sum(len(i["entities"]) for i in gold)
    print(f"[conll] {args.conll}: lấy {len(gold)} câu, {n_ent} entity gold (PERSON/ORGANIZATION/LOCATION)", file=sys.stderr)

    with open(args.gold, "w", encoding="utf-8") as f:
        json.dump(gold, f, ensure_ascii=False, indent=2)
    with open("conll_sample.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(item["text"] for item in gold) + "\n")

    # Mục 25-26: raw text -> NER -> structured
    entities = []
    for sid, item in enumerate(gold):
        entities.extend(extract_from_sentence(item["text"], tagger, sid))
    structured = to_structured(entities)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"sentences": [i["text"] for i in gold], "entities": entities, "structured": structured},
                  f, ensure_ascii=False, indent=2)
    print(f"[extract] {len(entities)} entity -> {args.out}, conll_sample.txt, {args.gold}", file=sys.stderr)

    # Mục 27: entity-level evaluation -> results.csv, ie_errors.csv
    evaluate(gold, tagger)
    return 0


# CLI
def run_extraction(text, tagger, out_path="ie_output.json"):
    entities, sentences = extract_from_text(text, tagger)
    structured = to_structured(entities)
    print("\nRaw text:\n  " + text.strip().replace("\n", "\n  "))
    print("\nStructured output:")
    print(json.dumps(structured, ensure_ascii=False, indent=2))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"text": text, "sentences": sentences, "entities": entities, "structured": structured},
                  f, ensure_ascii=False, indent=2)
    print(f"\nĐã ghi {out_path}")
    return structured


def main(argv=None):
    ap = argparse.ArgumentParser(description="LAB 09 - Information Extraction từ NER (Option A: News)")
    ap.add_argument("--text", help="văn bản đầu vào")
    ap.add_argument("--file", help="file .txt đầu vào")
    ap.add_argument("--demo", action="store_true", help="chạy trên đoạn văn bản demo")
    ap.add_argument("--evaluate", action="store_true", help="đánh giá entity-level trên tập gold (mục 27)")
    ap.add_argument("--gold", default="ie_gold.json", help="file gold annotation (JSON)")
    ap.add_argument("--sentences", help="file .txt (mỗi dòng một câu) để tạo template gold thay cho 30 câu mẫu")
    ap.add_argument("--conll", nargs="?", const=DEFAULT_CONLL,
                    help="đọc CoNLL-2003 parquet, chạy gộp mục 25-27 (mặc định " + DEFAULT_CONLL + ")")
    ap.add_argument("--n", type=int, default=30, help="số câu lấy từ CoNLL (mặc định 30)")
    ap.add_argument("--seed", type=int, default=42, help="seed chọn câu ngẫu nhiên từ CoNLL")
    ap.add_argument("--model", help="đường dẫn / tên model NER")
    ap.add_argument("--out", default="ie_output.json", help="file JSON kết quả trích xuất")
    args = ap.parse_args(argv)

    if args.conll:
        return run_conll(args, NERTagger(args.model))

    if args.evaluate and not os.path.exists(args.gold) and not GOLD_ANNOTATIONS:
        sents = SAMPLE_SENTENCES
        if args.sentences:
            with open(args.sentences, encoding="utf-8") as f:
                sents = [ln.strip() for ln in f if ln.strip()]
        make_gold_template(sents, args.gold)
        return 0

    if not (args.text or args.file or args.demo or args.evaluate):
        if os.path.exists(DEFAULT_CONLL):          # không tham số: chạy thẳng trên CoNLL-2003 local
            args.conll = DEFAULT_CONLL
            return run_conll(args, NERTagger(args.model))
        ap.print_help()
        return 1

    tagger = NERTagger(args.model)

    if args.evaluate:
        gold = load_gold(args.gold) if os.path.exists(args.gold) else gold_from_dict(GOLD_ANNOTATIONS)
        if len(gold) < 30:
            print(f"[cảnh báo] Đề yêu cầu khoảng 30 câu; hiện có {len(gold)} câu đã annotate.")
        if gold:
            evaluate(gold, tagger)
        return 0

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        text = DEMO_TEXT
    run_extraction(text, tagger, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
