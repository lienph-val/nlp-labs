"""
LAB 10 - Muc 15: Core Implementation
Tokenize (question, context) cho Extractive QA va kiem tra cac output:
    input_ids, attention_mask, offset_mapping, overflow_to_sample_mapping
"""

import os
import json
from transformers import AutoTokenizer

MODEL_NAME = "distilbert-base-uncased"   # "bert-base-uncased"
MAX_LENGTH = 384 # do dai toi da cua 1 window
STRIDE = 128 # so token overlap giua 2 window lien tiep

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)                 # thu muc nlp/
SQUAD_DIR = os.path.join(PROJECT_DIR, "data", "squad")
TRAIN_PATH = os.path.join(SQUAD_DIR, "train-v2.0.json")
DEV_PATH = os.path.join(SQUAD_DIR, "dev-v2.0.json")

def load_squad(path):
    """
    file SQuAD (json long nhau):
        [{"id", "question", "context", "answer_text", "answer_start"}, ...]

    Voi SQuAD 2.0, cau hoi khong co dap an se co answer_text = "" va answer_start = -1
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    samples = []
    for article in raw["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                if len(qa["answers"]) > 0:
                    answer = qa["answers"][0] # lay dap an dau tien
                    answer_text = answer["text"]
                    answer_start = answer["answer_start"]
                else: # cau hoi khong co dap an
                    answer_text = ""
                    answer_start = -1

                samples.append({
                    "id": qa["id"],
                    "question": qa["question"],
                    "context": context,
                    "answer_text": answer_text,
                    "answer_start": answer_start,
                })
    return samples

def tokenize_qa(tokenizer, questions, contexts):
    """
    Tokenize danh sach cau hoi + context.
    Context dai se duoc cat thanh nhieu window (sliding window).
    """
    inputs = tokenizer(
        questions,
        contexts,
        truncation="only_second", # chi cat context, khong cat question
        max_length=MAX_LENGTH,
        stride=STRIDE,
        return_overflowing_tokens=True,  # context dai -> sinh nhieu window
        return_offsets_mapping=True,  # vi tri (start_char, end_char) cua moi token
    )
    return inputs

def inspect_example(tokenizer, inputs, questions, contexts, window_idx):
    input_ids = inputs["input_ids"][window_idx]
    attention_mask = inputs["attention_mask"][window_idx]
    offsets = inputs["offset_mapping"][window_idx]
    sample_idx = inputs["overflow_to_sample_mapping"][window_idx]
    sequence_ids = inputs.sequence_ids(window_idx)   # 0 = question, 1 = context, None = token dac biet

    print(f"Window {window_idx}  (thuoc ve sample so {sample_idx})")
    print("Question:", questions[sample_idx])
    print("Do dai window (so token):", len(input_ids))

    print("\n input_ids ")
    print("10 id dau:", input_ids[:10])
    print("Decode lai:", tokenizer.decode(input_ids[:10]))

    print("\n attention_mask ")
    print("10 gia tri dau:", attention_mask[:10])
    print("(1 = token that, 0 = padding)")

    if "token_type_ids" in inputs:
        print("\n token_type_ids ")
        print("10 gia tri dau:", inputs["token_type_ids"][window_idx][:10])
    else:
        print("\n(Model nay khong dung token_type_ids)")

    print("\n offset_mapping + sequence_ids ")
    print(f"{'idx':>4} {'token':<15} {'offset':<12} {'seq_id':<6} text goc")
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    context = contexts[sample_idx]
    for i in range(min(25, len(tokens))):
        start_char, end_char = offsets[i]
        original_text = context[start_char:end_char] if sequence_ids[i] == 1 else ""
        print(f"{i:>4} {tokens[i]:<15} {str(offsets[i]):<12} {str(sequence_ids[i]):<6} {original_text}")
    print("(offset cua question van tinh theo question; offset cua context tinh theo context)")


def inspect_overflow(inputs, num_samples):
    """In overflow_to_sample_mapping: window nao thuoc sample nao."""
    mapping = inputs["overflow_to_sample_mapping"]
    print(f"So sample goc      : {num_samples}")
    print(f"So window sau token: {len(inputs['input_ids'])}")
    print("overflow_to_sample_mapping:", mapping)
    print("(vi tri i = window i, gia tri = chi so sample goc cua window do)")

def main():
    if not os.path.exists(TRAIN_PATH):
        print("Khong tim thay file:", TRAIN_PATH)
        return

    print("Dang doc data tu:", TRAIN_PATH)
    train_samples = load_squad(TRAIN_PATH)
    print("So sample trong train:", len(train_samples))

    samples = train_samples[:3]
    questions = [s["question"] for s in samples]
    contexts = [s["context"] for s in samples]

    # Them 1 sample co context dai
    long_sample = max(train_samples[:2000], key=lambda s: len(s["context"]))
    questions.append(long_sample["question"])
    contexts.append(long_sample["context"])

    print("\nLoad tokenizer:", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    inputs = tokenize_qa(tokenizer, questions, contexts)

    # Xem cac key co trong output
    print("\nCac key trong output:", list(inputs.keys()))

    # Giai thich overflow_to_sample_mapping
    inspect_overflow(inputs, num_samples=len(questions))

    # Xem chi tiet window dau tien
    inspect_example(tokenizer, inputs, questions, contexts, window_idx=0)

    # Neu sample dai bi chia thanh nhieu window -> xem window cuoi
    last_window = len(inputs["input_ids"]) - 1
    if last_window > 0:
        inspect_example(tokenizer, inputs, questions, contexts, window_idx=last_window)


if __name__ == "__main__":
    main()
