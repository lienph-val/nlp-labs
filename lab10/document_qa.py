"""
LAB 10 - Mục 27: Application - Document QA
"""

import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))     
PROJECT_DIR = os.path.dirname(CURRENT_DIR)                    
SQUAD_DIR = os.path.join(PROJECT_DIR, "data", "squad")        
DEV_PATH = os.path.join(SQUAD_DIR, "dev-v2.0.json")

MODEL_NAME = "distilbert-base-cased-distilled-squad"
MAX_LENGTH = 384
STRIDE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MiniDocumentQA:
    def __init__(self, model_name=MODEL_NAME, max_length=MAX_LENGTH, stride=STRIDE):
        print(f"Đang tải QA model '{model_name}' trên thiết bị: {DEVICE}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(DEVICE)
        self.model.eval()
        self.max_length = max_length
        self.stride = stride

    def answer(self, question: str, context: str, n_best_size: int = 20, max_answer_length: int = 40):
        """
        Nhận (question, context), xử lý sliding window nếu document dài,
        chạy qua QA model và trả về answer span tốt nhất.
        """
        inputs = self.tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt"
        )

        offset_mapping = inputs.pop("offset_mapping").tolist()
        _ = inputs.pop("overflow_to_sample_mapping")

        # Lưu sequence_ids cho từng window trước khi đưa tensor lên device
        seq_ids_per_window = [inputs.sequence_ids(i) for i in range(inputs["input_ids"].size(0))]

        model_inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**model_inputs)
            all_start_logits = outputs.start_logits.cpu().numpy()
            all_end_logits = outputs.end_logits.cpu().numpy()

        candidates = []
        num_windows = all_start_logits.shape[0]

        for win_idx in range(num_windows):
            start_logits = all_start_logits[win_idx]
            end_logits = all_end_logits[win_idx]
            offsets = offset_mapping[win_idx]
            seq_ids = seq_ids_per_window[win_idx]

            # Tính xác suất softmax để ước lượng độ tự tin (confidence)
            start_probs = np.exp(start_logits - np.max(start_logits))
            start_probs /= start_probs.sum()
            end_probs = np.exp(end_logits - np.max(end_logits))
            end_probs /= end_probs.sum()

            top_starts = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            top_ends = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()

            for s_idx in top_starts:
                for e_idx in top_ends:
                    # Chỉ xét các token thuộc phần context (sequence_id == 1)
                    if seq_ids[s_idx] != 1 or seq_ids[e_idx] != 1:
                        continue
                    if e_idx < s_idx or (e_idx - s_idx + 1) > max_answer_length:
                        continue

                    char_start = offsets[s_idx][0]
                    char_end = offsets[e_idx][1]
                    if char_end <= char_start:
                        continue

                    candidates.append({
                        "answer": context[char_start:char_end],
                        "start_char": char_start,
                        "end_char": char_end,
                        "score": float(start_logits[s_idx] + end_logits[e_idx]),
                        "confidence": float(start_probs[s_idx] * end_probs[e_idx]),
                        "window": win_idx
                    })

        if not candidates:
            return {
                "answer": "",
                "start_char": -1,
                "end_char": -1,
                "score": 0.0,
                "confidence": 0.0,
                "window": 0
            }

        best = max(candidates, key=lambda x: x["score"])
        return best


def load_sample_documents(squad_path, num_docs=3):
    """Lấy một vài đoạn văn bản (paragraphs) và câu hỏi mẫu từ data/squad/dev-v2.0.json."""
    if not os.path.exists(squad_path):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {squad_path}")

    with open(squad_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    docs = []
    for article in raw["data"]:
        title = article.get("title", "Untitled")
        for para in article["paragraphs"]:
            answerable_qas = [qa for qa in para["qas"] if len(qa["answers"]) > 0]
            if answerable_qas:
                docs.append({
                    "title": title,
                    "context": para["context"],
                    "sample_question": answerable_qas[0]["question"],
                    "gold_answer": answerable_qas[0]["answers"][0]["text"]
                })
            if len(docs) >= num_docs:
                return docs
    return docs

def main():
    qa_system = MiniDocumentQA()

    # PHẦN A: Demo trên Document từ data/squad/dev-v2.0.json
    print("PHẦN 1: KIỂM THỬ DOCUMENT QA TRÊN DỮ LIỆU LOCAL (dev-v2.0.json)")
    docs = load_sample_documents(DEV_PATH, num_docs=2)
    for idx, doc in enumerate(docs, 1):
        res = qa_system.answer(doc["sample_question"], doc["context"])
        print(f"\n[Document {idx} - Chủ đề: {doc['title']}]")
        print(f"Context  : {doc['context'][:180]}...")
        print(f"Question : {doc['sample_question']}")
        print(f"Gold     : {doc['gold_answer']}")
        print(f"Predicted: '{res['answer']}' (char [{res['start_char']}:{res['end_char']}], conf={res['confidence']:.3f})")

    # PHẦN B: Demo trên một tài liệu Wikipedia / Học thuật tùy chỉnh (Mục 27)
    print("PHẦN 2: KIỂM THỬ TRÊN TÀI LIỆU WIKIPEDIA / NLP DOMAIN")
    wiki_context = (
        "Extractive Question Answering is the task of identifying a contiguous span of text "
        "within a given context passage that directly answers a user's question. Unlike generative QA, "
        "extractive models such as BERT or DistilBERT predict two probability distributions over the "
        "input tokens: one for the start position and one for the end position of the answer span. "
        "This architecture was popularized by the Stanford Question Answering Dataset (SQuAD), "
        "released by Pranav Rajpurkar and colleagues in 2016."
    )
    custom_questions = [
        "What do extractive models predict over the input tokens?",
        "Who released the Stanford Question Answering Dataset?",
        "When was SQuAD released?"
    ]
    for q in custom_questions:
        res = qa_system.answer(q, wiki_context)
        print(f"Q: {q}")
        print(f"-> Answer span: '{res['answer']}' (span=({res['start_char']}, {res['end_char']}), conf={res['confidence']:.3f})\n")

    # PHẦN C: Chế độ nhập tương tác từ người dùng 
    print("PHẦN 3: CHẾ ĐỘ TƯƠNG TÁC NGƯỜI DÙNG (Gõ 'exit' để thoát)")
    print("Nhập context/document của bạn (hoặc bấm Enter để dùng lại đoạn văn NLP ở Phần 2):")
    user_context = input(">> Context: ").strip()
    if not user_context:
        user_context = wiki_context
        print("(Đang sử dụng đoạn văn NLP ở Phần 2 làm context)")

    while True:
        user_q = input("\n>> Nhập câu hỏi (hoặc 'exit'): ").strip()
        if user_q.lower() in {"exit", "quit", "q"}:
            print("Kết thúc chương trình Document QA.")
            break
        if not user_q:
            continue

        result = qa_system.answer(user_q, user_context)
        print(f"   [Answer Span] : '{result['answer']}'")
        print(f"   [Vị trí Char] : ({result['start_char']}, {result['end_char']}) | Confidence: {result['confidence']:.3f}")


if __name__ == "__main__":
    main()