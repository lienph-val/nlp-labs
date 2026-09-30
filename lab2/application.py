import csv
import math
import os
import re

# Import 
from ngram_lm import (NGramLanguageModel, load_documents, split_corpus, train_model, BASE_DIR)

RESULTS_PATH = os.path.join(BASE_DIR, "results.csv")


def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def word_probability(model, history, word, n, smoothing):
    """Tính P(word | history). Chỉ lấy tối đa (n-1) từ cuối của history."""
    order = min(n, len(history) + 1)   # history ngắn thì dùng n nhỏ hơn
    context = history[len(history) - (order - 1):] if order > 1 else []
    ngram = context + [word]

    if smoothing == "laplace":
        return model.probability_laplace(ngram)
    return model.probability(ngram)  # MLE


# MỤC 20: Next-word prediction
def predict_next_words(model, context_text, n=2, smoothing="laplace", top_k=5):
    """Trả về top_k từ có xác suất cao nhất sau context: [(từ, xác suất), ...]."""
    history = tokenize(context_text)

    scores = []
    for word in model.vocabulary:            
        p = word_probability(model, history, word, n, smoothing)
        scores.append((word, p))

    scores.sort(key=lambda item: item[1], reverse=True)
    return scores[:top_k]


def find_actual_next_word(context_text, documents):
    """Tìm từ thật sự đứng sau context trong dữ liệu (dùng tập test)."""
    context = tokenize(context_text)
    for text in documents:
        tokens = tokenize(text)
        for i in range(len(tokens) - len(context)):
            if tokens[i:i + len(context)] == context:
                return tokens[i + len(context)]
    return "(không tìm thấy)"


def run_next_word_prediction(model, contexts, test_docs, n=2, smoothing="laplace"):
    print(f"NEXT-WORD PREDICTION (n={n}, {smoothing})")
    rows = []
    for context in contexts:
        predictions = predict_next_words(model, context, n, smoothing)
        actual = find_actual_next_word(context, test_docs)

        print(f"\nInput: '{context}'")
        for rank, (word, p) in enumerate(predictions, start=1):
            print(f"  {rank}. {word:<15} {p:.4f}")

        best_word, best_p = predictions[0]
        rows.append([context, best_word, actual, best_p])

    print("\nBảng tổng hợp:")
    print(f"{'Context':<22}{'Prediction':<16}{'Actual next word'}")
    for context, best_word, actual, _ in rows:
        print(f"{context:<22}{best_word:<16}{actual}")
    return rows

# MỤC 21: Sentence ranking
def score_candidate(model, context_text, candidate_text, n=2, smoothing="laplace"):
    """log P(candidate | context) = tổng log P(từ | các từ đứng trước nó)."""
    history = tokenize(context_text)
    total_log_prob = 0.0

    for word in tokenize(candidate_text):
        p = word_probability(model, history, word, n, smoothing)
        if p == 0:  # có thể xảy ra với MLE
            return float("-inf")
        total_log_prob += math.log(p)
        history.append(word)  # từ vừa xét trở thành context

    return total_log_prob


def run_sentence_ranking(model, context_text, candidates, n=2, smoothing="laplace"):
    print(f"SENTENCE RANKING (n={n}, {smoothing})")
    print(f"Context: '{context_text}'\n")

    scored = []
    for name, text in candidates.items():
        log_prob = score_candidate(model, context_text, text, n, smoothing)
        per_word = log_prob / len(tokenize(text))  # chuẩn hóa theo độ dài
        scored.append((name, text, log_prob, per_word))

    scored.sort(key=lambda item: item[2], reverse=True) # log P cao nhất = hạng 1

    print(f"{'Hạng':<6}{'Candidate':<30}{'log P':<12}{'log P / từ'}")
    for rank, (name, text, log_prob, per_word) in enumerate(scored, start=1):
        print(f"{rank:<6}{name + '. ' + text:<30}{log_prob:<12.2f}{per_word:.2f}")
    return scored

def append_application_results(next_word_rows, ranking):
    with open(RESULTS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        writer.writerow([])
        writer.writerow(["20: NEXT-WORD PREDICTION"])
        writer.writerow(["Context", "Prediction", "Actual", "Probability"])
        for context, best_word, actual, p in next_word_rows:
            writer.writerow([context, best_word, actual, f"{p:.4f}"])
            
        writer.writerow([])
        writer.writerow(["21: SENTENCE RANKING"])
        writer.writerow(["Context", "Candidate", "Log Probability", "Log P / Word"])
        for rank, (name, text, log_prob, per_word) in enumerate(ranking, start=1):
            writer.writerow(["machine learning", text, f"{log_prob:.2f}", f"{per_word:.2f}"])


def main():
    documents = load_documents()
    train, valid, test = split_corpus(documents)
    model = train_model(train)

    N = 2 # bigram                
    SMOOTHING = "laplace" # smoothing

    # 5 contexts
    contexts = ["the cat", "natural language", "machine learning",
                "in the", "one of"]
    rows = run_next_word_prediction(model, contexts, test, N, SMOOTHING)

    candidates = {
        "A": "is useful for nlp",
        "B": "banana computer quickly",
        "C": "studies language models",
    }
    ranking = run_sentence_ranking(model, "machine learning", candidates, N, SMOOTHING)

    append_application_results(rows, ranking)

if __name__ == "__main__":
    main()