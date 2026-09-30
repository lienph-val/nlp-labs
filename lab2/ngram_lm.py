import re
import gzip
import json
import os
import csv
import random
from collections import Counter

import math


class NGramLanguageModel:

    def __init__(self, n):
        self.n = n

        self.vocabulary = set()
        self.vocab_size = 0

        self.unigram_counts = Counter()
        self.bigram_counts = Counter()
        self.trigram_counts = Counter()

        self.unigram_probs = {}
        self.bigram_probs = {}
        self.trigram_probs = {}

        # Tổng số token trong tập train, dùng cho Laplace unigram
        self.total_tokens = 0

    def build_vocabulary(self, corpus):
        vocabulary = set()
        for text in corpus:
            tokens = re.findall(r'\b\w+\b', text.lower())

            for token in tokens:
                vocabulary.add(token)

        self.vocabulary = vocabulary
        self.vocab_size = len(vocabulary)

    def count_ngrams(self, corpus):
        self.unigram_counts = Counter()
        self.bigram_counts = Counter()
        self.trigram_counts = Counter()

        for text in corpus:

            tokens = re.findall(r'\b\w+\b', text.lower())

            # Unigram
            for token in tokens:
                self.unigram_counts[token] += 1

            # Bigram
            for i in range(len(tokens) - 1):
                bigram = (tokens[i], tokens[i + 1])
                self.bigram_counts[bigram] += 1

            # Trigram
            for i in range(len(tokens) - 2):
                trigram = (tokens[i], tokens[i + 1], tokens[i + 2])
                self.trigram_counts[trigram] += 1

    def train_unigram(self):
        total_tokens = sum(self.unigram_counts.values())

        self.unigram_probs = {}

        for word, count in self.unigram_counts.items():
            self.unigram_probs[word] = count / total_tokens

    def train_bigram(self):
        self.bigram_probs = {}

        for (word1, word2), count in self.bigram_counts.items():

            denominator = self.unigram_counts[word1]

            self.bigram_probs[(word1, word2)] = (count / denominator)

    def train_trigram(self):
        self.trigram_probs = {}

        for (word1, word2, word3), count in self.trigram_counts.items():

            denominator = self.bigram_counts[(word1, word2)]

            self.trigram_probs[(word1, word2, word3)] = (count / denominator)

    def probability(self, tokens):
        k = len(tokens)
        if k == 1:
            return self.unigram_probs.get(tokens[0], 0)
        elif k == 2:
            return self.bigram_probs.get((tokens[0], tokens[1]), 0)
        elif k == 3:
            return self.trigram_probs.get((tokens[0], tokens[1], tokens[2]), 0)
        return 0

    def sentence_probability(self, sentence):
        tokens = re.findall(r'\b\w+\b', sentence.lower())
        if len(tokens) == 0:
            return 0

        probability = 1.0

        if self.n == 1:
            for token in tokens:
                probability *= self.probability([token])
        elif self.n == 2:
            probability *= self.probability([tokens[0]])

            for i in range(1, len(tokens)):
                probability *= self.probability([tokens[i - 1], tokens[i]])

        elif self.n == 3:
            probability *= self.probability([tokens[0]])

            if len(tokens) >= 2:
                probability *= self.probability([tokens[0], tokens[1]])

            for i in range(2, len(tokens)):
                probability *= self.probability([tokens[i - 2], tokens[i - 1], tokens[i]])

        return probability

    def sentence_log_probability(self, sentence):
        tokens = re.findall(r'\b\w+\b', sentence.lower())
        if len(tokens) == 0:
            #return 0.0 => sai 
            return float('-inf') # Log của xác suất 0 là âm vô cực
        
        log_prob = 0.0

        if self.n == 1:
            for token in tokens:
                p = self.probability([token])
                if p == 0:
                    return float('-inf') # Xử lý lỗi math domain error khi tính log(0)
                log_prob += math.log(p)
                
        elif self.n == 2:
            p = self.probability([tokens[0]])
            if p == 0:
                return float('-inf')
            log_prob += math.log(p)

            for i in range(1, len(tokens)):
                p = self.probability([tokens[i - 1], tokens[i]])
                if p == 0:
                    return float('-inf')
                log_prob += math.log(p)

        elif self.n == 3:
            p = self.probability([tokens[0]])
            if p == 0:
                return float('-inf')
            log_prob += math.log(p)

            if len(tokens) >= 2:
                p = self.probability([tokens[0], tokens[1]])
                if p == 0:
                    return float('-inf')
                log_prob += math.log(p)

            for i in range(2, len(tokens)):
                p = self.probability([tokens[i - 2], tokens[i - 1], tokens[i]])
                if p == 0:
                    return float('-inf')
                log_prob += math.log(p)

        return log_prob

    # 16

    def set_n(self, n):
        self.n = n

    def train_laplace(self):
        # Lưu thêm tổng số token N cho công thức unigram.
        self.total_tokens = sum(self.unigram_counts.values())

    def probability_laplace(self, tokens):
        # Công thức: P_Laplace(w | h) = (C(h, w) + 1) / (C(h) + V)
        k = len(tokens)
        V = self.vocab_size

        if k == 1:
            # Unigram: (C(w) + 1) / (N + V)
            count_w = self.unigram_counts.get(tokens[0], 0)
            return (count_w + 1) / (self.total_tokens + V)

        elif k == 2:
            # Bigram: (C(w1, w2) + 1) / (C(w1) + V)
            count_bigram = self.bigram_counts.get((tokens[0], tokens[1]), 0)
            count_context = self.unigram_counts.get(tokens[0], 0)
            return (count_bigram + 1) / (count_context + V)

        elif k == 3:
            # Trigram: (C(w1, w2, w3) + 1) / (C(w1, w2) + V)
            count_trigram = self.trigram_counts.get((tokens[0], tokens[1], tokens[2]), 0)
            count_context = self.bigram_counts.get((tokens[0], tokens[1]), 0)
            return (count_trigram + 1) / (count_context + V)

        return 0

    def corpus_log_probability(self, corpus, smoothing="mle"):
        # smoothing = "mle" => dùng self.probability()
        # smoothing = "laplace" => dùng self.probability_laplace()
        total_log_prob = 0.0
        total_tokens = 0
        zero_count = 0

        for text in corpus:
            tokens = re.findall(r'\b\w+\b', text.lower())

            for i in range(len(tokens)):
                order = min(self.n, i + 1)
                ngram = tokens[i - order + 1: i + 1]

                if smoothing == "laplace":
                    p = self.probability_laplace(ngram)
                else:
                    p = self.probability(ngram)

                total_tokens += 1

                if p == 0:
                    # log(0) không tính được => đếm lại rồi bỏ qua, cuối cùng sẽ gán tổng log prob = -inf
                    zero_count += 1
                else:
                    total_log_prob += math.log(p)

        if zero_count > 0:
            total_log_prob = float('-inf')

        return total_log_prob, total_tokens, zero_count

    def perplexity(self, corpus, smoothing="mle"):
        # PP(W) = exp( -(1/N) * sum( log P(w_i | context_i) ) )
        # Output: (perplexity, số lần xác suất = 0)
        log_prob, N, zero_count = self.corpus_log_probability(corpus, smoothing)

        if N == 0:
            return float('nan'), 0

        if log_prob == float('-inf'):
            # Có ít nhất một xác suất bằng 0 => perplexity là vô cực
            return float('inf'), zero_count

        return math.exp(-log_prob / N), zero_count

NUM_DOCUMENTS = 10000
# DATA_PATH = os.path.join("..", "data", "c4-train.00000-of-01024-30K.json.gz")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "..", "data", "c4-train.00000-of-01024-30K.json.gz")
 

def load_documents(filepath=DATA_PATH, num_documents=NUM_DOCUMENTS):
    documents = []

    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if 'text' in data:
                documents.append(data)

            if num_documents is not None and len(documents) >= num_documents:
                break

    print("Number of documents: ", len(documents))
    return documents


def split_corpus(documents, train_ratio=0.8, valid_ratio=0.1, seed=42):
    # Mặc định 80:10:10
    texts = [doc['text'] for doc in documents]

    random.seed(seed)      
    random.shuffle(texts)

    n_train = int(len(texts) * train_ratio)
    n_valid = int(len(texts) * valid_ratio)

    train = texts[:n_train]
    valid = texts[n_train:n_train + n_valid]
    test = texts[n_train + n_valid:]

    print("Train / Valid / Test documents:", len(train), len(valid), len(test))
    return train, valid, test


def train_model(train):
    model = NGramLanguageModel(3)

    model.build_vocabulary(train)
    model.count_ngrams(train)
    model.train_unigram()
    model.train_bigram()
    model.train_trigram()
    model.train_laplace()

    print("Vocabulary size:", model.vocab_size)
    print("Unique unigrams:", len(model.unigram_counts))
    print("Unique bigrams :", len(model.bigram_counts))
    print("Unique trigrams:", len(model.trigram_counts))
    return model


def evaluate_config(model, name, n, smoothing, train, valid, test):
    print(name)
    model.set_n(n)

    train_ppl, train_zero = model.perplexity(train, smoothing)
    valid_ppl, valid_zero = model.perplexity(valid, smoothing)
    test_ppl, test_zero = model.perplexity(test, smoothing)

    return {
        "model": name,
        "train_ppl": train_ppl,
        "valid_ppl": valid_ppl,
        "test_ppl": test_ppl,
        "train_zero": train_zero,
        "valid_zero": valid_zero,
        "test_zero": test_zero,
    }


def print_results(results):
    # In bảng perplexity
    print()
    print("%-18s %14s %14s %14s" % ("Model", "Train PPL", "Valid PPL", "Test PPL"))
    for row in results:
        print("%-18s %14.2f %14.2f %14.2f" % (
            row["model"], row["train_ppl"], row["valid_ppl"], row["test_ppl"]))

    # In số lần gặp xác suất = 0 
    print("%-18s %14s %14s %14s" % ("Model", "Train zeros", "Valid zeros", "Test zeros"))
    for row in results:
        print("%-18s %14d %14d %14d" % (
            row["model"], row["train_zero"], row["valid_zero"], row["test_zero"]))


def run_experiment_2(model, train, valid, test):
    configs = [
        ("Bigram MLE",      2, "mle"),
        ("Bigram Laplace",  2, "laplace"),
        ("Trigram MLE",     3, "mle"),
        ("Trigram Laplace", 3, "laplace"),
    ]

    results = []

    for name, n, smoothing in configs:
        row = evaluate_config(model, name, n, smoothing, train, valid, test)
        results.append(row)

    print_results(results)

    return results

RESULTS_PATH = os.path.join(BASE_DIR, "results.csv")
def save_results(results):
    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "train_ppl", "valid_ppl", "test_ppl",
                         "train_zero", "valid_zero", "test_zero"])
        for row in results:
            writer.writerow([row["model"], row["train_ppl"], row["valid_ppl"], row["test_ppl"],
                             row["train_zero"], row["valid_zero"], row["test_zero"]])

# 19
def run_experiment_3(model, train, valid, test, known_results=None):
    configs = [
        ("Unigram MLE",     1, "mle"),
        ("Unigram Laplace", 1, "laplace"),
        ("Bigram MLE",      2, "mle"),
        ("Bigram Laplace",  2, "laplace"),
        ("Trigram MLE",     3, "mle"),
        ("Trigram Laplace", 3, "laplace"),
    ]

    known = {}
    if known_results is not None:
        for row in known_results:
            known[row["model"]] = row

    results = []

    for name, n, smoothing in configs:
        if name in known:
            results.append(known[name])
        else:
            row = evaluate_config(model, name, n, smoothing, train, valid, test)
            results.append(row)

    print_results(results)
    print()
    print("Số token train:", model.total_tokens)
    print("Vocabulary size:", model.vocab_size)
    print("Số unigram khác nhau:", len(model.unigram_counts))
    print("Số bigram khác nhau :", len(model.bigram_counts))
    print("Số trigram khác nhau:", len(model.trigram_counts))

    return results

# def main():
#     corpus = [
#         "the cat eats fish",
#         "the cat likes fish"
#     ]

#     model = NGramLanguageModel(2)

#     model.build_vocabulary(corpus)
#     model.count_ngrams(corpus)
#     model.train_unigram()
#     model.train_bigram()

#     print("Vocabulary:", model.vocabulary)
#     print("Unigram counts:", model.unigram_counts)
#     print("Bigram counts:", model.bigram_counts)
#     print("Bigram probabilities:", model.bigram_probs)

#     sentence = "the cat eats fish"
#     print("Sentence probability:",
#           model.sentence_probability(sentence))


# if __name__ == "__main__":
#     main()

def main():
    documents = load_documents()
    train, valid, test = split_corpus(documents)

    model = train_model(train)

    results_16 = run_experiment_2(model, train, valid, test)

    results_19 = run_experiment_3(model, train, valid, test, known_results=results_16)

    save_results(results_19)


if __name__ == "__main__":
    main()