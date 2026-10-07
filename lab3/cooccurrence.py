import re
from collections import Counter
import numpy as np

def tokenize(sentence):
    return re.findall(r"[a-z0-9']+", sentence.lower())

def build_vocabulary(corpus, min_count=1):
    """ Xây dựng vocabulary từ corpus"""
    counts = Counter()

    for sentence in corpus:
        tokens = tokenize(sentence)
        for token in tokens:
            counts[token] += 1

    words = []
    for word, count in counts.items():
        if count >= min_count:
            words.append(word)

    words.sort()

    # gán số thứ tự 
    vocabulary = {}
    for i, word in enumerate(words):
        vocabulary[word] = i

    return vocabulary

def build_cooccurrence_matrix(corpus, vocab, window=1):
    """ 
    Xây dựng ma trận word-context X (|V| x |V|) 
    X[i, j] = số lần từ j xuất hiện trong cửa sổ +-window quanh từ i 
    """
    V = len(vocab)
    X = np.zeros((V, V))
 
    for sent in corpus:
        tokens = tokenize(sent)
        for i, target in enumerate(tokens):
            if target not in vocab:
                continue
            start = max(0, i - window)
            end = min(len(tokens), i + window + 1)
            for j in range(start, end):
                if j == i:
                    continue
                ctx = tokens[j]
                if ctx in vocab:
                    X[vocab[target], vocab[ctx]] += 1
    return X

def cosine_similarity(x, y):
    """
    cos(x, y) = (x . y) / (||x|| * ||y||)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    norm = np.linalg.norm(x) * np.linalg.norm(y)
    if norm == 0:
        return 0.0
    return float(np.dot(x, y) / norm)

def most_similar(word, matrix, vocabulary, top_k=5):
    # return: top_k từ gần `word` nhất theo cosine similarity
 
    if word not in vocabulary:
        raise KeyError(f"'{word}' không có trong vocabulary")
 
    idx_to_word = {i: w for w, i in vocabulary.items()}
    query = matrix[vocabulary[word]]
 
    scores = []
    for i in range(matrix.shape[0]):
        if i == vocabulary[word]:
            continue  # bỏ chính từ truy vấn
        scores.append((idx_to_word[i], cosine_similarity(query, matrix[i])))
 
    # sắp theo score giảm dần/ bằng điểm thì theo alphabet 
    scores.sort(key=lambda t: (-t[1], t[0]))
    return scores[:top_k]

if __name__ == "__main__":
    corpus = [
        "the cat eats fish",
        "the cat likes milk",
        "the dog eats meat",
        "the dog likes fish",
    ]
 
    vocab = build_vocabulary(corpus)
    X = build_cooccurrence_matrix(corpus, vocab, window=1)
 
    words = list(vocab)
    print("Vocabulary:", vocab)
    print("\n{:>6}".format("") + "".join(f"{w:>6}" for w in words))
    for w in words:
        row = X[vocab[w]]
        print(f"{w:>6}" + "".join(f"{int(v):>6}" for v in row))

    print("\ncos(cat, dog)  =", round(cosine_similarity(X[vocab['cat']], X[vocab['dog']]), 4))
    print("cos(cat, fish) =", round(cosine_similarity(X[vocab['cat']], X[vocab['fish']]), 4))
 
    print("\nmost_similar('cat', top_k=5):")
    for w, s in most_similar(word="cat", matrix=X, vocabulary=vocab, top_k=5):
        print(f"  {w:<6} {s:.4f}")
 
    