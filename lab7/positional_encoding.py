import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

np.random.seed(42)
D_MODEL = 8 # số chiều của embedding (chẵn để chia đủ cặp sin/cos)

# 11: Embedding matrix + Positional Encoding
def positional_encoding(seq_len, d_model):
    """PE(pos, 2i) = sin(pos / 10000^(2i/d)), PE(pos, 2i+1) = cos(...)"""
    pe = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for k in range(0, d_model, 2):  # k = 2i
            angle = pos / (10000 ** (k / d_model))
            pe[pos, k] = np.sin(angle)
            pe[pos, k + 1] = np.cos(angle)
    return pe


# Mỗi từ trong vocab có đúng 1 vector embedding (ngẫu nhiên, cố định)
vocab = ["I", "love", "NLP", "dog", "bites", "man"]
word2id = {w: i for i, w in enumerate(vocab)}
embedding_matrix = np.random.randn(len(vocab), D_MODEL)


def embed(words):
    """Danh sách từ -> ma trận embedding (T, d_model)"""
    return embedding_matrix[[word2id[w] for w in words]]


def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)  # trừ max để ổn định số học
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


# Ma trận chiếu Q, K, V (dùng chung cho mọi thí nghiệm)
Wq = np.random.randn(D_MODEL, D_MODEL)
Wk = np.random.randn(D_MODEL, D_MODEL)
Wv = np.random.randn(D_MODEL, D_MODEL)


def self_attention(X):
    """Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V"""
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    scores = Q @ K.T / np.sqrt(D_MODEL)
    weights = softmax(scores)
    return weights @ V


def make_input(words, use_pe):
    """Input của Transformer: embedding (+ positional encoding nếu use_pe)"""
    X = embed(words)
    if use_pe:
        X = X + positional_encoding(len(words), D_MODEL)
    return X


def print_experiment(title, words_a, words_b):
    """So sánh representation của 2 câu, có và không có positional encoding"""
    print(f"\n{title}: {words_a} vs {words_b}")
    for use_pe in [False, True]:
        Za = self_attention(make_input(words_a, use_pe))
        Zb = self_attention(make_input(words_b, use_pe))

        # (1) Representation của từng từ (cùng một từ ở 2 câu)
        print(f"\n[{'Có' if use_pe else 'Không'} positional encoding]")
        for w in words_a:
            va = Za[words_a.index(w)]
            vb = Zb[words_b.index(w)]
            print(f"  '{w}': khoảng cách giữa 2 câu = {np.linalg.norm(va - vb):.4f}")

        # (2) Representation của cả câu (mean pooling)
        sa, sb = Za.mean(axis=0), Zb.mean(axis=0)
        dist = np.linalg.norm(sa - sb)
        same = np.allclose(sa, sb)
        print(f"  Cả câu (mean pooling): khoảng cách = {dist:.4f} "
              f" {'Giống nhau (không phân biệt được)' if same else 'Khác nhau (phân biệt được)'}")


sentence = ["I", "love", "NLP"]
E = embed(sentence)
PE = positional_encoding(len(sentence), D_MODEL)

np.set_printoptions(precision=3, suppress=True)
print("Embedding (3 x 8):\n", E)
print("\nPositional encoding (3 x 8):\n", PE)
print("\nEmbedding + PE (input của Transformer):\n", E + PE)

# Đổi thứ tự "I love NLP" -> "NLP love I"
print_experiment("Mục 11", ["I", "love", "NLP"], ["NLP", "love", "I"])

# 12: Permutation experiment
print_experiment("Mục 12", ["dog", "bites", "man"], ["man", "bites", "dog"])

pe_plot = positional_encoding(50, D_MODEL)
plt.figure(figsize=(7, 4))
plt.imshow(pe_plot, cmap="RdBu", aspect="auto")
plt.colorbar()
plt.xlabel("Chiều embedding")
plt.ylabel("Vị trí (pos)")
plt.title("Positional Encoding (sin/cos)")
plt.tight_layout()
plt.savefig(BASE_DIR / "positional_encoding.png", dpi=150)
print("\nĐã lưu positional_encoding.png")
