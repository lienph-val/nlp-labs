import numpy as np


def softmax(x, axis=-1, temperature=1.0):
    """
    softmax(x_i / T) = exp(x_i/T - max) / sum_j exp(x_j/T - max)
    """
    x = np.asarray(x, dtype=float) / temperature
    x = x - np.max(x, axis=axis, keepdims=True) # trừ max theo từng hàng
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def entropy(p):
    """Entropy của một phân phối p"""
    p = np.asarray(p, dtype=float)
    p = p[p > 0] # bỏ p=0 để tránh log(0)
    return -np.sum(p * np.log(p))


def scaled_dot_product_attention(Q, K, V):
    Q, K, V = np.asarray(Q, float), np.asarray(K, float), np.asarray(V, float)
    d_k = K.shape[-1]

    scores = Q @ K.T / np.sqrt(d_k) # bước 1: S
    attention_weights = softmax(scores, axis=-1) # bước 2: A (mỗi hàng tổng = 1)
    context = attention_weights @ V # bước 3: C

    return scores, attention_weights, context


if __name__ == "__main__":
    # Test mục 17
    x = np.array([1000, 1001, 1002])
    print("softmax ổn định:", softmax(x))

    with np.errstate(all="ignore"):
        naive = np.exp(x) / np.sum(np.exp(x))
    print("softmax tính trực tiếp (không trừ max):", naive)

    # Test với Bài 4 (q = [1,0])
    Q = np.array([[1, 0]])
    K = np.array([[1, 0], [0, 1], [1, 1]])
    V = np.array([[1, 0], [0, 2], [2, 1]])
    s, a, c = scaled_dot_product_attention(Q, K, V)
    print("scores :", s)
    print("weights:", a, "| tổng =", a.sum())
    print("context:", c)