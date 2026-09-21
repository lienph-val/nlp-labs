import math

def build_vocabulary(documents):
    vocabulary = set()

    for document in documents:
        tokens = document.lower().split() #chuyển chữ thường, tách theo khoảng trắng
        for token in tokens:
            vocabulary.add(token)

    return sorted(vocabulary)

def compute_counts(documents, vocabulary):
    count_vectors = []

    for document in documents:
        tokens = document.lower().split()
        vector = []
        for term in vocabulary:
            count = tokens.count(term) #đếm số lần mỗi term xuất hiện trong document
            vector.append(count)
        count_vectors.append(vector)

    return count_vectors

def compute_tf(count_vector): 
    total_terms = sum(count_vector)

    if total_terms == 0:
        return [0.0] * len(count_vector)
    
    tf_vector = []
    
    for count in count_vector:
        tf = count / total_terms #count / tổng số từ
        tf_vector.append(tf)

    return tf_vector

def compute_idf(count_vectors):
    N = len(count_vectors)
    vocabulary_size = len(count_vectors[0])
    idf_vector = []

    for j in range(vocabulary_size):
        df = 0
        for vector in count_vectors:
            if vector[j] > 0:
                df += 1
        idf = math.log(N / df) #khác với sklearn 
        idf_vector.append(idf)

    return idf_vector

def compute_tfidf(tf_vector, idf_vector):
    tfidf_vector = []
    for tf, idf in zip(tf_vector, idf_vector):
        tfidf = tf * idf
        tfidf_vector.append(tfidf)
    return tfidf_vector

def cosine_similarity(x, y):
    if len(x) != len(y):
        raise ValueError(
            "Độ dài vecto phải bằng nhau"
        )

    dot_product = 0

    for a, b in zip(x, y):
        dot_product += a * b

    # Chuẩn x
    norm_x = math.sqrt(
        sum(a * a for a in x)
    )

    # Chuẩn y
    norm_y = math.sqrt(
        sum(b * b for b in y)
    )

    if norm_x == 0 or norm_y == 0:
        return 0.0 # Tránh chia cho 0

    return dot_product / (norm_x * norm_y)