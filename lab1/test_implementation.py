import math

from implementation import (
    build_vocabulary,
    compute_counts,
    compute_tf,
    compute_idf,
    compute_tfidf,
    cosine_similarity
)

documents = [
    "cat eats fish",
    "dog eats fish",
    "cat likes fish"
]

#1. Test build vocabulary

vocabulary = build_vocabulary(documents)

expected_vocabulary = [
    "cat",
    "dog",
    "eats",
    "fish",
    "likes"
]

assert vocabulary == expected_vocabulary

#2. Test compute counts

counts = compute_counts(
    documents,
    vocabulary
)

expected_counts = [
    [1, 0, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [1, 0, 0, 1, 1]
]

assert counts == expected_counts

#3. Test compute tf

tf = compute_tf(counts[0])

assert abs(tf[0] - 1/3) < 1e-9
assert abs(tf[2] - 1/3) < 1e-9
assert abs(tf[3] - 1/3) < 1e-9

#4. Test compute idf

idf = compute_idf(counts)

expected_idf = [
    0.4054651081,
    1.0986122887,
    0.4054651081,
    0.0,
    1.0986122887
]

for actual, expected in zip(idf, expected_idf):
    assert abs(actual - expected) < 1e-9

#5. Test compute tfidf 

tfidf = compute_tfidf(tf, idf)

expected_tfidf = [
    (1/3) * 0.4054651081,
    0,
    (1/3) * 0.4054651081,
    0,
    0
]

for actual, expected in zip(tfidf, expected_tfidf):
    assert abs(actual - expected) < 1e-9

# 6. Test cosine similarity

x = [1, 1, 1]
y = [1, 1, 0]

similarity = cosine_similarity(x, y)

expected_similarity = 2 / (6 ** 0.5)

assert abs(similarity - expected_similarity) < 1e-9

print("All unit tests passed.")