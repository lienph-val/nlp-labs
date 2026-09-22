import gzip
import json
import os
import csv

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
filepath = os.path.join(
    BASE_DIR,
    "..",
    "data",
    "c4-train.00000-of-01024-30K.json.gz"
)

documents = []

with gzip.open(filepath, "rt", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)

        if "text" in data:
            documents.append(data["text"])

print("Number of documents:", len(documents))

vectorizer = TfidfVectorizer()

X_tfidf = vectorizer.fit_transform(documents)

print("TF-IDF matrix shape:", X_tfidf.shape)


def search(query, top_k=5):
    #Tìm kiếm các document phù hợp nhất với query.

    # Chuyển query thành TF-IDF vector
    query_vector = vectorizer.transform([query])

    # Tính cosine similarity giữa query và toàn bộ documents
    similarities = cosine_similarity(
        query_vector,
        X_tfidf
    ).flatten()

    # Sắp xếp similarity giảm dần
    ranked_indices = np.argsort(similarities)[::-1]

    # Lấy Top-K
    top_indices = ranked_indices[:top_k]

    results = []

    for rank, index in enumerate(top_indices, start=1):

        results.append({
            "rank": rank,
            "document_id": int(index),
            "similarity": float(similarities[index]),
            "document": documents[index]
        })

    return results

def display_results(query, results):
    print("QUERY:", query)

    for result in results:

        print(
            f"\nRank: {result['rank']}"
        )

        print(
            f"Document ID: {result['document_id']}"
        )

        print(
            f"Similarity: {result['similarity']:.4f}"
        )

        print(
            "Document preview:",
            result["document"][:300].replace("\n", " ")
        )


# queries = [
#     "medical image classification",
#     "transformer language model",
#     "deep learning healthcare",
#     "natural language processing"
# ]

queries = [
    "medical image classification",
    "transformer language model",
    "deep learning healthcare",
    "natural language processing",
    "machine learning classification",
    "image processing"
]

all_results = []

for query in queries:

    results = search(query, top_k=5)

    display_results(query, results)

    # Lưu kết quả để tạo CSV
    for result in results:

        all_results.append({
            "query": query,
            "rank": result["rank"],
            "document_id": result["document_id"],
            "similarity": result["similarity"],
            "document": result["document"]
        })


output_path = os.path.join(
    BASE_DIR,
    "results.csv"
)

with open(
    output_path,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "query",
            "rank",
            "document_id",
            "similarity",
            "document"
        ]
    )

    writer.writeheader()

    writer.writerows(all_results)
