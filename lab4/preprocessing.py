import csv
import os
import re

import numpy as np
from gensim.utils import simple_preprocess
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

RESULT_COLUMNS = [
    "experiment", "representation", "classifier", "split",
    "vocab_size", "train_time", "accuracy", "precision", "recall", "f1", "note",
]


#  Làm sạch văn bản = bỏ thẻ HTML, dấu câu, giữ stopwords
def clean_text(text):
    text = re.sub(r"<[^>]+>", " ", str(text))
    return text.lower().strip()


def clean_all(text_list):
    cleaned = []
    for text in text_list:
        cleaned.append(clean_text(text))
    return cleaned

# Tách từ bằng gensim
def tokenize(text):
    return simple_preprocess(text, min_len=1)

#Tách danh sách văn bản thành danh sách từ
def tokenize_all(text_list):
    all_tokens = []
    for text in text_list:
        all_tokens.append(tokenize(text))
    return all_tokens


def encode_labels(label_list):
    """positive -> 1, negative -> 0."""
    encoded = []
    for label in label_list:
        if label == "positive":
            encoded.append(1)
        else:
            encoded.append(0)
    return encoded

# train/validation/test = 80/10/10
def split_data(texts, labels, seed=42):
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=seed, stratify=labels
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=seed, stratify=temp_labels
    )
    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels

# Average Word2Vec
def sentence_to_vector(tokens, w2v_model, dim=100):
    vectors = []
    for word in tokens:
        if word in w2v_model.wv:
            vectors.append(w2v_model.wv[word])
    if len(vectors) == 0:
        return np.zeros(dim)      # câu không có từ nào trong vocab -> vector 0
    return np.mean(vectors, axis=0)


def tokens_to_matrix(list_of_tokens, w2v_model, dim=100):
    """Chuyển nhiều câu thành ma trận (số câu x dim)."""
    vectors = []
    for tokens in list_of_tokens:
        vectors.append(sentence_to_vector(tokens, w2v_model, dim))
    return np.array(vectors)


def compute_metrics(y_true, y_pred):
    metrics = {}
    metrics["accuracy"] = round(accuracy_score(y_true, y_pred), 4)
    metrics["precision"] = round(precision_score(y_true, y_pred), 4)
    metrics["recall"] = round(recall_score(y_true, y_pred), 4)
    metrics["f1"] = round(f1_score(y_true, y_pred), 4)
    return metrics


def reset_results(path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()


def save_row(path, row):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, restval="")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_metrics(path, experiment, representation, classifier, split,
                 vocab_size, train_time, y_true, y_pred, note=""):
    metrics = compute_metrics(y_true, y_pred)
    row = {
        "experiment": experiment,
        "representation": representation,
        "classifier": classifier,
        "split": split,
        "vocab_size": vocab_size,
        "train_time": round(train_time, 2),
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "note": note,
    }
    save_row(path, row)
    return metrics