import re
from collections import Counter


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

def main():
    corpus = [
        "the cat eats fish",
        "the cat likes fish"
    ]

    model = NGramLanguageModel(2)

    model.build_vocabulary(corpus)
    model.count_ngrams(corpus)
    model.train_unigram()
    model.train_bigram()

    print("Vocabulary:", model.vocabulary)
    print("Unigram counts:", model.unigram_counts)
    print("Bigram counts:", model.bigram_counts)
    print("Bigram probabilities:", model.bigram_probs)

    sentence = "the cat eats fish"
    print("Sentence probability:",
          model.sentence_probability(sentence))


if __name__ == "__main__":
    main()