from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import chi2, mutual_info_classif
import numpy as np
import logging

class EmbeddingFactory:
    def __init__(self, method: str = 'tfidf', max_features: int = 5000):
        self.method = method
        self.vectorizer = None
        self.stat_scores = None
        self.max_features = max_features
        self.fitted = False

    def fit(self, texts, labels=None):
        if self.method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=(1, 2),
                stop_words='english'
            )
            self.vectorizer.fit(texts)
            self.fitted = True

        elif self.method in ['chi2', 'mi', 'bow']:
            self.vectorizer = CountVectorizer(max_features=self.max_features)
            X = self.vectorizer.fit_transform(texts)

            if self.method == 'chi2':
                if labels is None:
                    raise ValueError("Labels are required for chi2")
                self.stat_scores, _ = chi2(X, labels)

            elif self.method == 'mi':
                if labels is None:
                    raise ValueError("Labels are required for mutual information")
                self.stat_scores = mutual_info_classif(X, labels)

            elif self.method == 'bow':
                self.stat_scores = None

            self.fitted = True

        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def transform(self, texts):
        if not self.fitted:
            raise RuntimeError("Call fit() before transform()")

        X = self.vectorizer.transform(texts).toarray()

        if self.stat_scores is not None:
            return X * self.stat_scores
        return X
