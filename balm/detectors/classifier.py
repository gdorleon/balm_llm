import joblib
from typing import Iterable
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def build_detector_pipeline() -> Pipeline:
    # Basic text classification pipeline:
    # - TF-IDF on unigrams + bigrams
    # - Logistic regression with slight class weight adjustment
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9)),
        ("clf", LogisticRegression(max_iter=200, class_weight={0: 1.0, 1: 1.5}))
    ])
    return pipe

def save_detector(model: Pipeline, path: str):
    # Dump model to disk for later use
    joblib.dump(model, path)

def load_detector(path: str) -> Pipeline:
    # Reload saved model from disk
    return joblib.load(path)
