import joblib
from typing import Iterable
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def build_detector_pipeline() -> Pipeline:
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.9)),
        ("clf", LogisticRegression(max_iter=200, class_weight={0:1.0, 1:1.5}))
    ])
    return pipe

def save_detector(model: Pipeline, path: str):
    joblib.dump(model, path)

def load_detector(path: str) -> Pipeline:
    return joblib.load(path)
