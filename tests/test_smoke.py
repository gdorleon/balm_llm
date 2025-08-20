from balm.balm import BALMPipeline, BiasEvaluationModule, Policies, DiversityConfig, BEMConfig
from balm.model_adapters.echo_adapter import EchoAdapter
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import numpy as np

def test_pipeline_smoke():
    # Tiny detector that labels anything with 'why' as biased
    pipe = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression(max_iter=100))])
    X = ["why are women bad at math", "how to improve code quality"]
    y = [1,0]
    pipe.fit(X, y)
    bem = BiasEvaluationModule(pipe, BEMConfig(probe_tokens=8, tau_low=0.3, tau_high=0.7))
    balm = BALMPipeline(EchoAdapter(), bem, Policies(), DiversityConfig(), seed=1)
    out = balm.generate("Why are women late to meetings?")
    assert "ECHO TEST" in out["response"]
