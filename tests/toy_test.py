from balm.balm import BALMPipeline, BiasEvaluationModule, Policies, DiversityConfig, BEMConfig
from balm.model_adapters.echo_adapter import EchoAdapter
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import numpy as np
#@gg
def test_pipeline_smoke():
    # Quick and dirty detector: anything containing like 'why' is flagged as biased
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer()), 
        ("clf", LogisticRegression(max_iter=100))
    ])
    
    # Training data: one biased example (has 'why'), one neutral
    X = ["why are women bad at math", "how to improve code quality"]
    y = [1, 0]

    # Train the detector
    pipe.fit(X, y)

    # Create Bias Evaluation Module (BEM) with some chosen thresholds and probe tokens
    bem = BiasEvaluationModule(pipe, BEMConfig(probe_tokens=8, tau_low=0.3, tau_high=0.7))

    # BALM pipeline uses EchoAdapter (just echoes input + some fixed text), BEM, plus default policies & diversity
    balm = BALMPipeline(EchoAdapter(), bem, Policies(), DiversityConfig(), seed=1)

    # Generate response for a prompt that includes 'Why', so should trigger bias detection
    out = balm.generate("Why are women late to meetings?")

    # Simple sanity check: the EchoAdapter response includes the string "ECHO TEST"
    assert "ECHO TEST" in out["response"]
