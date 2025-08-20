from typing import List, Dict, Any, Tuple
import numpy as np
from collections import Counter, defaultdict
from sacrebleu.metrics import BLEU

def challenge_rate(labels: List[int]) -> float:
    # Average of binary labels — assumes 1 = challenge, 0 = non-challenge
    return float(np.mean(labels)) if labels else 0.0

def self_bleu_group(samples: List[str]) -> float:
    # Measures diversity within a set of texts using self-BLEU
    # Lower score = higher diversity (less overlap between samples)
    bleu = BLEU(effective_order=True)
    n = len(samples)
    if n < 2:
        return 0.0  # Not enough data to compute diversity

    scores = []
    for i in range(n):
        # Compare sample[i] against all other samples as references
        refs = [samples[j] for j in range(n) if j != i]
        score = bleu.corpus_score([samples[i]], [refs]).score / 100.0
        scores.append(score)

    return float(np.mean(scores))

def homogeneity_gap(minority_samples: List[str], majority_samples: List[str]) -> Tuple[float, float, float]:
    # Compute diversity scores (via self-BLEU) for majority and minority groups
    sim_dom = self_bleu_group(majority_samples)  # Higher = more similar (less diverse)
    sim_sub = self_bleu_group(minority_samples)
    
    # Positive gap means minority group is more homogeneous (less diverse)
    gap = sim_sub - sim_dom
    return sim_dom, sim_sub, gap
