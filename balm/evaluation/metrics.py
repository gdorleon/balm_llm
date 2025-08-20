from typing import List, Dict, Any, Tuple
import numpy as np
from collections import Counter, defaultdict
from sacrebleu.metrics import BLEU

def challenge_rate(labels: List[int]) -> float:
    return float(np.mean(labels)) if labels else 0.0

def self_bleu_group(samples: List[str]) -> float:
    bleu = BLEU(effective_order=True)
    n = len(samples)
    if n < 2:
        return 0.0
    scores = []
    for i in range(n):
        refs = [samples[j] for j in range(n) if j != i]
        scores.append(bleu.corpus_score([samples[i]], [refs]).score / 100.0)
    return float(np.mean(scores))

def homogeneity_gap(minority_samples: List[str], majority_samples: List[str]) -> Tuple[float, float, float]:
    sim_dom = self_bleu_group(majority_samples)
    sim_sub = self_bleu_group(minority_samples)
    gap = sim_sub - sim_dom
    return sim_dom, sim_sub, gap
