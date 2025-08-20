from typing import List
import sacrebleu

def self_bleu_similarity(a: str, b: str) -> float:
    # Here we compute BLEU score between two sentences and normalize to [0,1]
    bleu = sacrebleu.sentence_bleu(a, [b]).score / 100.0
    return bleu
