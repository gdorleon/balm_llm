from typing import List
import sacrebleu

def self_bleu_similarity(a: str, b: str) -> float:
    # Convert BLEU to a similarity in [0,1]
    bleu = sacrebleu.sentence_bleu(a, [b]).score / 100.0
    return bleu
