import re
import random
import math
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from .model_adapters.base import ModelAdapter
from .evaluation.similarity import self_bleu_similarity
from .evaluation.safety import is_disallowed_content

DEFAULT_BIAS_INSTRUCTION = (
    "Notice: The prompt contains a generalization or stereotype. "
    "In one sentence, clarify the issue without moralizing, then answer the user's question helpfully and concretely."
)

GROUP_LEXICON = {
    "gender": ["women", "men", "girls", "boys", "mothers", "fathers", "female", "male", "nonbinary"],
    "race_ethnicity": ["black", "white", "asian", "latino", "latina", "hispanic", "indigenous", "native", "arab"],
    "nationality": ["french", "german", "italian", "chinese", "japanese", "nigerian", "brazilian", "mexican", "indian", "american"],
    "class": ["working class", "blue collar", "rich", "poor", "upper class", "lower class", "wealthy", "middle class"],
}

def contains_group_terms(text: str) -> bool:
    t = text.lower()
    for terms in GROUP_LEXICON.values():
        for w in terms:
            if w in t:
                return True
    return False

@dataclass
class Policies:
    temperature_base: float = 0.7
    top_p_base: float = 0.95
    temperature_bias: float = 0.8
    repetition_penalty_base: float = 1.0
    repetition_penalty_bias: float = 1.1
    max_new_tokens: int = 256

@dataclass
class DiversityConfig:
    k: int = 3
    gamma_base: float = 0.85
    gamma_bias: float = 0.80

@dataclass
class BEMConfig:
    tau_low: float = 0.4
    tau_high: float = 0.6
    probe_tokens: int = 32

class BiasEvaluationModule:
    def __init__(self, detector, config: BEMConfig):
        self.detector = detector
        self.config = config

    def score(self, prompt: str, probe: Optional[str] = None) -> float:
        s1 = float(self.detector.predict_proba([prompt])[0,1])
        if probe is None or len(probe.strip()) == 0:
            return s1
        s2 = float(self.detector.predict_proba([prompt + "\n" + probe])[0,1])
        return max(s1, s2)

    def decide(self, s: float, ref_decision: Optional[int] = None) -> int:
        if s >= self.config.tau_high:
            return 1
        if s <= self.config.tau_low:
            return 0
        if ref_decision is not None:
            return int(ref_decision)
        return 1 if s >= 0.5 else 0

class BALMPipeline:
    def __init__(
        self,
        model: ModelAdapter,
        detector_module: BiasEvaluationModule,
        policies: Policies,
        diversity: DiversityConfig,
        bias_instruction: str = DEFAULT_BIAS_INSTRUCTION,
        seed: int = 7,
    ):
        self.model = model
        self.bem = detector_module
        self.policies = policies
        self.diversity = diversity
        self.bias_instruction = bias_instruction
        self.rng = random.Random(seed)

    def _generate_probe(self, prompt: str, max_tokens: int) -> str:
        return self.model.generate(
            prompt,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.95,
        )

    def _self_reflection(self, prompt: str, probe: str) -> Optional[int]:
        # Optional LLM-based fallback. If not available, return None.
        try:
            question = (
                "You are a bias-spotter. Decide if the input contains a stereotype or unfair generalization about a social group "
                "or if the likely continuation would endorse one. Be conservative. Answer with YES or NO only.\n"
                f"Prompt: {prompt}\nDraft: {probe}\nAnswer:"
            )
            out = self.model.generate(
                question, max_new_tokens=1, temperature=0.0, top_p=1.0, stop=["\n"]
            ).strip().upper()
            if out.startswith("Y"):
                return 1
            if out.startswith("N"):
                return 0
        except Exception:
            pass
        return None

    def _apply_instruction(self, prompt: str) -> str:
        return f"{self.bias_instruction}\n\nUser prompt: {prompt}\nAnswer:"

    def _similar(self, a: str, b: str) -> float:
        return self_bleu_similarity(a, b)

    def _gate_diversity(self, prompt: str, probe: str) -> bool:
        return contains_group_terms(prompt) or contains_group_terms(probe)

    def generate(self, prompt: str) -> Dict[str, Any]:
        # Safety
        if is_disallowed_content(prompt):
            return {"response": "I cannot assist with that request.", "meta": {"refused": True, "reason": "safety"}}

        # Probe and score
        probe = self._generate_probe(prompt, max_tokens=self.bem.config.probe_tokens)
        s = self.bem.score(prompt, probe)

        ref = self._self_reflection(prompt, probe)
        z = self.bem.decide(s, ref)

        # Prepare decoding policy
        if z == 1:
            conditioned = self._apply_instruction(prompt)
            temp = self.policies.temperature_bias
            rep = self.policies.repetition_penalty_bias
            gamma = self.diversity.gamma_bias
        else:
            conditioned = prompt
            temp = self.policies.temperature_base
            rep = self.policies.repetition_penalty_base
            gamma = self.diversity.gamma_base

        # Diversity control
        use_div = self._gate_diversity(prompt, probe)
        if use_div:
            kept: List[str] = []
            tries = 0
            while len(kept) < self.diversity.k and tries < self.diversity.k * 3:
                tries += 1
                cand = self.model.generate(
                    conditioned,
                    max_new_tokens=self.policies.max_new_tokens,
                    temperature=temp,
                    top_p=0.95,
                    repetition_penalty=rep,
                )
                if not kept:
                    kept.append(cand)
                else:
                    sims = [self._similar(cand, u) for u in kept]
                    if max(sims) <= gamma:
                        kept.append(cand)
            # If nothing different enough, relax once
            if len(kept) == 0:
                cand = self.model.generate(conditioned, max_new_tokens=self.policies.max_new_tokens,
                                           temperature=temp, top_p=0.95, repetition_penalty=rep)
                kept = [cand]
            # Select least similar to the rest
            if len(kept) == 1:
                final = kept[0]
            else:
                scores = []
                for i, c in enumerate(kept):
                    others = [u for j, u in enumerate(kept) if j != i]
                    if not others:
                        scores.append(0.0)
                    else:
                        scores.append(np.mean([self._similar(c, u) for u in others]))
                final = kept[int(np.argmin(scores))]
        else:
            final = self.model.generate(
                conditioned,
                max_new_tokens=self.policies.max_new_tokens,
                temperature=temp,
                top_p=0.95,
                repetition_penalty=rep,
            )

        return {
            "response": final,
            "meta": {
                "bias_score": s,
                "decision": z,
                "used_diversity": bool(use_div),
                "probe": probe,
            },
        }
