## @gg/@AP
import re
import random
import math
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from .model_adapters.base import ModelAdapter
from .evaluation.similarity import self_bleu_similarity
from .evaluation.safety import is_disallowed_content

# Instruction to bias-aware models when a stereotype/generalization is detected
DEFAULT_BIAS_INSTRUCTION = (
    "Notice: The prompt contains a generalization or stereotype. "
    "In one sentence, clarify the issue without moralizing, then answer the user's question helpfully and concretely."
)

# Predefined lists of group-related words to detect if prompt contains references to sensitive groups
GROUP_LEXICON = {
    "gender": ["women", "men", "girls", "boys", "mothers", "fathers", "female", "male", "nonbinary"],
    "race_ethnicity": ["black", "white", "asian", "latino", "latina", "hispanic", "indigenous", "native", "arab"],
    "nationality": [
        "french", "german", "italian", "chinese", "vietnamese", "japanese", "nigerian", "brazilian", "mexican",
        "indian", "american", "haitian", "canadian", "argentinian", "russian", "ukrainian", "turkish", "egyptian",
        "ethiopian", "kenyan", "south_african", "saudi", "iranian", "pakistani", "bangladeshi", "indonesian", "thai",
        "korean", "australian", "british", "syrian", "iraqi", "moroccan", "algerian", "tunisian", "senegalese",
        "ghanaian", "congolese", "peruvian", "chilean", "colombian", "venezuelan", "cuban", "philippine", "malaysian",
        "singaporean", "nepalese", "sri_lankan", "polish", "swedish", "dutch"
    ],
    "class": ["working class", "blue collar", "rich", "poor", "upper class", "lower class", "wealthy", "middle class"],
}

def contains_group_terms(text: str) -> bool:
    """Check if input text contains any terms related to social groups."""
    t = text.lower()
    for terms in GROUP_LEXICON.values():
        for w in terms:
            if w in t:
                return True
    return False

@dataclass
class Policies:
    """Holds parameters for language generation policies, including temperature and repetition penalty."""
    temperature_base: float = 0.7
    top_p_base: float = 0.95
    temperature_bias: float = 0.8
    repetition_penalty_base: float = 1.0
    repetition_penalty_bias: float = 1.1
    max_new_tokens: int = 256

@dataclass
class DiversityConfig:
    """Configuration for diversity control during generation (number of candidates and similarity threshold)."""
    k: int = 3  # Number of diverse candidates to generate
    gamma_base: float = 0.85  # Similarity threshold for base prompts
    gamma_bias: float = 0.80  # Similarity threshold for bias-conditioned prompts

@dataclass
class BEMConfig:
    """Config for bias evaluation thresholds and probe token length."""
    tau_low: float = 0.4  # Low threshold to decide no bias
    tau_high: float = 0.6  # High threshold to decide bias
    probe_tokens: int = 32  # Number of tokens to generate in probe phase

class BiasEvaluationModule:
    """Uses an external detector to evaluate if a prompt (or prompt+probe) shows bias/stereotype."""

    def __init__(self, detector, config: BEMConfig):
        self.detector = detector
        self.config = config

    def score(self, prompt: str, probe: Optional[str] = None) -> float:
        """
        Score bias probability on the prompt alone and optionally on prompt+probe concatenation.
        Return the higher score to be conservative.
        """
        s1 = float(self.detector.predict_proba([prompt])[0,1])
        if probe is None or len(probe.strip()) == 0:
            return s1
        s2 = float(self.detector.predict_proba([prompt + "\n" + probe])[0,1])
        return max(s1, s2)

    def decide(self, s: float, ref_decision: Optional[int] = None) -> int:
        """
        Decide bias presence from score and optional reference decision.
        Returns 1 if bias detected, 0 otherwise.
        """
        if s >= self.config.tau_high:
            return 1
        if s <= self.config.tau_low:
            return 0
        if ref_decision is not None:
            return int(ref_decision)
        # If unsure, default to bias if score >= 0.5
        return 1 if s >= 0.5 else 0

class BALMPipeline:
    """Pipeline managing bias-aware language generation with probe, evaluation, and diversity control."""

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
        """Generate a short probe text continuation from the model for bias evaluation."""
        return self.model.generate(
            prompt,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.95,
        )

    def _self_reflection(self, prompt: str, probe: str) -> Optional[int]:
        """
        Optional fallback: use LLM itself to decide if prompt + probe indicates bias.
        Returns 1 for bias, 0 for no bias, or None if no decision.
        """
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
        """Add bias instruction prompt to condition the model to handle stereotypes carefully."""
        return f"{self.bias_instruction}\n\nUser prompt: {prompt}\nAnswer:"

    def _similar(self, a: str, b: str) -> float:
        """Compute similarity score between two texts using self-BLEU metric."""
        return self_bleu_similarity(a, b)

    def _gate_diversity(self, prompt: str, probe: str) -> bool:
        """
        Decide whether to activate diversity control by checking
        if prompt or probe contains group-related terms.
        """
        return contains_group_terms(prompt) or contains_group_terms(probe)

    def generate(self, prompt: str) -> Dict[str, Any]:
        """
        Main method to generate a bias-aware response for a given prompt.
        Steps:
        1. Check for disallowed content and refuse if detected.
        2. Generate a probe text to evaluate bias.
        3. Score bias and optionally use self-reflection.
        4. Decide bias presence and set generation parameters accordingly.
        5. Generate multiple candidates if diversity control is active,
           else generate a single response.
        6. Return final response with meta information.
        """
        # Safety check: refuse to handle disallowed content
        if is_disallowed_content(prompt):
            return {"response": "I cannot assist with that request.", "meta": {"refused": True, "reason": "safety"}}

        # Generate probe to get bias score
        probe = self._generate_probe(prompt, max_tokens=self.bem.config.probe_tokens)
        s = self.bem.score(prompt, probe)

        # Optional LLM-based self-reflection bias decision
        ref = self._self_reflection(prompt, probe)
        z = self.bem.decide(s, ref)

        # Setup generation policy based on bias decision
        if z == 1:
            # Bias detected: add bias instruction and use bias-specific generation params
            conditioned = self._apply_instruction(prompt)
            temp = self.policies.temperature_bias
            rep = self.policies.repetition_penalty_bias
            gamma = self.diversity.gamma_bias
        else:
            # No bias detected: use plain prompt and base generation params
            conditioned = prompt
            temp = self.policies.temperature_base
            rep = self.policies.repetition_penalty_base
            gamma = self.diversity.gamma_base

        # Check if diversity control is needed (based on group terms)
        use_div = self._gate_diversity(prompt, probe)

        if use_div:
            # Generate multiple candidates ensuring diversity
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
                    # Calculate similarity to existing kept candidates
                    sims = [self._similar(cand, u) for u in kept]
                    # Keep candidate only if it is sufficiently different
                    if max(sims) <= gamma:
                        kept.append(cand)

            # If no diverse candidates found, generate one without similarity check
            if len(kept) == 0:
                cand = self.model.generate(conditioned,
                                           max_new_tokens=self.policies.max_new_tokens,
                                           temperature=temp,
                                           top_p=0.95,
                                           repetition_penalty=rep)
                kept = [cand]

            # Select candidate least similar to others to maximize diversity
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
            # No diversity needed, just generate a single response
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
