## @gg
from abc import ABC, abstractmethod
from typing import Optional, List

class ModelAdapter(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
        repetition_penalty: float = 1.0,
    ) -> str:
        """
        Abstract method to generate text from a prompt.

        Args:
            prompt: Input text prompt to condition generation.
            max_new_tokens: Max number of tokens to generate.
            temperature: Controls randomness; lower is more deterministic.
            top_p: Nucleus sampling threshold for diversity.
            stop: Optional list of strings where generation should stop.
            repetition_penalty: Penalizes repeated tokens to reduce loops.

        Returns:
            Generated text string.
        """
        ...
