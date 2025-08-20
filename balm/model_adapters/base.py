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
        ...
