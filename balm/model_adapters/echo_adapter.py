from typing import Optional, List
from .base import ModelAdapter

class EchoAdapter(ModelAdapter):
    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7,
                 top_p: float = 0.95, stop: Optional[List[str]] = None, repetition_penalty: float = 1.0) -> str:
        return f"[ECHO TEST OUTPUT]\\nPrompt:\\n{prompt}\\n[END]"
