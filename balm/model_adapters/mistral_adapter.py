import os
from typing import Optional, List
from mistralai import Mistral

from .base import ModelAdapter

class MistralAdapter(ModelAdapter):
    def __init__(self, model: str, api_key: Optional[str] = None):
        key = api_key or os.getenv("MISTRAL_API_KEY")
        if not key:
            raise RuntimeError("MISTRAL_API_KEY not set")
        self.client = Mistral(api_key=key)
        self.model = model

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7,
                 top_p: float = 0.95, stop: Optional[List[str]] = None, repetition_penalty: float = 1.0) -> str:
        resp = self.client.chat.complete(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            stop=stop
        )
        return resp.choices[0].message.content or ""
