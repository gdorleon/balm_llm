## @AP
import os
# import requests
from typing import Optional, List
from mistralai import Mistral

from .base import ModelAdapter

class MistralAdapter(ModelAdapter):
    def __init__(self, model: str, api_key: Optional[str] = None):
        # Use provided API key or fallback to env variable; error if missing
        key = api_key or os.getenv("MISTRAL_API_KEY")
        if not key:
            raise RuntimeError("MISTRAL_API_KEY not set")
        
        # Initialize Mistral client with API key
        self.client = Mistral(api_key=key)
        self.model = model

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
        repetition_penalty: float = 1.0
    ) -> str:
        # Use Mistral client to complete chat prompt
        resp = self.client.chat.complete(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            stop=stop
        )
        # Return generated content or empty string if missing
        return resp.choices[0].message.content or ""
