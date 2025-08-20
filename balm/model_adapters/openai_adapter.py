## @gg
import os
from typing import Optional, List
from openai import OpenAI

from .base import ModelAdapter

class OpenAIAdapter(ModelAdapter):
    def __init__(self, model: str, api_key: Optional[str] = None):
        # Use provided API key or fallback to environment variable; raise error if missing
        # 
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")
        
        # Initialize OpenAI client with the API key ....
        self.client = OpenAI(api_key=key)
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
        messages = [{"role": "user", "content": prompt}]
        
        # Call OpenAI's chat completion endpoint with given parameters
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            stop=stop
        )
        
        # Return the generated text or empty string if none provided
        return resp.choices[0].message.content or ""
