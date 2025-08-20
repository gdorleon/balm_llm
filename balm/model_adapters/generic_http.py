import os, json, requests
from typing import Optional, List
from .base import ModelAdapter

class GenericHTTPAdapter(ModelAdapter):
    """
    A minimal HTTP adapter that can be used for providers like xAI or OpenRouter.
    Edit endpoint and payload format as needed.
    """
    def __init__(self, endpoint: str, model: str, api_key: Optional[str] = None, header_name: str = "Authorization"):
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key or os.getenv("GENERIC_API_KEY") or os.getenv("XAI_API_KEY")
        self.header_name = header_name

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7,
                 top_p: float = 0.95, stop: Optional[List[str]] = None, repetition_penalty: float = 1.0) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers[self.header_name] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop
        }
        r = requests.post(self.endpoint, headers=headers, data=json.dumps(payload), timeout=60)
        r.raise_for_status()
        data = r.json()
        # try OpenAI-style format
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            return json.dumps(data)
