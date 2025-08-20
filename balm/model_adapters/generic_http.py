## @gg / @AP
import os, json, requests
from typing import Optional, List
from .base import ModelAdapter

class GenericHTTPAdapter(ModelAdapter):
    """
    Minimal HTTP adapter for any model provider supporting a similar API,
    e.g. xAI, OpenRouter. Customize endpoint and payload as needed.
    """
    def __init__(self, endpoint: str, model: str, api_key: Optional[str] = None, header_name: str = "Authorization"):
        self.endpoint = endpoint
        self.model = model
        # Use provided API key or fall back to environment variables
        self.api_key = api_key or os.getenv("GENERIC_API_KEY") or os.getenv("XAI_API_KEY")
        self.header_name = header_name

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
        repetition_penalty: float = 1.0
    ) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers[self.header_name] = f"Bearer {self.api_key}"

        # Compose request payload in a common chat format
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop
        }

        # Make the POST request with a timeout to avoid hanging
        r = requests.post(self.endpoint, headers=headers, data=json.dumps(payload), timeout=60)
        r.raise_for_status()

        data = r.json()
        # Try to extract generated text in OpenAI-style response format
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            # Fallback: return raw JSON if unexpected response format
            return json.dumps(data)
