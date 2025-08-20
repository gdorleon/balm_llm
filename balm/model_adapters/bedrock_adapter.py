import os
import json
from typing import Optional, List
import boto3

from .base import ModelAdapter

class BedrockAdapter(ModelAdapter):
    def __init__(self, model_id: str, region: Optional[str] = None):
        region = region or os.getenv("AWS_REGION") or "us-east-1"
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7,
                 top_p: float = 0.95, stop: Optional[List[str]] = None, repetition_penalty: float = 1.0) -> str:
        body = {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": max_new_tokens,
                "temperature": temperature,
                "topP": top_p
            }
        }
        resp = self.client.invoke_model(modelId=self.model_id, body=json.dumps(body))
        out = json.loads(resp["body"].read())
        # Titan returns an array of results
        return out.get("results", [{}])[0].get("outputText", "")
