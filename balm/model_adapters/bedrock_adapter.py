## @AP
import os
import json
from typing import Optional, List
import boto3

from .base import ModelAdapter

class BedrockAdapter(ModelAdapter):
    def __init__(self, model_id: str, region: Optional[str] = None):
        # Use provided region, environment variable, or default to us-east-1
        region = region or os.getenv("AWS_REGION") or "us-east-1"
        # Initialize AWS Bedrock client for runtime calls
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
        repetition_penalty: float = 1.0
    ) -> str:
        # Prepare request body with prompt and generation config
        body = {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": max_new_tokens,
                "temperature": temperature,
                "topP": top_p
            }
        }
        # Invoke the Bedrock model endpoint
        resp = self.client.invoke_model(modelId=self.model_id, body=json.dumps(body))

        # Parse JSON response; Titan model returns a list of results
        out = json.loads(resp["body"].read())

        # Return the generated text or empty string if missing
        return out.get("results", [{}])[0].get("outputText", "")
