import os, json
from pathlib import Path
import yaml
import joblib
from tqdm import tqdm

from balm.balm import BALMPipeline, BiasEvaluationModule, BEMConfig, Policies, DiversityConfig
from balm.model_adapters.base import ModelAdapter
from balm.model_adapters.openai_adapter import OpenAIAdapter
from balm.model_adapters.bedrock_adapter import BedrockAdapter
from balm.model_adapters.mistral_adapter import MistralAdapter
from balm.model_adapters.generic_http import GenericHTTPAdapter
from balm.model_adapters.echo_adapter import EchoAdapter

def load_models(which: str, cfg: dict):
    if which == "gpt35":
        return OpenAIAdapter(cfg["openai"]["gpt35"], api_key=os.getenv("OPENAI_API_KEY") or cfg["openai"]["api_key"])
    if which == "gpt4":
        return OpenAIAdapter(cfg["openai"]["gpt4"], api_key=os.getenv("OPENAI_API_KEY") or cfg["openai"]["api_key"])
    if which == "titan_express":
        return BedrockAdapter(cfg["bedrock"]["titan_express"], region=cfg["bedrock"]["region"])
    if which == "titan_premier":
        return BedrockAdapter(cfg["bedrock"]["titan_premier"], region=cfg["bedrock"]["region"])
    if which == "mistral_large":
        return MistralAdapter(cfg["mistral"]["large"], api_key=os.getenv("MISTRAL_API_KEY") or cfg["mistral"]["api_key"])
    if which == "generic_http":
        entry = cfg["generic_http"]
        return GenericHTTPAdapter(endpoint=entry["endpoint"], model=entry["model"], api_key=os.getenv("XAI_API_KEY") or entry.get("api_key"))
    if which == "echo":
        return EchoAdapter()
    raise ValueError(f"Unknown model key {which}")

def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--models", type=str, required=True, help="comma separated keys: gpt35,gpt4,titan_express,titan_premier,mistral_large,generic_http,echo")
    p.add_argument("--detector_path", type=str, required=True)
    p.add_argument("--config_balm", type=str, default="configs/balm.yaml")
    p.add_argument("--config_models", type=str, default="configs/models.yaml")
    p.add_argument("--out_dir", type=str, default="runs/constructed")
    p.add_argument("--data_dir", type=str, default="data/constructed")
    args = p.parse_args()

    with open(args.config_balm, "r") as f:
        balm_cfg = yaml.safe_load(f)
    with open(args.config_models, "r") as f:
        model_cfg = yaml.safe_load(f)

    det = joblib.load(args.detector_path)
    bem = BiasEvaluationModule(det, BEMConfig(
        tau_low=balm_cfg["tau_low"], tau_high=balm_cfg["tau_high"], probe_tokens=balm_cfg["probe_tokens"]
    ))
    policies = Policies(
        temperature_base=balm_cfg["temperature_base"],
        top_p_base=0.95,
        temperature_bias=balm_cfg["temperature_bias"],
        repetition_penalty_base=balm_cfg["repetition_penalty_base"],
        repetition_penalty_bias=balm_cfg["repetition_penalty_bias"],
        max_new_tokens=balm_cfg["max_new_tokens"],
    )
    diversity = DiversityConfig(
        k=balm_cfg["k_candidates"],
        gamma_base=balm_cfg["gamma_base"],
        gamma_bias=balm_cfg["gamma_bias"],
    )

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Load datasets
    splits = {
        "biased": list(read_jsonl(Path(args.data_dir)/"biased.jsonl")),
        "neutral": list(read_jsonl(Path(args.data_dir)/"neutral.jsonl")),
        "near_neutral": list(read_jsonl(Path(args.data_dir)/"near_neutral.jsonl")),
    }

    for mk in args.models.split(","):
        adapter = load_models(mk, model_cfg)
        balm = BALMPipeline(adapter, bem, policies, diversity, seed=balm_cfg["seed"])
        out_file = Path(args.out_dir)/f"{mk}.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for split_name, items in splits.items():
                for item in tqdm(items, desc=f"{mk}:{split_name}"):
                    res = balm.generate(item["prompt"])
                    rec = {
                        "id": item["id"],
                        "category": item["category"],
                        "split": split_name,
                        "prompt": item["prompt"],
                        "response": res["response"],
                        "meta": res["meta"],
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote {out_file}")

if __name__ == "__main__":
    main()
