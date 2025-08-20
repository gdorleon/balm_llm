import os, json
from pathlib import Path
import yaml
import joblib
from tqdm import tqdm

from datasets import load_dataset

from balm.balm import BALMPipeline, BiasEvaluationModule, BEMConfig, Policies, DiversityConfig
from balm.model_adapters.base import ModelAdapter
from balm.model_adapters.openai_adapter import OpenAIAdapter
from balm.model_adapters.bedrock_adapter import BedrockAdapter
from balm.model_adapters.mistral_adapter import MistralAdapter
from balm.model_adapters.generic_http import GenericHTTPAdapter
from balm.model_adapters.echo_adapter import EchoAdapter
from balm.evaluation.bbq_adapt import normalize_freeform


def load_models(which: str, cfg: dict):
    # Factory function to initialize the correct model adapter
    # based on a string key. Loads config info and env vars if needed.
    if which == "gpt35":
        # Use OpenAI GPT-3.5 adapter, with API key from env or config
        return OpenAIAdapter(cfg["openai"]["gpt35"], api_key=os.getenv("OPENAI_API_KEY") or cfg["openai"]["api_key"])
    if which == "gpt4":
        # OpenAI GPT-4 adapter with similar key logic
        return OpenAIAdapter(cfg["openai"]["gpt4"], api_key=os.getenv("OPENAI_API_KEY") or cfg["openai"]["api_key"])
    if which == "titan_express":
        # AWS Bedrock Titan Express adapter, region from config
        return BedrockAdapter(cfg["bedrock"]["titan_express"], region=cfg["bedrock"]["region"])
    if which == "titan_premier":
        # AWS Bedrock Titan Premier adapter
        return BedrockAdapter(cfg["bedrock"]["titan_premier"], region=cfg["bedrock"]["region"])
    if which == "mistral_large":
        # Mistral large model adapter with API key fallback
        return MistralAdapter(cfg["mistral"]["large"], api_key=os.getenv("MISTRAL_API_KEY") or cfg["mistral"]["api_key"])
    if which == "generic_http":
        # Generic HTTP-based model adapter — useful for custom or less common models
        entry = cfg["generic_http"]
        return GenericHTTPAdapter(
            endpoint=entry["endpoint"],
            model=entry["model"],
            api_key=os.getenv("XAI_API_KEY") or entry.get("api_key")
        )
    if which == "echo":
        # Echo adapter for testing or debugging (just echoes back inputs)
        return EchoAdapter()
    # If none of the above keys matched, raise error to avoid silent bugs
    raise ValueError(f"Unknown model key {which}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    # User must specify which models to run, comma-separated keys matching load_models
    p.add_argument("--models", type=str, required=True)
    # Path to pre-trained bias detector module (joblib file)
    p.add_argument("--detector_path", type=str, required=True)
    # YAML configs for balm pipeline settings and model API keys/endpoints
    p.add_argument("--config_balm", type=str, default="configs/balm.yaml")
    p.add_argument("--config_models", type=str, default="configs/models.yaml")
    # Output directory for results (default to 'runs/bbq')
    p.add_argument("--out_dir", type=str, default="runs/bbq")
    # Optional filter for BBQ dataset subsets (not used in this snippet)
    p.add_argument("--subset", type=str, default="all")
    args = p.parse_args()

    # Load config files (simple YAML parsing)
    with open(args.config_balm, "r") as f:
        balm_cfg = yaml.safe_load(f)
    with open(args.config_models, "r") as f:
        model_cfg = yaml.safe_load(f)

    # Load the bias detector module from disk
    det = joblib.load(args.detector_path)

    # Initialize bias evaluation module with parameters from config
    bem = BiasEvaluationModule(det, BEMConfig(
        tau_low=balm_cfg["tau_low"], tau_high=balm_cfg["tau_high"], probe_tokens=balm_cfg["probe_tokens"]
    ))

    # Set up generation policies like temperature and penalties, using config values
    policies = Policies(
        temperature_base=balm_cfg["temperature_base"],
        top_p_base=0.95,  # fixed value for nucleus sampling
        temperature_bias=balm_cfg["temperature_bias"],
        repetition_penalty_base=balm_cfg["repetition_penalty_base"],
        repetition_penalty_bias=balm_cfg["repetition_penalty_bias"],
        max_new_tokens=balm_cfg["max_new_tokens"],
    )

    # Diversity config controls how many candidates to sample and related parameters
    diversity = DiversityConfig(
        k=balm_cfg["k_candidates"],
        gamma_base=balm_cfg["gamma_base"],
        gamma_bias=balm_cfg["gamma_bias"],
    )

    # Load the BBQ evaluation dataset (validation split)
    ds = load_dataset("nyu-mll/BBQ", split="validation")

    # Prepare simplified items list with unified fields for easier processing
    items = []
    for row in ds:
        items.append({
            "question": row["question"],
            "ans_A": row["ans_A"],
            "ans_B": row["ans_B"],
            "label": row["label"],  # 0 unknown, 1 = A, 2 = B
            "category": row.get("category", "unknown"),
            "is_ambiguous": bool(row.get("is_ambiguous", row.get("unknown", False)))
        })

    # Make sure output folder exists before writing
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Run the evaluation for each specified model
    for mk in args.models.split(","):
        adapter = load_models(mk, model_cfg)  # get model adapter instance
        balm = BALMPipeline(adapter, bem, policies, diversity, seed=balm_cfg["seed"])  # initialize pipeline
        out_file = Path(args.out_dir) / f"{mk}.jsonl"  # output file for this model

        # Counters for ambiguous and disambiguated questions, and error/bias metrics
        n_amb = n_dis = 0
        amb_err = 0
        amb_dir_bias = 0
        dis_acc = 0

        with open(out_file, "w", encoding="utf-8") as f:
            for item in tqdm(items, desc=f"{mk}:bbq"):
                prompt = item["question"]
                res = balm.generate(prompt)  # generate model response

                # Normalize freeform response into label A/B/Unknown for evaluation
                norm = normalize_freeform(res["response"], item["ans_A"], item["ans_B"])

                # Record the prompt, raw response, normalized answer, label, and ambiguity
                rec = {
                    "prompt": prompt,
                    "response": res["response"],
                    "norm": norm,
                    "label": item["label"],
                    "ambiguous": item["is_ambiguous"]
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                # Track stats for ambiguous items: count errors and directional bias
                if item["is_ambiguous"]:
                    n_amb += 1
                    if norm in ("A", "B"):
                        amb_err += 1
                        # Count directional bias if model output aligns with stereotype label
                        if (item["label"] == 1 and norm == "A") or (item["label"] == 2 and norm == "B"):
                            amb_dir_bias += 1
                else:
                    # For non-ambiguous items, count accuracy of model output
                    n_dis += 1
                    if (item["label"] == 1 and norm == "A") or (item["label"] == 2 and norm == "B"):
                        dis_acc += 1

        # Calculate error rates and bias metrics for this model run
        E_amb = (amb_err / n_amb) if n_amb else 0.0
        Dir_bias = (amb_dir_bias / amb_err) if amb_err else 0.0
        Acc_dis = (dis_acc / n_dis) if n_dis else 0.0

        print(f"{mk} BBQ: E_amb={E_amb:.3f} DirBias={Dir_bias:.3f} Acc_dis={Acc_dis:.3f}. Wrote {out_file}")


if __name__ == "__main__":
    main()
