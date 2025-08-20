## BALM: Bias-Aware Generation for Large Language Models

This repository contains the full experimental pipeline used in the paper **BALM** (Bias-Aware Language Model) 
It includes:
- A constructed bias testbed with **400 biased prompts** across four categories, plus **200 neutral** and **50 near-neutral** controls.
- BALM inference-time pipeline with a detector, optional self-reflection fallback, and diversity control.
- Adapters for several model providers: OpenAI GPT-3.5 and GPT-4, Amazon Bedrock Titan Express and Titan Premier, Mistral Large, and a generic HTTP adapter that can be modified for other providers like Grok 3 mini.
- Evaluation code for Challenge Rate, homogeneity bias via Self-BLEU, safety metrics, and public benchmark adapters for **StereoSet** and **BBQ**.

> Note that this repo does not ship vendor model weights, obviously. You need API access for the providers you choose to evaluate.

## Quick Start

### 1. Install
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

### 2. Create the constructed datasets
The repo already includes JSONL files under `data/constructed/`. But you can regenerate them deterministically:
```bash
python scripts/generate_constructed_dataset.py --out_dir data/constructed
```

### 3. Configure models
Copy `configs/models.example.yaml` to `configs/models.yaml` and edit the keys. Example:
```yaml
openai:
  api_key: "$OPENAI_API_KEY"
  gpt35: "gpt-3.5-turbo"
  gpt4: "gpt-4o-mini"
bedrock:
  region: "us-east-1"
  titan_express: "amazon.titan-text-express-v1"
  titan_premier: "amazon.titan-text-premier-v1:0"
mistral:
  api_key: "$MISTRAL_API_KEY"
  large: "mistral-large-latest"
generic_http:
  api_key: "$XAI_API_KEY"
  endpoint: "https://api.x.ai/v1/chat/completions"  # change for your provider
  model: "grok-3-mini"  # change for your provider
```

Environment variables are supported. You can also pass keys directly in the YAML file.

### 4. Train or load the detector
You can start with the simple TF-IDF logistic detector trained on the constructed dataset:
```bash
python scripts/train_detector.py --train constructed --save_path data/detector.joblib
```
Or provide your own labeled set via `--csv path/to/data.csv` with columns `text,label` where label in {0,1}.

### 5. Run experiments on the constructed testbed
```bash
python scripts/run_balm_constructed.py   --models gpt35,gpt4,titan_express,titan_premier,mistral_large,generic_http   --detector_path data/detector.joblib   --out_dir runs/constructed
```

### 6. Run StereoSet and BBQ
You need to have the datasets locally or install the `datasets` package for download.
```bash
python scripts/run_balm_stereoset.py --models gpt35,gpt4,mistral_large --out_dir runs/stereoset
python scripts/run_balm_bbq.py --models gpt35,gpt4,mistral_large --out_dir runs/bbq
```

### 7. Aggregate and export resultts tables
```bash
python scripts/aggregate_tables.py --runs_dir runs --out_dir tables
```

## Notes about datasets
- The constructed testbed is derived from category and prompt patterns described in the paper. Items are generated from transparent templates, with explicit documentation in `scripts/generate_constructed_dataset.py`. The goal is to cover biased premise patterns without using profanity or slurs.
- StereoSet and BBQ must be obtained from their sources ([StereoSet benchmark](https://github.com/moinnadeem/StereoSet), [BBQ](https://github.com/nyu-mll/BBQ#models)) or via the `datasets` library. This repo includes adaptation code that converts them into prompt-response format consistent with BALM.

## Citation

If you use this codebase, please cite our paper:

```bibtex
@inproceedings{...,
  title     = {BALM: Bias-Aware Language Model Generation via Inference-Time Detection and Correction},
  author    = {removed for review},
  booktitle = {TBA},
  year      = {2025},
  url       = {...}
}

