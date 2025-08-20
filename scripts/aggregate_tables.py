import os
import json
from pathlib import Path
## @AP
def aggregate_constructed(run_dir: str):
    """
    Aggregates results for constructed prompts from JSONL files in the directory.
    
    Prints a CSV summary with:
    - model name
    - total records processed
    - count of bias decision == 1
    - count of diversity usage
    """
    rows = []
    for p in Path(run_dir).glob("*.jsonl"):
        model = p.stem  # filename without extension, used as model name
        n = 0           # total count of lines (records)
        n_bias = 0      # count where decision == 1
        n_div = 0       # count where diversity was used
        with open(p, encoding="utf-8") as f:
            for line in f:
                n += 1
                rec = json.loads(line)
                if rec["meta"].get("decision", 0) == 1:
                    n_bias += 1
                if rec["meta"].get("used_diversity", False):
                    n_div += 1
        rows.append((model, n, n_bias, n_div))
    rows.sort()
    
    print("Model, total, decision1_count, diversity_used")
    for r in rows:
        print(",".join(map(str, r)))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate bias detection and diversity stats from runs.")
    parser.add_argument("--runs_dir", type=str, required=True, help="Directory containing run outputs")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory for outputs (not used here)")

    args = parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Look for a "constructed" subfolder inside runs_dir
    cons = Path(args.runs_dir) / "constructed"
    if cons.exists():
        print("Constructed summary:")
        aggregate_constructed(str(cons))
    else:
        print(f"No constructed directory found at {cons}")

if __name__ == "__main__":
    main()
