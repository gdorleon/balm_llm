import os, json
from pathlib import Path
import numpy as np

def aggregate_constructed(run_dir: str):
    # This function is a stub that collects simple counts. In the full paper, we use human annotations for CR and utility.
    # Here we only compute counts of BALM decision==1 and basic lengths.
    rows = []
    for p in Path(run_dir).glob("*.jsonl"):
        model = p.stem
        n = 0
        n_bias = 0
        n_div = 0
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
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    args = p.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Just demonstrate aggregation.
    cons = Path(args.runs_dir)/"constructed"
    if cons.exists():
        print("Constructed summary:")
        aggregate_constructed(str(cons))

if __name__ == "__main__":
    main()
