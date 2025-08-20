import os, json
from pathlib import Path
import numpy as np

def aggregate_constructed(run_dir: str):
    # This function reads all .jsonl files in the given directory (run_dir)
    # It summarizes some counts from each file:
    # - total number of records
    # - how many had "decision" == 1 in their metadata
    # - how many used "diversity" according to metadata
    # Note: In the full project, more complex human annotations are used,
    # but here we just do simple counting for demonstration.

    rows = []  # to hold summary info per model/file

    # Iterate over all jsonl files in the directory
    for p in Path(run_dir).glob("*.jsonl"):
        model = p.stem  # filename without extension, used as model name
        n = 0          # total records counter
        n_bias = 0     # counts of records with decision==1
        n_div = 0      # counts of records that used diversity flag

        with open(p, encoding="utf-8") as f:
            for line in f:
                n += 1
                rec = json.loads(line)  # parse the json object

                # Check if "decision" key exists and equals 1; count if so
                if rec["meta"].get("decision", 0) == 1:
                    n_bias += 1

                # Check if diversity flag was used; count if true
                if rec["meta"].get("used_diversity", False):
                    n_div += 1

        # Store aggregated counts for this file/model
        rows.append((model, n, n_bias, n_div))

    # Sort by model name alphabetically for consistent output
    rows.sort()

    # Print a header and then each summary row as CSV
    print("Model, total, decision1_count, diversity_used")
    for r in rows:
        print(",".join(map(str, r)))


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", type=str, required=True, help="Directory containing run outputs")
    p.add_argument("--out_dir", type=str, required=True, help="Directory to save outputs (not used here but created)")

    args = p.parse_args()
    # Ensure output directory exists even if not used further here
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Compose path to 'constructed' subfolder inside runs_dir
    cons = Path(args.runs_dir) / "constructed"
    if cons.exists():
        print("Constructed summary:")
        aggregate_constructed(str(cons))


if __name__ == "__main__":
    main()
