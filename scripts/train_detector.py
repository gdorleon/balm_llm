import json, random, os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from balm.detectors.classifier import build_detector_pipeline, save_detector


def load_constructed(path="data/constructed"):
    # Helper to load the dataset splits from JSONL files
    def read_split(name):
        items = []
        # Read each line as JSON and collect into a list
        with open(Path(path) / f"{name}.jsonl", encoding="utf-8") as f:
            for line in f:
                items.append(json.loads(line))
        return items

    # Load all 3 splits: biased (positive), neutral, and near_neutral (both negative)
    biased = read_split("biased")
    neutral = read_split("neutral")
    near = read_split("near_neutral")

    # For training the detector conservatively, label biased as 1 (positive),
    # neutral and near_neutral as 0 (negative)
    X = [x["prompt"] for x in biased + neutral + near]
    y = [1] * len(biased) + [0] * (len(neutral) + len(near))

    # Return a DataFrame for easier handling downstream
    return pd.DataFrame({"text": X, "label": y})


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--train", choices=["constructed"], default="constructed",
                   help="Which dataset to use for training")
    p.add_argument("--csv", type=str, default=None,
                   help="Optional CSV file with 'text' and 'label' columns instead of constructed data")
    p.add_argument("--save_path", type=str, default="data/detector.joblib",
                   help="Where to save the trained detector model")
    args = p.parse_args()

    # Load dataset either from CSV or from built-in constructed JSONL splits
    if args.csv:
        df = pd.read_csv(args.csv)
    else:
        df = load_constructed()

    # Split data into train and test sets, stratify by label to keep balanced classes
    Xtr, Xte, ytr, yte = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    # Build the pipeline with vectorizer + classifier (default BALM detector setup)
    pipe = build_detector_pipeline()

    # Train on the training set
    pipe.fit(Xtr, ytr)

    # Evaluate on the test set: precision, recall, F1-score (binary)
    pr, rc, f1, _ = precision_recall_fscore_support(yte, pipe.predict(Xte), average="binary")

    # Also calculate AUC-ROC if possible (skip if probs not available)
    try:
        auc = roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1])
    except Exception:
        auc = float("nan")

    # Print concise performance summary
    print(f"Precision {pr:.3f} Recall {rc:.3f} F1 {f1:.3f} AUROC {auc:.3f}")

    # Make sure the directory for saving detector exists
    Path(os.path.dirname(args.save_path)).mkdir(parents=True, exist_ok=True)

    # Save the trained detector pipeline for later reuse
    save_detector(pipe, args.save_path)
    print(f"Saved detector to {args.save_path}")


if __name__ == "__main__":
    main()
