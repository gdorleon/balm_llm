import json, random, os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from balm.detectors.classifier import build_detector_pipeline, save_detector

def load_constructed(path="data/constructed"):
    def read_split(name):
        items = []
        with open(Path(path)/f"{name}.jsonl", encoding="utf-8") as f:
            for line in f:
                items.append(json.loads(line))
        return items
    biased = read_split("biased")
    neutral = read_split("neutral")
    near = read_split("near_neutral")
    # Label biased=1, neutral=0, near_neutral=0 for training the detector conservatively
    X = [x["prompt"] for x in biased + neutral + near]
    y = [1]*len(biased) + [0]*(len(neutral)+len(near))
    return pd.DataFrame({"text": X, "label": y})

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--train", choices=["constructed"], default="constructed")
    p.add_argument("--csv", type=str, default=None, help="optional CSV with columns text,label")
    p.add_argument("--save_path", type=str, default="data/detector.joblib")
    args = p.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
    else:
        df = load_constructed()

    Xtr, Xte, ytr, yte = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])
    pipe = build_detector_pipeline()
    pipe.fit(Xtr, ytr)
    pr, rc, f1, _ = precision_recall_fscore_support(yte, pipe.predict(Xte), average="binary")
    try:
        auc = roc_auc_score(yte, pipe.predict_proba(Xte)[:,1])
    except Exception:
        auc = float("nan")
    print(f"Precision {pr:.3f} Recall {rc:.3f} F1 {f1:.3f} AUROC {auc:.3f}")
    Path(os.path.dirname(args.save_path)).mkdir(parents=True, exist_ok=True)
    save_detector(pipe, args.save_path)
    print(f"Saved detector to {args.save_path}")

if __name__ == "__main__":
    main()
