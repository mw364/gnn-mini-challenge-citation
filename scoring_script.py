# scoring_script.py
import sys
import pandas as pd
from sklearn.metrics import f1_score

def main():
    if len(sys.argv) < 2:
        print("Usage: python scoring_script.py submissions/your_file.csv")
        sys.exit(1)

    submission_file = sys.argv[1]
    sub = pd.read_csv(submission_file)
    truth = pd.read_csv("data/test_labels.csv")

    required_cols = {"id", "target"}
    if not required_cols.issubset(sub.columns):
        raise ValueError("Submission must have columns: id,target")
    if not required_cols.issubset(truth.columns):
        raise ValueError("test_labels.csv must have columns: id,target")

    merged = truth.merge(sub[["id", "target"]], on="id", how="inner", suffixes=("_true", "_pred"))

    if len(merged) != len(truth):
        missing = set(truth["id"]) - set(sub["id"])
        raise ValueError(f"Submission missing {len(missing)} ids. Example missing id: {next(iter(missing)) if missing else 'N/A'}")

    score = f1_score(merged["target_true"], merged["target_pred"], average="macro")
    print(f"Submission Macro-F1: {score:.4f}")

if __name__ == "__main__":
    main()
