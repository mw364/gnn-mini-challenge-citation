# starter_code/baseline.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA_DIR = os.path.join(ROOT, "data")
SUB_DIR = os.path.join(ROOT, "submissions")
os.makedirs(SUB_DIR, exist_ok=True)

def main():
    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")

    train = pd.read_csv(train_path)
    if "target" not in train.columns:
        raise ValueError("train.csv must contain a 'target' column.")

    y = train["target"]
    X = train.drop(columns=["target"])
    if "id" in X.columns:
        X = X.drop(columns=["id"])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    score = f1_score(y_val, y_pred, average="macro")
    print(f"Validation Macro-F1: {score:.4f}")

    test = pd.read_csv(test_path)
    test_ids = test["id"].copy() if "id" in test.columns else pd.Series(range(len(test)))
    X_test = test.drop(columns=["id"]) if "id" in test.columns else test

    test_preds = clf.predict(X_test)
    out = pd.DataFrame({"id": test_ids, "target": test_preds})
    out_path = os.path.join(SUB_DIR, "sample_submission.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
