
import json
import sys
import traceback

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

import bot_v3


LABELS = [0, 1, 2]
NAMES = ["SELL", "HOLD", "BUY"]


def main():
    print("=" * 60)
    print("COINDCX ML MODEL DIAGNOSTICS")
    print("READ-ONLY ANALYSIS - NO LIVE ORDERS")
    print("=" * 60)

    with open("config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    client = bot_v3.CoinDCX(cfg)
    ml = bot_v3.MLEngine(client)

    print("\nFetching active INR futures contracts...")
    pairs = ml.refresh_universe(force=True)

    if not pairs:
        print("ERROR: No active futures pairs found.")
        return

    print("Active pairs:", ", ".join(pairs))

    train_x = []
    train_y = []
    valid_x = []
    valid_y = []
    pair_data = {}

    for pair in pairs:
        print(f"\nPreparing {pair}...")
        result = ml._label(pair)

        if result is None:
            print("Skipped: insufficient candle data")
            continue

        features, x, y = result

        if len(x) < 180:
            print("Skipped: insufficient labelled samples")
            continue

        cut = int(len(x) * 0.80)

        # Purge 3 bars to avoid label overlap at split boundary.
        if cut <= 100 or len(x) - cut < 35:
            print("Skipped: insufficient train/validation rows")
            continue

        x_train = x.iloc[:cut - 3]
        y_train = y.iloc[:cut - 3]

        x_valid = x.iloc[cut:]
        y_valid = y.iloc[cut:]

        if y_train.nunique() < 3 or y_valid.nunique() < 3:
            print("Skipped: one or more classes missing")
            continue

        train_x.append(x_train)
        train_y.append(y_train)
        valid_x.append(x_valid)
        valid_y.append(y_valid)

        pair_data[pair] = (x_train, y_train, x_valid, y_valid)

        print(
            f"Train: {len(x_train)} | "
            f"Validation: {len(x_valid)}"
        )

    if not train_x:
        print("\nERROR: No usable datasets.")
        return

    X_train = pd.concat(train_x)
    Y_train = pd.concat(train_y)

    X_valid = pd.concat(valid_x)
    Y_valid = pd.concat(valid_y)

    if Y_train.nunique() < 3:
        print("ERROR: Training data does not contain all 3 classes.")
        return

    print("\nTraining diagnostic model...")
    print("Training rows:", len(X_train))
    print("Validation rows:", len(X_valid))

    model = HistGradientBoostingClassifier(
        max_iter=120,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1.5,
        random_state=42,
    )

    model.fit(X_train, Y_train)
    predictions = model.predict(X_valid)

    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)

    print("Accuracy:", round(accuracy_score(Y_valid, predictions), 4))
    print(
        "Balanced accuracy:",
        round(balanced_accuracy_score(Y_valid, predictions), 4),
    )

    print("\nCONFUSION MATRIX")
    print("Rows = Actual | Columns = Predicted")
    print("Order: SELL, HOLD, BUY")
    print(confusion_matrix(Y_valid, predictions, labels=LABELS))

    print("\nCLASSIFICATION REPORT")
    print(
        classification_report(
            Y_valid,
            predictions,
            labels=LABELS,
            target_names=NAMES,
            zero_division=0,
        )
    )

    print("\nPREDICTION DISTRIBUTION")
    for label, name in zip(LABELS, NAMES):
        actual_count = int((Y_valid == label).sum())
        predicted_count = int((predictions == label).sum())

        print(
            f"{name}: actual={actual_count}, "
            f"predicted={predicted_count}"
        )

    print("\nPER-PAIR BALANCED ACCURACY")
    for pair, (xtr, ytr, xv, yv) in pair_data.items():
        pair_predictions = model.predict(xv)

        score = balanced_accuracy_score(yv, pair_predictions)

        print(f"{pair}: {score:.4f}")

    print("\n" + "=" * 60)
    print("DIAGNOSTICS COMPLETED")
    print("No live orders were placed.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nDIAGNOSTIC ERROR:")
        traceback.print_exc()
        sys.exit(1)
  
