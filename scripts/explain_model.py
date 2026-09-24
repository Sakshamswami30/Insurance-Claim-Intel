import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from scripts.train_risk_model import (
    load_data,
    engineer_features,
)


def load_model():
    model = xgb.XGBClassifier()
    model.load_model("models/risk_classifier.json")
    with open("models/feature_names.json") as f:
        feature_names = json.load(f)
    return model, feature_names


def main():
    print("Loading data and model...")
    df = load_data()
    X = engineer_features(df)

    model, feature_names = load_model()

    print("\nBuilding SHAP explainer...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Handle both old and new shap return formats
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # 1. Global feature importance
    print("\n" + "=" * 55)
    print("GLOBAL FEATURE IMPORTANCE (mean |SHAP|)")
    print("=" * 55)
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    for i in order:
        print(f"  {feature_names[i]:25s} {mean_abs[i]:.4f}")

    # 2. Explain a few high-risk predictions
    print("\n" + "=" * 55)
    print("SAMPLE EXPLANATIONS — HIGH-RISK CLAIMS")
    print("=" * 55)

    predictions = model.predict_proba(X)[:, 1]
    high_risk_indices = np.argsort(predictions)[::-1][:3]

    for idx in high_risk_indices:
        record = df.iloc[idx]
        print(f"\n{record['filename']}  (predicted risk: {predictions[idx]:.3f})")
        print(f"  claim_type={record['claim_type']}, amount={record['claim_amount']}, provider={record['provider_name']}")

        row_shap = shap_values[idx]
        row_order = np.argsort(np.abs(row_shap))[::-1][:5]

        print("  Top reasons:")
        for j in row_order:
            direction = "up risk" if row_shap[j] > 0 else "down risk"
            print(f"    {feature_names[j]:25s} {direction}  (SHAP={row_shap[j]:+.4f})")


if __name__ == "__main__":
    main()