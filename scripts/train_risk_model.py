import json
import pandas as pd
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import xgboost as xgb


CLEAN_PATH = "data/samples/claims_ground_truth_clean.json"
MESSY_PATH = "data/samples/claims_ground_truth_messy.json"

HIGH_RISK_TYPES = {"Property", "Auto"}

PROVIDER_RISK = {
    "AutoFix Garage": 0.80,
    "QuickCare Diagnostics": 0.60,
    "City General Hospital": 0.30,
    "Apollo Clinic": 0.30,
    "Manipal Hospital": 0.30,
    "Fortis Healthcare": 0.20,
    "Max Super Speciality": 0.20,
}


def load_data() -> pd.DataFrame:
    """Load both ground truth files, tag source, return as DataFrame."""
    rows = []
    for path, source in [(CLEAN_PATH, "clean"), (MESSY_PATH, "messy")]:
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
        for r in records:
            r["source"] = source
            rows.append(r)
    return pd.DataFrame(rows)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw fields into numeric features the model can use."""
    features = pd.DataFrame(index=df.index)

    # Amount features
    features["claim_amount"] = df["claim_amount"].fillna(0)
    features["amount_missing"] = df["claim_amount"].isna().astype(int)

    # Claim type one-hot
    features["is_property"] = (df["claim_type"] == "Property").astype(int)
    features["is_auto"] = (df["claim_type"] == "Auto").astype(int)
    features["is_medical"] = (df["claim_type"] == "Medical").astype(int)
    features["is_travel"] = (df["claim_type"] == "Travel").astype(int)

    # Days between incident and filed
    def gap_days(row):
        try:
            d1 = datetime.strptime(row["incident_date"], "%Y-%m-%d")
            d2 = datetime.strptime(row["filed_date"], "%Y-%m-%d")
            return (d2 - d1).days
        except (TypeError, ValueError):
            return -1

    features["days_gap"] = df.apply(gap_days, axis=1)
    features["gap_missing"] = (features["days_gap"] == -1).astype(int)

    # Provider risk weight
    features["provider_risk"] = (
        df["provider_name"].map(PROVIDER_RISK).fillna(0.30)
    )

    # Missing-value indicators for other fields
    for field in ["policy_number", "claimant_name", "diagnosis", "provider_name", "incident_date"]:
        features[f"{field}_missing"] = df[field].isna().astype(int)

    return features


def main():
    print("Loading data...")
    df = load_data()
    print(f"Total records: {len(df)}")
    print(f"High-risk: {df['is_high_risk'].sum()} / {len(df)}")

    print("\nEngineering features...")
    X = engineer_features(df)
    y = df["is_high_risk"].astype(int)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Features: {list(X.columns)}")

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

    print("\nTraining XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    print("\nEvaluating...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
    print(f"F1:        {f1_score(y_test, y_pred):.3f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.3f}")

    # Save model and feature names
    Path("models").mkdir(exist_ok=True)
    model.save_model("models/risk_classifier.json")
    with open("models/feature_names.json", "w") as f:
        json.dump(list(X.columns), f, indent=2)
    print("\nModel saved to models/risk_classifier.json")


if __name__ == "__main__":
    main()