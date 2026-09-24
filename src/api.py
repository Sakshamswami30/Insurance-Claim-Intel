import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from src.ocr import extract_text_from_pdf
from src.extract import extract_claim
from src.risk_labeler import PROVIDER_RISK


app = FastAPI(title="Insurance Claim Intelligence API")

# Load model once at startup
model = xgb.XGBClassifier()
model.load_model("models/risk_classifier.json")
with open("models/feature_names.json") as f:
    FEATURE_NAMES = json.load(f)
explainer = shap.TreeExplainer(model)


def build_features(extracted: dict) -> pd.DataFrame:
    """Turn extracted fields into the feature row the model expects."""
    row = {}

    amount = extracted.get("claim_amount") or 0
    row["claim_amount"] = amount
    row["amount_missing"] = int(extracted.get("claim_amount") is None)

    # Normalize claim_type to lowercase for comparison — the LLM sometimes
    # returns "Property", sometimes "property". Both should map the same way.
    ct = (extracted.get("claim_type") or "").strip().lower()
    row["is_property"] = int(ct == "property")
    row["is_auto"] = int(ct == "auto")
    row["is_medical"] = int(ct == "medical")
    row["is_travel"] = int(ct == "travel")

    # filed_date isn't on the PDF — use neutral default
    row["days_gap"] = 30
    row["gap_missing"] = 1

    row["provider_risk"] = PROVIDER_RISK.get(extracted.get("provider_name"), 0.30)

    for field in ["policy_number", "claimant_name", "diagnosis", "provider_name", "incident_date"]:
        row[f"{field}_missing"] = int(extracted.get(field) is None)

    return pd.DataFrame([row])[FEATURE_NAMES]


def explain_prediction(feature_row: pd.DataFrame, top_n: int = 4) -> list:
    """Return the top N feature contributions for a prediction."""
    shap_values = explainer.shap_values(feature_row)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    row_shap = shap_values[0]
    order = np.argsort(np.abs(row_shap))[::-1][:top_n]

    reasons = []
    for j in order:
        reasons.append({
            "feature": FEATURE_NAMES[j],
            "direction": "increases_risk" if row_shap[j] > 0 else "decreases_risk",
            "weight": round(float(row_shap[j]), 4),
        })
    return reasons


@app.get("/")
def root():
    return {"status": "ok", "service": "Insurance Claim Intelligence API"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    contents = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        text = extract_text_from_pdf(tmp_path)
        if not text or len(text.strip()) < 20:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")

        extracted = extract_claim(text).model_dump()
        features = build_features(extracted)

        risk_score = float(model.predict_proba(features)[0, 1])
        is_high_risk = risk_score > 0.60

        reasons = explain_prediction(features)

        return JSONResponse({
            "filename": file.filename,
            "extracted_fields": extracted,
            "risk_score": round(risk_score, 4),
            "is_high_risk": is_high_risk,
            "top_reasons": reasons,
        })

    finally:
        Path(tmp_path).unlink(missing_ok=True)