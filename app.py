import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import streamlit as st
import xgboost as xgb
from dotenv import load_dotenv

from src.ocr import extract_text_from_pdf
from src.extract import extract_claim
from src.risk_labeler import PROVIDER_RISK


# ---------- Config ----------

MAX_FILE_SIZE_MB = 10
HIGH_RISK_THRESHOLD = 0.60
SAMPLE_RESPONSE_PATH = "sample_response.json"
GITHUB_URL = "https://github.com/Sakshamswami30/Insurance-Claim-Intel"

FEATURE_EXPLANATIONS = {
    "claim_amount": "The claimed amount. Larger claims are statistically more likely to require review.",
    "amount_missing": "Flagged when the claim amount couldn't be extracted from the document.",
    "is_property": "Property claims are historically more fraud-prone than other claim types.",
    "is_auto": "Auto claims are historically more fraud-prone than other claim types.",
    "is_medical": "Medical claims are less fraud-prone — hospitals generate documented records.",
    "is_travel": "Travel claims have moderate fraud risk.",
    "days_gap": "Days between the incident and the claim filing. Faster filing can indicate lower scrutiny.",
    "gap_missing": "Flagged when filing date information is unavailable.",
    "provider_risk": "A weight assigned to the service provider based on historical patterns.",
    "policy_number_missing": "Flagged when the policy number couldn't be extracted.",
    "claimant_name_missing": "Flagged when the claimant's name couldn't be extracted.",
    "diagnosis_missing": "Flagged when the diagnosis/claim reason couldn't be extracted.",
    "provider_name_missing": "Flagged when the provider's name couldn't be extracted.",
    "incident_date_missing": "Flagged when the incident date couldn't be extracted.",
}

load_dotenv()

st.set_page_config(
    page_title="Insurance Claim Intelligence",
    page_icon="◈",
    layout="wide",
)


# ---------- Theme ----------

CUSTOM_CSS = """
<style>
    .stApp { background-color: #0A0E14; }
    section.main > div { padding-top: 2rem; max-width: 1100px; }

    h1, h2, h3, p, span, div, label { color: #E6EDF3; }

    .app-title {
        color: #22D3EE;
        font-size: 30px;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 6px;
    }
    .app-sub {
        color: #8B98A9;
        font-size: 15px;
        margin-bottom: 12px;
    }
    .privacy-note {
        color: #6B7887;
        font-size: 12.5px;
        margin-bottom: 22px;
    }
    .section-title {
        color: #E6EDF3;
        font-size: 17px;
        font-weight: 600;
        margin: 26px 0 12px 0;
    }
    .card {
        background: #111823;
        border: 1px solid #1E2A3A;
        border-radius: 12px;
        padding: 22px 24px;
        margin-bottom: 18px;
    }
    .field-row {
        display: flex;
        justify-content: space-between;
        padding: 10px 0;
        border-bottom: 1px solid #1A2333;
    }
    .field-row:last-child { border-bottom: none; }
    .field-label { color: #8B98A9; font-size: 13.5px; }
    .field-value { color: #E6EDF3; font-size: 13.5px; font-weight: 500; }

    .risk-score { font-size: 46px; font-weight: 700; letter-spacing: -0.03em; margin: 4px 0; }
    .risk-high  { color: #EF4444; }
    .risk-mid   { color: #F59E0B; }
    .risk-low   { color: #10B981; }
    .risk-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.04em;
        margin-top: 8px;
    }
    .badge-high { background: rgba(239, 68, 68, 0.15); color: #EF4444; }
    .badge-mid  { background: rgba(245, 158, 11, 0.15); color: #F59E0B; }
    .badge-low  { background: rgba(16, 185, 129, 0.15); color: #10B981; }

    .reason-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 0;
        border-bottom: 1px solid #1A2333;
    }
    .reason-row:last-child { border-bottom: none; }
    .reason-feature { color: #E6EDF3; font-size: 13.5px; min-width: 160px; }
    .reason-up { color: #EF4444; font-size: 13.5px; font-weight: 600; }
    .reason-down { color: #10B981; font-size: 13.5px; font-weight: 600; }

    .bar-wrap {
        flex: 1;
        margin: 0 18px;
        height: 6px;
        background: #0A0E14;
        border-radius: 3px;
        overflow: hidden;
        position: relative;
    }
    .bar {
        height: 100%;
        border-radius: 3px;
    }
    .bar-up { background: #EF4444; }
    .bar-down { background: #10B981; }

    .footer {
        color: #6B7887;
        font-size: 12px;
        text-align: center;
        margin-top: 60px;
        padding-top: 24px;
        border-top: 1px solid #1A2333;
        line-height: 1.8;
    }
    .footer a { color: #22D3EE; text-decoration: none; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------- Load model once ----------

@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model("models/risk_classifier.json")
    with open("models/feature_names.json") as f:
        feature_names = json.load(f)
    explainer = shap.TreeExplainer(model)
    return model, feature_names, explainer


# ---------- Pipeline ----------

def build_features(extracted: dict, feature_names: list) -> pd.DataFrame:
    row = {}
    row["claim_amount"] = extracted.get("claim_amount") or 0
    row["amount_missing"] = int(extracted.get("claim_amount") is None)

    ct = (extracted.get("claim_type") or "").strip().lower()
    row["is_property"] = int(ct == "property")
    row["is_auto"] = int(ct == "auto")
    row["is_medical"] = int(ct == "medical")
    row["is_travel"] = int(ct == "travel")

    row["days_gap"] = 30
    row["gap_missing"] = 1

    row["provider_risk"] = PROVIDER_RISK.get(extracted.get("provider_name"), 0.30)

    for field in ["policy_number", "claimant_name", "diagnosis", "provider_name", "incident_date"]:
        row[f"{field}_missing"] = int(extracted.get(field) is None)

    return pd.DataFrame([row])[feature_names]


def explain(feature_row, feature_names, explainer, top_n=4):
    shap_values = explainer.shap_values(feature_row)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    row_shap = shap_values[0]
    order = np.argsort(np.abs(row_shap))[::-1][:top_n]
    return [
        {
            "feature": feature_names[j],
            "direction": "increases_risk" if row_shap[j] > 0 else "decreases_risk",
            "weight": round(float(row_shap[j]), 4),
        }
        for j in order
    ]


def run_pipeline(pdf_path: str, filename: str):
    model, feature_names, explainer = load_model()

    with st.status("Analyzing claim…", expanded=True) as status:
        st.write("⬡ Reading PDF…")
        text = extract_text_from_pdf(pdf_path)
        if not text or len(text.strip()) < 20:
            status.update(label="Could not read this PDF", state="error")
            st.error("This PDF looks empty or is a scanned image. Try a text-based PDF.")
            return None

        st.write("⬡ Extracting fields with the LLM…")
        extracted = extract_claim(text).model_dump()

        st.write("⬡ Scoring risk with the classifier…")
        features = build_features(extracted, feature_names)
        risk_score = float(model.predict_proba(features)[0, 1])

        st.write("⬡ Generating explanation with SHAP…")
        reasons = explain(features, feature_names, explainer)

        status.update(label="Analysis complete.", state="complete", expanded=False)

    return {
        "filename": filename,
        "extracted_fields": extracted,
        "risk_score": round(risk_score, 4),
        "is_high_risk": risk_score > HIGH_RISK_THRESHOLD,
        "top_reasons": reasons,
    }


# ---------- Rendering ----------

def render_results(result: dict):
    fields = result["extracted_fields"]

    st.markdown('<div class="section-title">Extracted fields</div>', unsafe_allow_html=True)
    rows_html = ""
    for k, v in fields.items():
        label = k.replace("_", " ").title()
        if v is None:
            display = "—"
        elif k == "claim_amount":
            display = f"₹ {v:,.2f}"
        else:
            display = str(v)
        rows_html += (
            f'<div class="field-row"><span class="field-label">{label}</span>'
            f'<span class="field-value">{display}</span></div>'
        )
    st.markdown(f'<div class="card">{rows_html}</div>', unsafe_allow_html=True)

    score = result["risk_score"]
    if score >= 0.75:
        cls, badge, label = "risk-high", "badge-high", "HIGH RISK"
    elif score >= HIGH_RISK_THRESHOLD:
        cls, badge, label = "risk-mid", "badge-mid", "MEDIUM RISK"
    else:
        cls, badge, label = "risk-low", "badge-low", "LOW RISK"

    st.markdown('<div class="section-title">Risk assessment</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="card">'
        f'<div class="{cls} risk-score">{score:.4f}</div>'
        f'<span class="risk-badge {badge}">{label}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Why this score</div>', unsafe_allow_html=True)

    max_abs = max(abs(r["weight"]) for r in result["top_reasons"]) or 1.0

    reasons_html = ""
    for r in result["top_reasons"]:
        arrow = "↑" if r["direction"] == "increases_risk" else "↓"
        cls = "reason-up" if r["direction"] == "increases_risk" else "reason-down"
        bar_cls = "bar-up" if r["direction"] == "increases_risk" else "bar-down"
        bar_width = min(abs(r["weight"]) / max_abs * 100, 100)
        tooltip = FEATURE_EXPLANATIONS.get(r["feature"], "")
        reasons_html += (
            f'<div class="reason-row" title="{tooltip}">'
            f'<span class="reason-feature">{arrow} {r["feature"]}</span>'
            f'<div class="bar-wrap"><div class="bar {bar_cls}" style="width:{bar_width:.0f}%"></div></div>'
            f'<span class="{cls}">{r["weight"]:+.4f}</span>'
            f'</div>'
        )
    st.markdown(f'<div class="card">{reasons_html}</div>', unsafe_allow_html=True)

    st.download_button(
        label="Download analysis (JSON)",
        data=json.dumps(result, indent=2),
        file_name=f"analysis_{result['filename'].replace('.pdf','')}.json",
        mime="application/json",
    )


# ---------- Main ----------

st.markdown('<div class="app-title">◈ Insurance Claim Intelligence</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-sub">Upload a claim PDF. Get extracted fields, a risk score, and a plain-English explanation.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="privacy-note">🔒 Your file is processed locally. Only the extracted text is sent to the LLM for field extraction.</div>',
    unsafe_allow_html=True,
)

with st.expander("How this works"):
    st.markdown("""
**1. PDF upload → text extraction**
Your file is read locally using `pypdf`. The PDF itself never leaves your machine.

**2. Field extraction (LLM)**
The extracted text is sent to Groq's API (Llama 3.3 70B), which returns structured JSON validated against a fixed schema. If the model returns malformed output, the pipeline retries automatically.

**3. Risk classification (XGBoost)**
The extracted fields are turned into 15 numeric features and scored by a gradient-boosted classifier trained on 400 labeled claims. ROC-AUC: 0.95.

**4. Explainability (SHAP)**
For every prediction, SHAP computes how much each feature pushed the score up or down. That's what appears under **"Why this score."**

The model is not a fraud detector — it's a demonstration of a pipeline. In production, the labels would come from real investigations, not a synthetic rule.
""")

uploaded = st.file_uploader(
    "Drop a claim PDF here",
    type=["pdf"],
    accept_multiple_files=False,
    label_visibility="collapsed",
)

col1, col2 = st.columns([1, 4])
with col1:
    sample_clicked = st.button("Try a sample claim")

if sample_clicked:
    if Path(SAMPLE_RESPONSE_PATH).exists():
        with open(SAMPLE_RESPONSE_PATH) as f:
            render_results(json.load(f))
    else:
        st.error("Sample response file not found.")

elif uploaded is not None:
    size_mb = uploaded.size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        st.error(f"File is {size_mb:.1f} MB. Maximum allowed is {MAX_FILE_SIZE_MB} MB.")
    else:
        safe_name = "".join(c for c in uploaded.name if c.isalnum() or c in "._- ").strip() or "upload.pdf"

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        try:
            result = run_pipeline(tmp_path, safe_name)
            if result:
                render_results(result)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

st.markdown(
    f'<div class="footer">'
    f'Built with Python 3.11 · Groq (Llama 3.3) · XGBoost · SHAP · Streamlit<br>'
    f'Extraction accuracy: 100% clean / 95.7% messy · Model ROC-AUC: 0.95 · '
    f'<a href="{GITHUB_URL}" target="_blank">View source on GitHub</a>'
    f'</div>',
    unsafe_allow_html=True,
)