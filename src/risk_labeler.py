import json
import random
from datetime import datetime
from pathlib import Path


# Provider risk weights — some providers appear more often in suspicious claims.
# These are hand-picked for the synthetic dataset; in reality they'd be learned.
PROVIDER_RISK = {
    "AutoFix Garage": 0.80,
    "QuickCare Diagnostics": 0.60,
    "City General Hospital": 0.30,
    "Apollo Clinic": 0.30,
    "Manipal Hospital": 0.30,
    "Fortis Healthcare": 0.20,
    "Max Super Speciality": 0.20,
}

# Claims from these types are historically more fraud-prone
HIGH_RISK_TYPES = {"Property", "Auto"}

# Threshold above which a claim is labeled high-risk
RISK_THRESHOLD = 0.60


def days_between(incident: str, filed: str) -> int:
    """Days between incident and filing. Lower = faster filing = higher risk."""
    d1 = datetime.strptime(incident, "%Y-%m-%d")
    d2 = datetime.strptime(filed, "%Y-%m-%d")
    return (d2 - d1).days


def compute_risk_score(record: dict, max_amount: float) -> float:
    """Compute a risk score between 0 and 1 for one claim."""
    # 1. Normalized claim amount (0–1)
    amount = record.get("claim_amount") or 0
    amount_norm = min(amount / max_amount, 1.0)

    # 2. Claim type signal
    type_signal = 1.0 if record.get("claim_type") in HIGH_RISK_TYPES else 0.0

    # 3. Short gap between incident and filing = higher risk
    #    Normalized against 60 days (our generator uses 1–60 day gaps)
    gap = 60  # default if dates are missing
    try:
        gap = days_between(record["incident_date"], record["filed_date"])
    except (KeyError, TypeError):
        pass
    gap_norm = max(0.0, 1.0 - (gap / 60.0))

    # 4. Provider risk weight
    provider_signal = PROVIDER_RISK.get(record.get("provider_name"), 0.30)

    # Combine
    score = (
        0.40 * amount_norm
        + 0.30 * type_signal
        + 0.20 * gap_norm
        + 0.10 * provider_signal
    )

    # Add noise so the model can't perfectly memorize the rule
    score += random.uniform(-0.10, 0.10)

    return round(max(0.0, min(score, 1.0)), 4)


def enrich_file(path: str) -> None:
    """Add risk_score and is_high_risk to every record."""
    p = Path(path)
    with open(p, encoding="utf-8") as f:
        records = json.load(f)

    max_amount = max((r.get("claim_amount") or 0) for r in records)

    for record in records:
        score = compute_risk_score(record, max_amount)
        record["risk_score"] = score
        record["is_high_risk"] = score > RISK_THRESHOLD

    with open(p, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    high_risk_count = sum(1 for r in records if r["is_high_risk"])
    print(f"{p.name}: {high_risk_count}/{len(records)} high-risk")


enrich_file("data/samples/claims_ground_truth_clean.json")
enrich_file("data/samples/claims_ground_truth_messy.json")