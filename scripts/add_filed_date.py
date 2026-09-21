import json
import random
from datetime import datetime, timedelta
from pathlib import Path


def random_incident_datetime() -> datetime:
    year = random.choice([2024, 2025, 2026])
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    return datetime(year, month, day)


def enrich_file(path: str) -> None:
    p = Path(path)
    with open(p, encoding="utf-8") as f:
        records = json.load(f)

    for record in records:
        if "filed_date" in record:
            continue

        incident_str = record.get("incident_date")
        if incident_str:
            incident = datetime.strptime(incident_str, "%Y-%m-%d")
        else:
            # incident_date was a missing field in the PDF — pick a fallback
            incident = random_incident_datetime()

        days_gap = random.randint(1, 60)
        filed = incident + timedelta(days=days_gap)
        record["filed_date"] = filed.strftime("%Y-%m-%d")

    with open(p, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"Updated {len(records)} records in {p}")


enrich_file("data/samples/claims_ground_truth_clean.json")
enrich_file("data/samples/claims_ground_truth_messy.json")