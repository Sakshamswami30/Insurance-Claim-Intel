import json
from pathlib import Path

from src.ocr import extract_text_from_pdf
from src.extract import extract_claim


GROUND_TRUTH_PATH = "data/samples/claims_ground_truth.json"
CLAIMS_DIR = "data/samples/claims"

COMPARE_FIELDS = [
    "policy_number",
    "claim_type",
    "claimant_name",
    "diagnosis",
    "provider_name",
    "incident_date",
    "claim_amount",
]


def normalize(value):
    """Basic cleanup for comparison — lowercase strings, strip whitespace."""
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, float):
        return round(value, 2)
    return value


def main():
    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        ground_truth = json.load(f)

    results = {field: {"correct": 0, "total": 0} for field in COMPARE_FIELDS}
    failures = []

    for record in ground_truth:
        filename = record["filename"]
        pdf_path = Path(CLAIMS_DIR) / filename
        print(f"Processing {filename}...")

        text = extract_text_from_pdf(str(pdf_path))
        extracted = extract_claim(text)
        extracted_dict = extracted.model_dump()

        for field in COMPARE_FIELDS:
            expected = normalize(record.get(field))
            got = normalize(extracted_dict.get(field))
            results[field]["total"] += 1

            if expected == got:
                results[field]["correct"] += 1
            else:
                failures.append({
                    "file": filename,
                    "field": field,
                    "expected": expected,
                    "got": got,
                })

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    for field in COMPARE_FIELDS:
        r = results[field]
        pct = 100 * r["correct"] / r["total"] if r["total"] else 0
        print(f"{field:20s} {r['correct']:3d}/{r['total']:3d}  ({pct:5.1f}%)")

    print("\n" + "=" * 50)
    print("SAMPLE FAILURES (first 10)")
    print("=" * 50)
    for f in failures[:10]:
        print(f"{f['file']} | {f['field']}")
        print(f"  expected: {f['expected']!r}")
        print(f"  got:      {f['got']!r}")


if __name__ == "__main__":
    main()