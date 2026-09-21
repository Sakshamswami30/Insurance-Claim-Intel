import json
from pathlib import Path

from src.ocr import extract_text_from_pdf
from src.extract import extract_claim


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
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, float):
        return round(value, 2)
    return value


def evaluate_batch(truth_path: str, claims_dir: str, label: str):
    with open(truth_path, encoding="utf-8") as f:
        ground_truth = json.load(f)

    results = {field: {"correct": 0, "total": 0} for field in COMPARE_FIELDS}
    failures = []

    for record in ground_truth:
        filename = record["filename"]
        pdf_path = Path(claims_dir) / filename
        print(f"[{label}] Processing {filename}...")

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

    print(f"\n{'=' * 50}")
    print(f"RESULTS — {label}")
    print(f"{'=' * 50}")
    for field in COMPARE_FIELDS:
        r = results[field]
        pct = 100 * r["correct"] / r["total"] if r["total"] else 0
        print(f"{field:20s} {r['correct']:3d}/{r['total']:3d}  ({pct:5.1f}%)")

    total_correct = sum(r["correct"] for r in results.values())
    total_fields = sum(r["total"] for r in results.values())
    overall = 100 * total_correct / total_fields if total_fields else 0
    print(f"\nOVERALL: {total_correct}/{total_fields} ({overall:.1f}%)")

    if failures:
        print(f"\nSAMPLE FAILURES ({label}, first 10)")
        for f in failures[:10]:
            print(f"  {f['file']} | {f['field']}")
            print(f"    expected: {f['expected']!r}")
            print(f"    got:      {f['got']!r}")

    return results, failures


def main():
    print("Running CLEAN batch...")
    evaluate_batch(
        truth_path="data/samples/claims_ground_truth_clean.json",
        claims_dir="data/samples/claims_clean",
        label="CLEAN",
    )

    print("\n" + "#" * 50 + "\n")

    print("Running MESSY batch...")
    evaluate_batch(
        truth_path="data/samples/claims_ground_truth_messy.json",
        claims_dir="data/samples/claims_messy",
        label="MESSY",
    )


if __name__ == "__main__":
    main()