import json
import time
from pathlib import Path

from groq import RateLimitError

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


def extract_with_retry(text: str, max_attempts: int = 5):
    """Retry extraction on rate limit, respecting Groq's suggested wait."""
    for attempt in range(max_attempts):
        try:
            return extract_claim(text)
        except RateLimitError as e:
            wait = 30
            msg = str(e)
            if "try again in" in msg:
                try:
                    tail = msg.split("try again in")[1].split("s")[0].strip()
                    wait = float(tail) + 2
                except Exception:
                    pass
            print(f"    rate limited, waiting {wait:.0f}s (attempt {attempt + 1}/{max_attempts})")
            time.sleep(wait)
    raise RuntimeError("Exceeded max retries on rate limit")


def evaluate_batch(truth_path: str, claims_dir: str, label: str):
    with open(truth_path, encoding="utf-8") as f:
        ground_truth = json.load(f)

    checkpoint_path = Path(f"data/samples/eval_checkpoint_{label.lower()}.json")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing checkpoint if present
    completed = {}
    if checkpoint_path.exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            completed = json.load(f)
        print(f"[{label}] Resuming from checkpoint: {len(completed)} documents already done")

    results = {field: {"correct": 0, "total": 0} for field in COMPARE_FIELDS}
    failures = []

    for record in ground_truth:
        filename = record["filename"]

        if filename in completed:
            # Replay prior result from checkpoint
            prior = completed[filename]
            for field in COMPARE_FIELDS:
                results[field]["total"] += 1
                if prior["extracted"].get(field) == prior["expected"].get(field):
                    results[field]["correct"] += 1
                else:
                    failures.append({
                        "file": filename,
                        "field": field,
                        "expected": prior["expected"].get(field),
                        "got": prior["extracted"].get(field),
                    })
            continue

        print(f"[{label}] Processing {filename}...")
        time.sleep(2)

        pdf_path = Path(claims_dir) / filename
        text = extract_text_from_pdf(str(pdf_path))
        extracted = extract_with_retry(text)
        extracted_dict = extracted.model_dump()

        expected_norm = {field: normalize(record.get(field)) for field in COMPARE_FIELDS}
        got_norm = {field: normalize(extracted_dict.get(field)) for field in COMPARE_FIELDS}

        for field in COMPARE_FIELDS:
            results[field]["total"] += 1
            if expected_norm[field] == got_norm[field]:
                results[field]["correct"] += 1
            else:
                failures.append({
                    "file": filename,
                    "field": field,
                    "expected": expected_norm[field],
                    "got": got_norm[field],
                })

        # Save checkpoint after each document
        completed[filename] = {
            "expected": expected_norm,
            "extracted": got_norm,
        }
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(completed, f, indent=2)

    # Print summary
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