import json
import random
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from src.doc_generator import (
    generate_claim_data,
    POLICY_PREFIXES,
    FIRST_NAMES,
    LAST_NAMES,
    CLAIM_TYPES,
    DIAGNOSES,
    PROVIDERS,
)


# Alternate labels for the same field — a real source of extraction failures
LABEL_VARIANTS = {
    "policy_number": ["Policy Number", "Policy No", "Policy #"],
    "claim_type": ["Claim Type", "Type of Claim", "Claim Category"],
    "claimant_name": ["Claimant Name", "Claimant", "Name of Claimant"],
    "diagnosis": ["Diagnosis", "Medical Diagnosis", "Condition"],
    "provider_name": ["Treatment Provider", "Provider", "Hospital / Provider"],
    "incident_date": ["Incident Date", "Date of Incident", "Event Date"],
    "claim_amount": ["Claimed Amount", "Amount Claimed", "Claim Amount"],
}

NOISE_HEADERS = [
    "National Insurance Corporation - Claim Processing Division",
    "Confidential - For Internal Use Only",
    "Form 42-B: Standard Claim Submission",
    "Claim Intake Document - Retain for Records",
]

NOISE_FOOTERS = [
    "This document was generated electronically. No signature required.",
    "For queries, contact claims@insurer.example",
    "Reference: CLAIM-PROC-2026-v3",
    "Page 1 of 1",
]


def choose_label(field: str) -> str:
    return random.choice(LABEL_VARIANTS[field])


def draw_messy_claim_pdf(output_path: str, data: dict, missing_field: str | None) -> None:
    """Write one messy claim PDF."""
    c = canvas.Canvas(output_path, pagesize=A4)
    _, height = A4
    y = height - 60

    # Noise header
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(80, y, random.choice(NOISE_HEADERS))
    y -= 40

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(80, y, "INSURANCE CLAIM FORM")
    y -= 40

    # Build the field list, skip the missing one
    field_keys = [k for k in data.keys() if k != "filename"]
    if missing_field and missing_field in field_keys:
        field_keys.remove(missing_field)

    # Shuffle order
    random.shuffle(field_keys)

    c.setFont("Helvetica", 12)
    for key in field_keys:
        label = choose_label(key)
        value = data[key]
        if key == "claim_amount":
            display = f"INR {value:.2f}"
        else:
            display = str(value)
        c.drawString(80, y, f"{label}: {display}")
        y -= 25

    # Noise footer
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(80, 80, random.choice(NOISE_FOOTERS))

    c.save()


def generate_messy_batch(count: int, output_dir: str, truth_path: str, missing_rate: float = 0.2) -> None:
    """Generate `count` messy claim PDFs and write ground truth."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ground_truth = []
    for i in range(1, count + 1):
        data = generate_claim_data()

        # Decide whether this document is missing a field
        missing_field = None
        if random.random() < missing_rate:
            missing_field = random.choice(list(data.keys()))

        filename = f"claim_messy_{i:03d}.pdf"
        pdf_path = out / filename
        draw_messy_claim_pdf(str(pdf_path), data, missing_field)

        record = dict(data)
        record["filename"] = filename
        # Ground truth reflects what was actually printed
        if missing_field:
            record[missing_field] = None
        ground_truth.append(record)

    with open(truth_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Generated {count} messy PDFs in {output_dir}")
    print(f"Ground truth written to {truth_path}")