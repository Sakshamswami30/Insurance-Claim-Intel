import json
import random
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


# Value pools — we draw randomly from these
POLICY_PREFIXES = ["POL", "CLM", "INS"]
FIRST_NAMES = ["Rahul", "Priya", "Amit", "Sneha", "Vikram", "Neha", "Arjun", "Kavya", "Rohan", "Anjali"]
LAST_NAMES = ["Sharma", "Patel", "Singh", "Verma", "Iyer", "Reddy", "Khan", "Gupta", "Nair", "Mehta"]
CLAIM_TYPES = ["Medical", "Auto", "Property", "Travel"]
DIAGNOSES = [
    "Type 2 Diabetes",
    "Hypertension",
    "Fracture - left femur",
    "Acute appendicitis",
    "Viral fever",
    "Road accident - minor injury",
    "Water damage - kitchen",
    "Lost baggage",
]
PROVIDERS = [
    "City General Hospital",
    "Apollo Clinic",
    "Fortis Healthcare",
    "Max Super Speciality",
    "Manipal Hospital",
    "AutoFix Garage",
    "QuickCare Diagnostics",
]


def random_policy_number() -> str:
    prefix = random.choice(POLICY_PREFIXES)
    number = random.randint(10000, 99999)
    return f"{prefix}-{number}"


def random_name() -> str:
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def random_date() -> str:
    year = random.choice([2024, 2025, 2026])
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    return f"{year:04d}-{month:02d}-{day:02d}"


def random_amount() -> float:
    return round(random.uniform(1000, 250000), 2)


def generate_claim_data() -> dict:
    """Return a dict with all the values for one claim."""
    return {
        "policy_number": random_policy_number(),
        "claim_type": random.choice(CLAIM_TYPES),
        "claimant_name": random_name(),
        "diagnosis": random.choice(DIAGNOSES),
        "provider_name": random.choice(PROVIDERS),
        "incident_date": random_date(),
        "claim_amount": random_amount(),
    }


def draw_claim_pdf(output_path: str, data: dict) -> None:
    """Write one claim PDF using the given values."""
    c = canvas.Canvas(output_path, pagesize=A4)
    _, height = A4
    y = height - 80

    c.setFont("Helvetica-Bold", 16)
    c.drawString(80, y, "INSURANCE CLAIM FORM")
    y -= 40

    c.setFont("Helvetica", 12)
    fields = [
        ("Policy Number", data["policy_number"]),
        ("Claim Type", data["claim_type"]),
        ("Claimant Name", data["claimant_name"]),
        ("Diagnosis", data["diagnosis"]),
        ("Treatment Provider", data["provider_name"]),
        ("Incident Date", data["incident_date"]),
        ("Claimed Amount", f"INR {data['claim_amount']:.2f}"),
    ]
    for label, value in fields:
        c.drawString(80, y, f"{label}: {value}")
        y -= 25

    c.save()


def generate_batch(count: int, output_dir: str, truth_path: str) -> None:
    """Generate `count` claim PDFs and write a ground-truth JSON."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ground_truth = []
    for i in range(1, count + 1):
        data = generate_claim_data()
        filename = f"claim_{i:03d}.pdf"
        pdf_path = out / filename
        draw_claim_pdf(str(pdf_path), data)

        record = dict(data)
        record["filename"] = filename
        ground_truth.append(record)

    with open(truth_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Generated {count} PDFs in {output_dir}")
    print(f"Ground truth written to {truth_path}")