from src.extract import extract_claim

sample = """
Claim submitted under policy POL-88421.
Patient diagnosed with Type 2 Diabetes, treated at City General Hospital.
Incident date: 2026-03-15. Claimed amount: 12,450.00 INR.
"""

result = extract_claim(sample)
print(result.model_dump_json(indent=2))