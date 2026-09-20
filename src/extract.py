import os
import json
from dotenv import load_dotenv
from groq import Groq
from pydantic import ValidationError

from src.schemas import ClaimExtraction
from src.config import GROQ_MODEL

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])


def build_prompt(text: str) -> str:
    return f"""You are an insurance claim extraction engine.

Extract the following fields from the document text below:
- policy_number
- claim_type (one of: medical, auto, property, travel, life)
- claimant_name
- diagnosis
- claim_amount (number only, no currency symbol)
- incident_date (YYYY-MM-DD format)
- provider_name

Rules:
1. Return ONLY a valid JSON object. No explanation, no markdown, no code fences.
2. If a field is not present in the text, set it to null.
3. Do not invent information. If you are unsure, use null.

Document text:
\"\"\"
{text}
\"\"\"
"""


def parse_json(raw: str) -> dict:
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in model output")
    return json.loads(raw[start:end])


def extract_claim(text: str, max_retries: int = 2) -> ClaimExtraction:
    prompt = build_prompt(text)
    last_error = None

    for attempt in range(max_retries + 1):
        raw = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        ).choices[0].message.content

        try:
            data = parse_json(raw)
            return ClaimExtraction.model_validate(data)
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            last_error = str(e)
            print(f"  retry {attempt + 1}: {last_error}")
            prompt = (
                f"Your previous output was invalid.\n"
                f"Error: {last_error}\n"
                f"Invalid output was:\n{raw}\n\n"
                f"Return ONLY a valid JSON object with the required fields. "
                f"No explanation, no markdown."
            )

    raise RuntimeError(f"Extraction failed after {max_retries} retries: {last_error}")