from pydantic import BaseModel, Field
from typing import Optional


class ClaimExtraction(BaseModel):
    policy_number: Optional[str] = Field(
        default=None,
        description="Policy number as written on the document"
    )
    claim_type: Optional[str] = Field(
        default=None,
        description="One of: medical, auto, property, travel, life"
    )
    diagnosis: Optional[str] = Field(
        default=None,
        description="Diagnosis or reason for claim"
    )
    claim_amount: Optional[float] = Field(
        default=None,
        description="Claimed amount as a number, without currency symbol"
    )
    incident_date: Optional[str] = Field(
        default=None,
        description="Date of incident in YYYY-MM-DD format"
    )
    provider_name: Optional[str] = Field(
        default=None,
        description="Name of hospital, garage, or service provider"
    )