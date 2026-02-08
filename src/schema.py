"""Pydantic models for input validation."""

from pydantic import BaseModel, Field


class SalaryInput(BaseModel):
    """Input model for salary prediction."""

    country: str = Field(..., description="Developer's country")
    years_code_pro: float = Field(
        ..., ge=0, description="Years of professional coding experience"
    )
    education_level: str = Field(..., description="Education level")

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "country": "United States",
                "years_code_pro": 5.0,
                "education_level": "Bachelor's degree",
            }
        }
