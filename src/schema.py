"""Pydantic models for input validation."""

from pydantic import BaseModel, Field


class SalaryInput(BaseModel):
    """Input model for salary prediction."""

    country: str = Field(..., description="Developer's country")
    years_code: float = Field(
        ...,
        ge=0,
        description="Including any education, how many years have you been coding in total?",
    )
    education_level: str = Field(..., description="Education level")
    dev_type: str = Field(..., description="Developer type")
    industry: str = Field(..., description="Industry the developer works in")
    age: str = Field(..., description="Developer's age range")

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "country": "United States",
                "years_code": 5.0,
                "education_level": "Bachelor's degree",
                "dev_type": "Developer, back-end",
                "industry": "Software Development",
                "age": "25-34 years old",
            }
        }
