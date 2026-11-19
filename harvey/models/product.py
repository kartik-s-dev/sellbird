"""Multi-product data model."""

from pydantic import BaseModel, Field, field_validator


class Product(BaseModel):
    id: str
    name: str
    description: str = ""
    pricing: str = ""

    key_benefits: list[str] = Field(default_factory=list)
    objection_responses: dict[str, str] = Field(default_factory=dict)

    offer: dict = Field(default_factory=dict)
    icp: dict = Field(default_factory=dict)
    business_context: str = ""
    qualification_rules: dict = Field(default_factory=dict)
    sales_context: str = ""

    is_active: bool = True

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Product name cannot be empty")
        return value