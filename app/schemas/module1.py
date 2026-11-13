"""
Module 1 (Script Intake and Answer Evidence) Pydantic v2 Schemas.
Aligned with SRS Section 4.3 & 4.4 (ExternalInterfaces.docx).
"""
from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class RectSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: float = Field(..., ge=0, le=100)
    y: float = Field(..., ge=0, le=100)
    width: float = Field(..., gt=0, le=100)
    height: float = Field(..., gt=0, le=100)


class EvidenceSpanSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    span_id: str
    text: str
    page_number: int = Field(..., ge=1, le=10)
    region_id: str
    bbox: RectSchema


class AnswerReadySchema(BaseModel):
    """
    AnswerReady (Module 1 -> Module 2) immutable contract.
    Module 2 rejects any record that lacks explicit approval_status == 'APPROVED'.
    """
    model_config = ConfigDict(extra="forbid")

    contract: Literal["AnswerReady"] = "AnswerReady"
    schema_version: Literal["1.0"] = "1.0"
    assessment_id: str
    script_id: str
    pseudonymous_script_code: str
    question_id: str
    transcript_version: int = Field(..., ge=1)
    approved_text: str
    evidence: list[EvidenceSpanSchema]
    approval_status: Literal["PENDING", "APPROVED", "SUPERSEDED"]
    routing_status: Literal[
        "NOT_EXTRACTED",
        "NEEDS_REVIEW",
        "READY_FOR_AI",
        "MANUAL_MARKING",
        "NOT_ATTEMPTED",
    ]


class McqPageImagePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pageNumber: int = Field(..., ge=1, le=10)
    mimeType: str = "image/jpeg"
    base64Data: str


class McqExtractionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scriptId: Optional[str] = None
    images: list[McqPageImagePayload] = Field(..., min_length=1, max_length=10)


class McqItemSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    questionNumber: int = Field(..., ge=1)
    selectedOption: Literal[
        "A", "B", "C", "D", "E",
        "1", "2", "3", "4", "5",
        "BLANK", "MULTIPLE",
    ]
    markedOptions: Optional[list[str]] = None
    confidence: float | Literal["unknown"]
    note: Optional[str] = None
    edited: Optional[bool] = None


class McqExtractionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    model: Optional[str] = None
    optionStyle: Optional[Literal["ALPHA", "NUMERIC"]] = None
    optionLabels: Optional[list[str]] = None
    items: list[McqItemSchema] = Field(default_factory=list)
    error: Optional[str] = None

