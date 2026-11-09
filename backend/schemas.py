"""Pydantic models: the case record (DESIGN.md section 5) and the LLM output contracts."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Verdict = Literal["Supported", "Contradicted", "Not covered"]
Scope = Literal["in_scope", "out_of_scope"]
SourceType = Literal["image", "audio", "text"]
CaseStatus = Literal[
    "needs_input",                 # nothing readable yet
    "awaiting_text_confirmation",  # volunteer checks OCR / transcript
    "awaiting_review",             # verdicts ready for volunteer
    "needs_manual_review",         # model output invalid after retry
    "approved",
    "rejected",
    "replied",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------- case record ----------

class InputSource(BaseModel):
    type: SourceType
    file: str | None = None
    extracted_text: str = ""
    flags: list[str] = Field(default_factory=list)


class SourceOfClaim(BaseModel):
    file: str
    excerpt: str


class Evidence(BaseModel):
    source_id: str
    quote: str
    source_date: str | None = None
    quote_verified: bool = False


class Claim(BaseModel):
    claim_id: str
    claim: str
    claim_en: str
    scope: Scope = "in_scope"
    source_of_claim: SourceOfClaim
    verdict: Verdict = "Not covered"
    evidence: Evidence | None = None
    flags: list[str] = Field(default_factory=list)
    retrieved_ids: list[str] = Field(default_factory=list)


class Review(BaseModel):
    status: Literal["pending", "approved", "rejected"] = "pending"
    reviewer: str | None = None
    edits: list[dict] = Field(default_factory=list)
    decisions: dict[str, Literal["approve", "edit", "reject"]] = Field(default_factory=dict)
    note: str | None = None


class Reply(BaseModel):
    text: str | None = None
    audio_file: str | None = None
    audio_seconds: float | None = None
    flags: list[str] = Field(default_factory=list)


class Case(BaseModel):
    case_id: str
    status: CaseStatus
    created_at: str = Field(default_factory=now_iso)
    input_language: str | None = None
    input_sources: list[InputSource] = Field(default_factory=list)
    links_found: list[str] = Field(default_factory=list)
    injection_flags: list[str] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    review: Review = Field(default_factory=Review)
    reply: Reply = Field(default_factory=Reply)


# ---------- volunteer input ----------

class ReviewDecision(BaseModel):
    claim_id: str
    action: Literal["approve", "edit", "reject"]
    verdict: Verdict | None = None
    source_id: str | None = None
    quote: str | None = None


# ---------- LLM output contracts ----------

class ExtractedClaim(BaseModel):
    claim: str
    claim_en: str = ""
    scope: Scope = "in_scope"
    excerpt: str = ""
    source_index: int = 0

    @field_validator("claim")
    @classmethod
    def _not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("claim must not be empty")
        return v.strip()


class ExtractionOutput(BaseModel):
    input_language: str = "en"
    claims: list[ExtractedClaim] = Field(default_factory=list)


class VerdictOutput(BaseModel):
    verdict: Verdict
    source_id: str | None = None
    quote: str | None = None
    reason: str = ""


class ReplyOutput(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def _not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty")
        return v.strip()
