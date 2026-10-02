from typing import Literal
from pydantic import BaseModel, Field

REVIEW_CONFIDENCE_THRESHOLD = 70

class ContextPage(BaseModel):
    source: str
    page: int

class Candidate(BaseModel):
    family: str
    confidence: int = Field(ge=0, le=100)
    why: str

class ProvenanceEntry(BaseModel):
    file: str
    path: str
    source: str
    page: int
    excerpt: str
    confidence: int = Field(ge=0, le=100)
    needs_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)

class ProvenanceData(BaseModel):
    version: int = 1
    status: Literal["complete", "incomplete", "unknown_family"]
    pack_id: str
    model: str
    generated_at: str
    reason: str | None = None
    candidates: list[Candidate] = Field(default_factory=list)
    context_pages: list[ContextPage] = Field(default_factory=list)
    entries: list[ProvenanceEntry] = Field(default_factory=list)
