"""
Pydantic models shared across modules:
  - Transcript data shapes (Phase 1)
  - Chunking data shapes (Phase 3)
  - LLM structured-output schemas (Phase 2/4/5)
  - Job/API request-response schemas (Phase 6)
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Phase 1 — transcript
# ---------------------------------------------------------------------------

class TranscriptSegment(BaseModel):
    """One caption line/segment as returned by youtube-transcript-api or Whisper."""
    text: str
    start: float  # seconds
    duration: float = 0.0

    @property
    def end(self) -> float:
        return self.start + self.duration


class TranscriptResult(BaseModel):
    video_id: str
    video_title: Optional[str] = None
    language_code: Optional[str] = None
    is_generated: bool = True
    source: str = "captions"  # "captions" | "whisper"
    segments: List[TranscriptSegment]

    @property
    def full_text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())

    @property
    def duration_seconds(self) -> float:
        if not self.segments:
            return 0.0
        last = self.segments[-1]
        return last.end


# ---------------------------------------------------------------------------
# Phase 3 — chunking
# ---------------------------------------------------------------------------

class TranscriptChunk(BaseModel):
    index: int
    text: str
    start_time: float
    end_time: float

    @property
    def word_count(self) -> int:
        return len(self.text.split())


# ---------------------------------------------------------------------------
# Phase 2/4/5 — structured LLM output for the final document
# ---------------------------------------------------------------------------

class Section(BaseModel):
    """One section of the final summary document."""
    heading: str = Field(description="Short, descriptive section title")
    timestamp: Optional[str] = Field(
        default=None,
        description="Approximate MM:SS or H:MM:SS timestamp where this section starts in the video",
    )
    bullets: List[str] = Field(
        default_factory=list,
        description="Key points for this section, as short, self-contained bullet points",
    )
    detail: Optional[str] = Field(
        default=None,
        description="1-3 sentences of connective prose expanding on the bullets, optional",
    )


class VideoSummary(BaseModel):
    """The final structured document that gets rendered into the PDF."""
    title: str = Field(description="A clear, descriptive title for the video/document")
    overview: str = Field(description="A 2-4 sentence summary of what the whole video covers")
    key_takeaways: List[str] = Field(
        default_factory=list,
        description="3-7 top-level takeaways a reader should remember, most important first",
    )
    sections: List[Section] = Field(default_factory=list)


class ChunkSummary(BaseModel):
    """Intermediate per-chunk summary (Phase 4, map step)."""
    chunk_index: int
    start_time: float
    end_time: float
    heading: str = Field(description="Short title describing what this part of the video covers")
    bullets: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Phase 6 — job / API schemas
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    PENDING = "pending"
    FETCHING_TRANSCRIPT = "fetching_transcript"
    CHUNKING = "chunking"
    SUMMARIZING_CHUNKS = "summarizing_chunks"
    SYNTHESIZING = "synthesizing"
    GENERATING_PDF = "generating_pdf"
    DONE = "done"
    ERROR = "error"


class SummarizeRequest(BaseModel):
    url: str = Field(description="Any standard YouTube video URL")


class SummarizeResponse(BaseModel):
    job_id: str
    status: JobStatus


class JobRecord(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.PENDING
    progress_message: str = "Queued"
    url: str
    video_id: Optional[str] = None
    video_title: Optional[str] = None
    used_whisper_fallback: bool = False
    num_chunks: Optional[int] = None
    pdf_filename: Optional[str] = None
    error: Optional[str] = None
    created_at: float
    updated_at: float
