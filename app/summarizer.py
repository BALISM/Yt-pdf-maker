"""
Phase 2 — one-shot summarization for short transcripts.
Phase 4 — map-reduce summarization for long transcripts:
    map:    summarize_chunk() runs once per chunk (Phase 3 output)
    reduce: synthesize_summaries() merges all chunk summaries into one
            coherent VideoSummary, deduplicating repeated points and
            resolving references across chunks.

Both paths return the same `VideoSummary` structured object (see models.py),
so pdf_generator.py never needs to know whether chunking happened at all.
"""
from __future__ import annotations

import json
import logging

from google import genai
from google.genai import types

from app.config import settings, require_gemini_key
from app.models import ChunkSummary, Section, TranscriptChunk, VideoSummary

logger = logging.getLogger(__name__)

_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=require_gemini_key())
    return _client


# ---------------------------------------------------------------------------
# Phase 2 — single-call summarization (short videos)
# ---------------------------------------------------------------------------

_SINGLE_CALL_PROMPT = """\
You are turning a YouTube video transcript into a clear, well-organized \
study document. You will be given the raw transcript text (auto-generated \
captions, so expect minor typos, run-on sentences, and no punctuation in \
places).

Produce a structured summary of the video:
- A short, descriptive title (don't just repeat "Video Summary").
- A 2-4 sentence overview of what the video covers as a whole.
- 3-7 key takeaways: the most important points a reader should remember, \
most important first.
- A set of sections that walk through the video's content in order. Each \
section needs a short heading, a handful of concise bullet points \
capturing the substance (not vague filler like "the speaker discusses \
various topics"), and optionally 1-3 sentences of connective detail.

Write for someone who has NOT watched the video and wants to understand it \
from this document alone. Be concrete: names, numbers, examples, and \
conclusions from the transcript matter more than restating that "the \
speaker talks about X".

Video title (if known): {video_title}

Transcript:
{transcript}
"""


def summarize_single(transcript_text: str, video_title: str | None = None) -> VideoSummary:
    """Phase 2: send the whole transcript in one call. Only safe for videos
    short enough that the transcript comfortably fits in a single request —
    see chunking.needs_chunking() for the decision."""
    client = get_client()
    prompt = _SINGLE_CALL_PROMPT.format(
        video_title=video_title or "(unknown)",
        transcript=transcript_text,
    )

    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=VideoSummary,
            temperature=0.3,
        ),
    )
    return _parse_or_raise(response, VideoSummary)


# ---------------------------------------------------------------------------
# Phase 4 — map step: summarize one chunk
# ---------------------------------------------------------------------------

_CHUNK_PROMPT = """\
This is one segment of a longer YouTube video transcript (segment \
{chunk_index_1based} of {total_chunks}, covering roughly {start} to {end} \
in the video). Auto-generated captions, so expect minor typos and no \
punctuation in places.

Summarize ONLY what is discussed in this segment:
- A short heading describing what this part of the video covers.
- 2-6 concise bullet points capturing the substance of this segment \
(concrete points, not "the speaker continues talking").

Do not try to summarize the whole video — you only have this one segment. \
Do not invent a conclusion or wrap-up unless this segment actually contains one.

Segment transcript:
{chunk_text}
"""


def summarize_chunk(chunk: TranscriptChunk, total_chunks: int) -> ChunkSummary:
    from app.chunking import format_timestamp

    client = get_client()
    prompt = _CHUNK_PROMPT.format(
        chunk_index_1based=chunk.index + 1,
        total_chunks=total_chunks,
        start=format_timestamp(chunk.start_time),
        end=format_timestamp(chunk.end_time),
        chunk_text=chunk.text,
    )

    # Lightweight inline JSON schema for the map step: just heading + bullets.
    # Timing is attached after parsing since the model doesn't need to invent it.
    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema={
                "type": "object",
                "properties": {
                    "heading": {"type": "string"},
                    "bullets": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["heading", "bullets"],
            },
            temperature=0.3,
        ),
    )
    data = _parse_json_or_raise(response)
    return ChunkSummary(
        chunk_index=chunk.index,
        start_time=chunk.start_time,
        end_time=chunk.end_time,
        heading=data.get("heading", f"Segment {chunk.index + 1}"),
        bullets=data.get("bullets", []),
    )


# ---------------------------------------------------------------------------
# Phase 4 — reduce step: synthesize all chunk summaries into one document
# ---------------------------------------------------------------------------

_SYNTHESIS_PROMPT = """\
You are given a list of section summaries that were produced independently \
from consecutive segments of ONE YouTube video, in chronological order. \
Because each one was written without seeing the others, there may be:
- Points repeated across multiple segments (merge and deduplicate them).
- References that only make sense with context from an earlier segment \
(resolve them using that earlier context instead of leaving them vague).
- Uneven granularity between segments (even them out).

Your job is to synthesize these into ONE coherent structured document for \
the whole video:
- A short, descriptive overall title.
- A 2-4 sentence overview of the whole video.
- 3-7 key takeaways for the whole video, most important first, with no \
duplicates.
- A final set of sections. You do not need a 1:1 mapping to the input \
segments — merge segments that cover the same topic, split a segment if it \
clearly covers two unrelated things, and keep the original approximate \
timestamp (MM:SS) for each resulting section so a reader can still jump to \
that part of the video.

Video title (if known): {video_title}

Segment summaries, in chronological order (JSON):
{chunk_summaries_json}
"""


def synthesize_summaries(
    chunk_summaries: list[ChunkSummary],
    video_title: str | None = None,
) -> VideoSummary:
    from app.chunking import format_timestamp

    client = get_client()

    serializable = [
        {
            "segment_index": cs.chunk_index + 1,
            "approx_start": format_timestamp(cs.start_time),
            "approx_end": format_timestamp(cs.end_time),
            "heading": cs.heading,
            "bullets": cs.bullets,
        }
        for cs in chunk_summaries
    ]

    prompt = _SYNTHESIS_PROMPT.format(
        video_title=video_title or "(unknown)",
        chunk_summaries_json=json.dumps(serializable, indent=2, ensure_ascii=False),
    )

    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=VideoSummary,
            temperature=0.3,
        ),
    )
    return _parse_or_raise(response, VideoSummary)


def summarize_long_transcript(
    chunks: list[TranscriptChunk],
    video_title: str | None = None,
    on_chunk_done=None,
) -> VideoSummary:
    """Full Phase 4 pipeline: map over chunks, then reduce."""
    chunk_summaries: list[ChunkSummary] = []
    for chunk in chunks:
        cs = summarize_chunk(chunk, total_chunks=len(chunks))
        chunk_summaries.append(cs)
        if on_chunk_done:
            on_chunk_done(len(chunk_summaries), len(chunks))
    return synthesize_summaries(chunk_summaries, video_title=video_title)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class SummarizationError(Exception):
    pass


def _parse_or_raise(response, model_cls):
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, model_cls):
        return parsed
    # Fall back to manual JSON parsing if .parsed didn't populate for some reason.
    data = _parse_json_or_raise(response)
    try:
        return model_cls.model_validate(data)
    except Exception as e:  # pragma: no cover - defensive
        raise SummarizationError(f"Model returned data that didn't match the expected schema: {e}") from e


def _parse_json_or_raise(response) -> dict:
    text = getattr(response, "text", None)
    if not text:
        raise SummarizationError("Empty response from Gemini")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise SummarizationError(f"Gemini did not return valid JSON: {e}") from e
