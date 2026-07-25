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
You are an expert principal researcher and technical analyst. You are turning a YouTube video transcript into an exhaustive, highly structured, executive-grade intelligence report.

Your goal is to extract MAXIMUM knowledge, granular detail, key takeaways, and actionable insights. Write for a reader who wants complete comprehension without watching the video. Avoid vague filler like "the speaker discusses topic X". Instead, state exact facts, names, numbers, steps, technologies, arguments, and conclusions.

Produce a comprehensive structured summary containing:
- title: A clear, highly descriptive title for the document.
- tagline: A punchy 1-sentence executive subtitle summarizing the core message.
- estimated_read_time: Estimated read time (e.g., "5 min read").
- overview: A thorough 2-3 paragraph Executive Summary explaining the background context, primary problem/topic, key solutions or findings, and overall impact.
- key_takeaways: 5-8 high-yield core takeaways. Each takeaway must be self-contained, highly informative, and packed with concrete facts.
- sections: A sequential set of sections covering the content in depth. For each section:
  - heading: Clear section title.
  - timestamp: Approximate timestamp (MM:SS) where this topic begins.
  - bullets: 3-8 comprehensive bullet points with specific facts, examples, steps, or explanations.
  - detail: 2-4 sentences of deep connective prose expanding on the technical or contextual nuances.
  - key_quote: (Optional) A memorable quote, golden nugget, or key verbatim statement from the speaker in this section.
  - actionable_tips: 1-3 practical action items, recommendations, or key takeaways for this section.
- key_terms: 4-8 important technical or domain-specific terms/concepts introduced in the video, with clear definitions.
- deep_dive_qa: 3-6 comprehensive Question & Answer pairs addressing the core questions answered by the video.
- conclusion: A 2-3 sentence final synthesis and summary recommendation.

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

Summarize ONLY what is discussed in this segment with high analytical detail:
- heading: A short heading describing what this part of the video covers.
- bullets: 3-8 concise but detailed bullet points capturing exact facts, numbers, methodologies, or arguments.
- key_quote: (Optional) Any notable quote or key statement in this segment.

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
                    "key_quote": {"type": "string"},
                },
                "required": ["heading", "bullets"],
            },
        ),
    )
    data = _parse_json_or_raise(response)
    return ChunkSummary(
        chunk_index=chunk.index,
        start_time=chunk.start_time,
        end_time=chunk.end_time,
        heading=data.get("heading", f"Segment {chunk.index + 1}"),
        bullets=data.get("bullets", []),
        key_quote=data.get("key_quote"),
    )


# ---------------------------------------------------------------------------
# Phase 4 — reduce step: synthesize all chunk summaries into one document
# ---------------------------------------------------------------------------

_SYNTHESIS_PROMPT = """\
You are given section summaries produced from consecutive segments of ONE YouTube video, in chronological order.

Your task is to synthesize all segment summaries into ONE master executive intelligence document:
- title: Descriptive title.
- tagline: 1-sentence punchy subtitle.
- estimated_read_time: Estimated read time.
- overview: 2-3 paragraph Executive Summary explaining context, problem, solution, and implications.
- key_takeaways: 5-8 top core takeaways (no duplicates, high density).
- sections: Logically merged sections with headings, timestamps (MM:SS), comprehensive bullets, explanatory detail paragraphs, key quotes, and actionable tips.
- key_terms: 4-8 key technical terms/concepts with definitions.
- deep_dive_qa: 3-6 comprehensive Q&A pairs covering fundamental questions answered by the video.
- conclusion: Final synthesis and summary recommendation.

Video title (if known): {video_title}

Segment summaries (JSON):
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