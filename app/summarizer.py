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


CATEGORY_INSTRUCTIONS = {
    "lecture": """
SPECIAL CATEGORY FOCUS — ACADEMIC LECTURE & STUDY NOTES:
Format and structure this document specifically as comprehensive, ultra-detailed academic study notes:
- Emphasize core definitions, formulas, theories, historical backgrounds, proofs, main equations, and conceptual frameworks.
- Break sections down like textbook chapters with clear, educational, and conceptual headings.
- Include a "Student Self-Study Quiz / Exam Review Q&A" section in deep_dive_qa with 4-6 high-yield questions containing exhaustive, paragraph-long explanations.
- Format key_takeaways as core study concepts to memorize, explaining the "why" and "how" behind each.
- Ensure every bullet point in the sections is a rich, multi-sentence conceptual explanation. If the speaker explains a topic, expand it fully using their examples, equations, or analogies.
""",
    "sports": """
SPECIAL CATEGORY FOCUS — SPORTS MATCH RECAP & HIGHLIGHTS:
Format and structure this report specifically as a high-energy, in-depth sports match breakdown:
- Highlight key match moments, score timelines, play-by-play turning points, and standout player performances.
- Focus on tactical maneuvers, manager strategies, key team statistics, and detailed play analysis.
- Structure key_takeaways as the major game-changing plays and match outcomes.
""",
    "movie": """
SPECIAL CATEGORY FOCUS — FILM REVIEW & NARRATIVE PLOT DIGEST:
Format and structure this report specifically as a detailed cinematic review & plot digest:
- Provide a logline overview, character arc breakdowns, narrative plot beats, subtext, and cinematic themes.
- Highlight iconic quotes and memorable dialogue.
- Include a critical verdict and artistic review in the conclusion.
""",
    "tutorial": """
SPECIAL CATEGORY FOCUS — TECHNICAL TUTORIAL & CODING GUIDE:
Format and structure this report specifically as an exhaustive step-by-step developer tutorial:
- Focus on setup prerequisites, step-by-step commands/code snippets, system architecture, file layouts, and code logic.
- Provide a clear, detailed conceptual walk-through of the code or steps.
- Include common troubleshooting tips, edge cases, and best practices in key_terms and actionable_tips.
""",
    "business": """
SPECIAL CATEGORY FOCUS — EXECUTIVE BUSINESS BRIEF & MARKET ANALYSIS:
Format and structure this report specifically as an executive board-level brief:
- Focus on key business metrics, financial figures, market trends, strategic ROI, competitive positioning, and risks.
- Structure actionable_tips as strategic executive recommendations.
""",
    "podcast": """
SPECIAL CATEGORY FOCUS — PODCAST & INTERVIEW HIGHLIGHTS:
Format and structure this report specifically as a podcast episode digest:
- Focus on key guest insights, notable verbatim quotes, core debates, and timestamped topic shifts.
- Structure Q&A around the central questions posed to the guest.
""",
    "auto": """
AUTOMATIC ADAPTIVE MODE:
Analyze the transcript content and adapt the terminology, structure, and focus to best fit the domain (academic lecture, sports, film, coding tutorial, business, or interview).
"""
}


# ---------------------------------------------------------------------------
# Phase 2 — single-call summarization (short videos)
# ---------------------------------------------------------------------------

_SINGLE_CALL_PROMPT = """\
You are an expert principal researcher and technical analyst. You are turning a YouTube video transcript into an exhaustive, highly structured, high-quality intelligence report.

Your goal is to extract MAXIMUM knowledge, granular detail, key takeaways, and actionable insights. Write for a reader who wants complete comprehension without watching the video. Avoid vague filler like "the speaker discusses topic X". Instead, state exact facts, names, numbers, steps, technologies, arguments, and conclusions.

{category_instruction}

Produce a comprehensive structured summary containing:
- title: A clear, highly descriptive title for the document.
- tagline: A punchy 1-sentence executive subtitle summarizing the core message.
- estimated_read_time: Estimated read time (e.g., "15 min read").
- overview: A thorough, multi-paragraph Executive Summary (at least 2-3 dense paragraphs) explaining the background context, primary problem/topic, key solutions or findings, and overall impact.
- key_takeaways: 5-8 high-yield core takeaways. Each takeaway must be self-contained, highly informative, and packed with concrete facts.
- sections: A sequential set of sections covering the content in depth. For each section:
  - heading: Clear section title.
  - timestamp: Approximate timestamp (MM:SS) where this topic begins.
  - bullets: 3-8 comprehensive, detailed bullet points (each bullet should be 2-3 full sentences explaining the details, facts, examples, or steps).
  - detail: A dense, explanatory paragraph (3-4 sentences) expanding on the technical or contextual nuances.
  - key_quote: (Optional) A memorable quote, golden nugget, or key verbatim statement from the speaker in this section.
  - actionable_tips: 1-3 practical action items, recommendations, or key takeaways for this section.
- key_terms: 4-8 important technical or domain-specific terms/concepts introduced in the video, with clear, detailed definitions.
- deep_dive_qa: 3-6 comprehensive Question & Answer pairs addressing the core questions answered by the video. The answers must be fully explained and detailed.
- conclusion: A 3-4 sentence final synthesis and summary recommendation.

Video title (if known): {video_title}

Transcript:
{transcript}
"""



def summarize_single(
    transcript_text: str,
    video_title: str | None = None,
    category: str = "auto",
    **kwargs,
) -> VideoSummary:
    """Phase 2: send the whole transcript in one call."""
    client = get_client()
    cat_instr = CATEGORY_INSTRUCTIONS.get(category.lower(), CATEGORY_INSTRUCTIONS["auto"])
    prompt = _SINGLE_CALL_PROMPT.format(
        video_title=video_title or "(unknown)",
        category_instruction=cat_instr,
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
    summary = _parse_or_raise(response, VideoSummary)
    summary.category = category
    return summary


# ---------------------------------------------------------------------------
# Phase 4 — map step: summarize one chunk
# ---------------------------------------------------------------------------

_CHUNK_PROMPT = """\
This is one segment of a longer YouTube video transcript (segment \
{chunk_index_1based} of {total_chunks}, covering roughly {start} to {end} \
in the video). Auto-generated captions, so expect minor typos and no \
punctuation in places.

Summarize ONLY what is discussed in this segment with maximum detail and analytical depth:
- heading: A descriptive heading capturing the core theme/topic of this segment.
- bullets: 3-8 rich, multi-sentence bullet points capturing exact facts, numbers, equations, technical terms, methodologies, and reasoning discussed in the segment. Each bullet must be self-contained and descriptive.
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

Your task is to synthesize all segment summaries into ONE master executive intelligence document of extremely high depth and educational quality:

{category_instruction}

- title: Descriptive title.
- tagline: 1-sentence punchy subtitle.
- estimated_read_time: Estimated read time (e.g., "15 min read").
- overview: A thorough, multi-paragraph Executive Summary (at least 2-3 dense paragraphs) explaining background context, problem, solution, and educational/practical implications.
- key_takeaways: 5-8 top core takeaways (no duplicates, high density, rich information explaining the "why").
- sections: Logically merged sections. Each section must contain:
  - heading: Clear conceptual title.
  - timestamp: Approximate timestamp (MM:SS).
  - bullets: 3-8 comprehensive, detailed bullet points (each bullet should be 2-3 full sentences explaining the details, facts, examples, or steps).
  - detail: A dense, explanatory paragraph (3-4 sentences) expanding on the technical or contextual nuances.
  - key_quote: (Optional) Verbatim quote.
  - actionable_tips: 1-3 practical action items, recommendations, or key takeaways for this section.
- key_terms: 4-8 key technical terms/concepts with clear, detailed definitions.
- deep_dive_qa: 3-6 comprehensive Q&A pairs covering fundamental questions answered by the video. The answers must be fully explained and detailed.
- conclusion: Final synthesis and summary recommendation (3-4 sentences).

Video title (if known): {video_title}

Segment summaries (JSON):
{chunk_summaries_json}
"""



def synthesize_summaries(
    chunk_summaries: list[ChunkSummary],
    video_title: str | None = None,
    category: str = "auto",
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

    cat_instr = CATEGORY_INSTRUCTIONS.get(category.lower(), CATEGORY_INSTRUCTIONS["auto"])
    prompt = _SYNTHESIS_PROMPT.format(
        video_title=video_title or "(unknown)",
        category_instruction=cat_instr,
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
    summary = _parse_or_raise(response, VideoSummary)
    summary.category = category
    return summary


def summarize_long_transcript(
    chunks: list[TranscriptChunk],
    video_title: str | None = None,
    category: str = "auto",
    on_chunk_done=None,
) -> VideoSummary:
    """Full Phase 4 pipeline: map over chunks, then reduce."""
    chunk_summaries: list[ChunkSummary] = []
    for chunk in chunks:
        cs = summarize_chunk(chunk, total_chunks=len(chunks))
        chunk_summaries.append(cs)
        if on_chunk_done:
            on_chunk_done(len(chunk_summaries), len(chunks))
    return synthesize_summaries(chunk_summaries, video_title=video_title, category=category)


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


# ---------------------------------------------------------------------------
# Phase 8 — Q&A chatbot over video summary
# ---------------------------------------------------------------------------

_QA_PROMPT = """\
You are a knowledgeable AI assistant. The user has generated a detailed summary \
of a YouTube video and now wants to ask questions about it. Use ONLY the \
information from the summary below to answer the user's question. If the answer \
is not contained in the summary, say so honestly.

Be concise but thorough. Use specific facts, numbers, and details from the summary.

Video Summary:
Title: {title}
Overview: {overview}

Key Takeaways:
{takeaways}

Sections:
{sections}

Key Terms:
{terms}

Q&A from Summary:
{qa}

Conclusion: {conclusion}

---
User Question: {question}
"""


def answer_question(summary: VideoSummary, question: str) -> str:
    """Answer a user question using only the video summary as context."""
    client = get_client()

    takeaways = "\n".join(f"- {t}" for t in (summary.key_takeaways or []))

    sections_text = ""
    for i, sec in enumerate(summary.sections or []):
        sections_text += f"\n### {i+1}. {sec.heading}"
        if sec.timestamp:
            sections_text += f" [{sec.timestamp}]"
        sections_text += "\n"
        for b in (sec.bullets or []):
            sections_text += f"  - {b}\n"
        if sec.detail:
            sections_text += f"  Detail: {sec.detail}\n"
        if sec.key_quote:
            sections_text += f'  Quote: "{sec.key_quote}"\n'

    terms_text = "\n".join(
        f"- {kt.term}: {kt.definition}"
        for kt in (summary.key_terms or [])
    )

    qa_text = "\n".join(
        f"Q: {qa.question}\nA: {qa.answer}"
        for qa in (summary.deep_dive_qa or [])
    )

    prompt = _QA_PROMPT.format(
        title=summary.title or "",
        overview=summary.overview or "",
        takeaways=takeaways or "(none)",
        sections=sections_text or "(none)",
        terms=terms_text or "(none)",
        qa=qa_text or "(none)",
        conclusion=summary.conclusion or "(none)",
        question=question,
    )

    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=prompt,
    )
    text = getattr(response, "text", None)
    if not text:
        raise SummarizationError("Empty response from Gemini for Q&A")
    return text.strip()