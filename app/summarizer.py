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
You are a university professor creating textbook-quality study notes. This is NOT a transcript summary — it is a complete learning resource.

CRITICAL INSTRUCTIONS:
1. IDENTIFY THE CORE ACADEMIC TOPIC(S) from the transcript (e.g., "Time Complexity", "Big-O Notation", "Sorting Algorithms", "Data Structures", etc.).
2. USE YOUR OWN EXPERT KNOWLEDGE to massively enrich the notes beyond what the speaker said. The speaker may explain things briefly or skip details — you must fill in all the gaps with proper academic content.
3. For every concept mentioned in the video, provide:
   - The formal/textbook definition (not just the speaker's casual explanation)
   - Mathematical notation and formulas where applicable (e.g., O(n log n), T(n) = 2T(n/2) + O(n))
   - At least 1-2 concrete worked examples that illustrate the concept
   - Common misconceptions students have about this topic
   - How this concept connects to related topics in the field
4. Structure sections like textbook chapters with clear, educational headings.
5. In key_terms, provide rigorous academic definitions — not simplified one-liners.
6. In deep_dive_qa, create 5-6 exam-style questions with exhaustive, multi-paragraph answers that a student could use to study for a test. Include worked-through examples in the answers.
7. In actionable_tips, include specific study strategies, practice problem suggestions, and conceptual checkpoints.
8. The overview must explain the prerequisite knowledge needed, the learning objectives, and why this topic matters in the broader field.
9. Format key_takeaways as core study concepts that explain the "why" and "how" — not just the "what".
10. If the video covers algorithms, data structures, math, or CS topics: include Big-O complexities, space complexities, best/worst/average case analysis, comparison tables, and step-by-step algorithm traces where relevant.

The final output should be so detailed and well-explained that a student could study ONLY from these notes and fully understand the topic — even if they never watch the video.
""",
    "sports": """
SPECIAL CATEGORY FOCUS — SPORTS MATCH RECAP & HIGHLIGHTS:
You are a sports analyst creating a comprehensive match report.
- Use your knowledge of the teams, players, and sport to add context beyond what the commentator says.
- Highlight key match moments, score timelines, play-by-play turning points, and standout player performances.
- Add tactical analysis: formations, strategy shifts, and coaching decisions.
- Include relevant historical stats or records that contextualize the match.
- Structure key_takeaways as the major game-changing plays and match outcomes.
""",
    "movie": """
SPECIAL CATEGORY FOCUS — FILM REVIEW & NARRATIVE PLOT DIGEST:
You are a film critic creating an in-depth cinematic analysis.
- Use your knowledge of film theory, the director's filmography, and cinematic techniques to enrich the review.
- Provide a logline overview, character arc breakdowns, narrative plot beats, subtext, thematic analysis, and cinematography notes.
- Compare to similar films in the genre where relevant.
- Highlight iconic quotes and memorable dialogue.
- Include a critical verdict and artistic review in the conclusion.
""",
    "tutorial": """
SPECIAL CATEGORY FOCUS — TECHNICAL TUTORIAL & CODING GUIDE:
You are a senior developer creating an exhaustive technical reference guide.
- Use your own technical knowledge to supplement the tutorial with best practices, alternative approaches, and deeper explanations of WHY each step works.
- Focus on setup prerequisites, step-by-step commands/code snippets, system architecture, file layouts, and code logic.
- Add explanations of underlying concepts (e.g., if the tutorial uses async/await, explain the event loop).
- Include common pitfalls, debugging tips, performance considerations, and security best practices in key_terms and actionable_tips.
- Provide links-style references to official documentation topics where relevant.
""",
    "business": """
SPECIAL CATEGORY FOCUS — EXECUTIVE BUSINESS BRIEF & MARKET ANALYSIS:
You are a strategy consultant creating a board-level intelligence brief.
- Use your knowledge of business frameworks (SWOT, Porter's Five Forces, etc.) to add analytical depth.
- Focus on key business metrics, financial figures, market trends, strategic ROI, competitive positioning, and risks.
- Contextualize the discussion within broader industry trends.
- Structure actionable_tips as strategic executive recommendations with clear rationale.
""",
    "podcast": """
SPECIAL CATEGORY FOCUS — PODCAST & INTERVIEW HIGHLIGHTS:
You are a journalist creating a comprehensive interview digest.
- Use your knowledge of the guest's background and the topic to add context beyond what was said.
- Focus on key guest insights, notable verbatim quotes, core debates, and timestamped topic shifts.
- Highlight areas of agreement, disagreement, and nuance in the discussion.
- Structure Q&A around the central questions posed to the guest.
""",
    "auto": """
AUTOMATIC ADAPTIVE MODE:
First, identify the primary domain of this content (academic lecture, sports, film, coding tutorial, business, or interview).
Then adapt your approach: use your own expert knowledge of that domain to enrich and supplement the transcript content. Do not just summarize — teach, explain, and contextualize.
"""
}


# ---------------------------------------------------------------------------
# Phase 2 — single-call summarization (short videos)
# ---------------------------------------------------------------------------

_SINGLE_CALL_PROMPT = """\
You are an expert principal researcher, subject-matter specialist, and technical analyst. You are turning a YouTube video transcript into an exhaustive, highly structured, textbook-quality intelligence report.

IMPORTANT — YOUR ROLE IS NOT JUST A SUMMARIZER:
1. First, identify the core topic(s) and academic/professional domain of the video.
2. Extract all information from the transcript with maximum detail.
3. Then AUGMENT the content using your own expert knowledge of the topic:
   - Add formal definitions where the speaker uses casual language
   - Include additional examples, analogies, and worked-through illustrations
   - Provide mathematical notation, formulas, or technical specifications where applicable
   - Fill in gaps the speaker skipped or glossed over
   - Add prerequisite context so the reader understands foundational concepts
   - Connect ideas to the broader field or related topics

Write for a reader who wants COMPLETE mastery of the topic without watching the video. The output should be educational, detailed, and serve as a standalone learning resource. Avoid vague filler like "the speaker discusses topic X". Instead, EXPLAIN the topic itself with exact facts, definitions, examples, numbers, steps, and conclusions.

{category_instruction}

Produce a comprehensive structured summary containing:
- title: A clear, highly descriptive title for the document.
- tagline: A punchy 1-sentence executive subtitle summarizing the core message.
- estimated_read_time: Estimated read time (e.g., "15 min read").
- overview: A thorough Executive Summary (at least 2-3 dense paragraphs) covering: prerequisite knowledge needed, the primary topic and why it matters, key concepts covered, and the practical takeaway. This should read like a textbook introduction.
- key_takeaways: 6-8 high-yield core takeaways. Each must be a self-contained, richly detailed statement explaining the concept, its significance, and how it works — not just naming it.
- sections: A sequential set of sections covering the content in depth. For each section:
  - heading: Clear, educational section title.
  - timestamp: Approximate timestamp (MM:SS) where this topic begins.
  - bullets: 4-8 comprehensive bullet points. EACH bullet MUST be 2-4 full sentences that explain the concept in detail with examples, numbers, or step-by-step reasoning. Do NOT write one-line bullets.
  - detail: A dense explanatory paragraph (4-6 sentences) that provides deeper context, connects this section to prior knowledge, explains WHY something works the way it does, or walks through an example.
  - key_quote: (Optional) A memorable or important verbatim statement from the speaker.
  - actionable_tips: 1-3 practical action items — for academic content, these should be study strategies, practice exercises, or conceptual checkpoints.
- key_terms: 6-10 important terms/concepts with rigorous, detailed definitions (2-3 sentences each). Include formal definitions, not just casual explanations.
- deep_dive_qa: 4-6 comprehensive Q&A pairs. Questions should test deep understanding. Answers must be thorough (paragraph-length) with examples, comparisons, or worked-through solutions where applicable.
- conclusion: A 3-5 sentence final synthesis covering what was learned, how it fits into the bigger picture, and recommended next steps for further learning.

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

Extract and explain what is discussed in this segment with maximum detail and analytical depth. \
Where the speaker mentions a concept briefly, use your own expert knowledge to expand the explanation \
with formal definitions, examples, or technical context.

- heading: A descriptive heading capturing the core theme/topic of this segment.
- bullets: 4-8 rich, multi-sentence bullet points. Each bullet must:
  - Capture the exact facts, numbers, equations, technical terms, and reasoning discussed
  - Expand on concepts using your own knowledge where the speaker's explanation is brief or incomplete
  - Be self-contained and educational — a reader should learn the concept from the bullet alone
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
You are an expert subject-matter specialist. You are given section summaries produced from consecutive segments of ONE YouTube video, in chronological order.

Your task is to synthesize all segment summaries into ONE master intelligence document of textbook-quality depth:

IMPORTANT — DO NOT JUST MERGE THE SUMMARIES:
1. Identify the core academic/professional topic from the segment summaries.
2. Use your own expert knowledge to ENRICH the content: add formal definitions, additional examples, deeper explanations, mathematical notation, and context that the original video may not have covered thoroughly.
3. The final document should be a standalone educational resource — a reader should be able to learn the topic fully from this document alone.

{category_instruction}

- title: Descriptive title.
- tagline: 1-sentence punchy subtitle.
- estimated_read_time: Estimated read time (e.g., "15 min read").
- overview: A thorough Executive Summary (at least 2-3 dense paragraphs) covering: prerequisite knowledge, what the video teaches, why it matters, and key conclusions. Write this like a textbook chapter introduction.
- key_takeaways: 6-8 top core takeaways (no duplicates). Each must richly explain the concept, its significance, and how it works.
- sections: Logically merged sections. Each section must contain:
  - heading: Clear, educational title.
  - timestamp: Approximate timestamp (MM:SS).
  - bullets: 4-8 comprehensive bullet points. EACH bullet MUST be 2-4 full sentences with detailed explanations, examples, formulas, or step-by-step reasoning. Augment with your own knowledge.
  - detail: A dense explanatory paragraph (4-6 sentences) providing deeper context, connecting to prior knowledge, and explaining WHY things work the way they do.
  - key_quote: (Optional) Verbatim quote from the speaker.
  - actionable_tips: 1-3 practical action items, study strategies, or conceptual checkpoints.
- key_terms: 6-10 key terms with rigorous, detailed definitions (2-3 sentences each).
- deep_dive_qa: 4-6 comprehensive Q&A pairs. Answers must be thorough (paragraph-length) with examples or worked-through solutions.
- conclusion: Final synthesis (3-5 sentences) covering what was learned, how it fits into the bigger picture, and recommended next steps.

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