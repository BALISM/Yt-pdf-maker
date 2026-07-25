"""
Phase 3 — Chunking long transcripts.

Context on why this exists at all: modern Gemini Flash models have ~1M token
context windows, so in raw "does it fit" terms, almost no single YouTube
video's transcript actually requires chunking. We chunk anyway, for two
reasons that matter more than the context-window ceiling:

  1. Per-chunk summaries are more focused and reliable than asking one call
     to hold a 2-hour lecture in its head and produce a clean structure.
  2. It's the "map" half of a map-reduce pipeline (Phase 4) — chunk summaries
     get synthesized into one coherent document rather than being the final
     output themselves.

Two strategies are implemented:

  - naive_chunk_by_words:      fixed-size chunks, simplest, can cut mid-thought.
  - chunk_by_timestamp_gaps:   same size target, but prefers to break at a
                                natural pause in the transcript (a gap between
                                two caption segments) near the target size,
                                which usually lines up with a topic shift.

chunk_by_timestamp_gaps is what the app uses by default; naive is kept
because it's a useful baseline to compare against and is genuinely simpler
to reason about when something looks wrong.
"""
from __future__ import annotations

from typing import List

from app.config import settings
from app.models import TranscriptChunk, TranscriptSegment

# Rough heuristic: English averages ~0.75 words per token for typical prose.
# We don't have a real Gemini tokenizer call in the hot path (it costs an API
# round trip), so this is intentionally conservative — it will slightly
# over-count tokens, which just means slightly smaller chunks, which is safe.
WORDS_PER_TOKEN = 0.75


def estimate_tokens(text: str) -> int:
    word_count = len(text.split())
    return int(word_count / WORDS_PER_TOKEN)


def needs_chunking(full_text: str, target_tokens: int | None = None) -> bool:
    target = target_tokens or settings.chunk_target_tokens
    return estimate_tokens(full_text) > target


# ---------------------------------------------------------------------------
# Strategy 1: naive fixed-size chunking
# ---------------------------------------------------------------------------

def naive_chunk_by_words(
    segments: List[TranscriptSegment],
    target_tokens: int | None = None,
) -> List[TranscriptChunk]:
    """
    Walk the segment list and cut a new chunk every time we cross the target
    word budget. Simplest possible approach — will happily cut a sentence,
    or even a thought, in half.
    """
    target = target_tokens or settings.chunk_target_tokens
    target_words = int(target * WORDS_PER_TOKEN)

    chunks: List[TranscriptChunk] = []
    buffer: List[str] = []
    buffer_words = 0
    chunk_start = segments[0].start if segments else 0.0
    last_end = chunk_start

    for seg in segments:
        buffer.append(seg.text.strip())
        buffer_words += len(seg.text.split())
        last_end = seg.end

        if buffer_words >= target_words:
            chunks.append(
                TranscriptChunk(
                    index=len(chunks),
                    text=" ".join(buffer).strip(),
                    start_time=chunk_start,
                    end_time=last_end,
                )
            )
            buffer = []
            buffer_words = 0
            chunk_start = last_end

    if buffer:
        chunks.append(
            TranscriptChunk(
                index=len(chunks),
                text=" ".join(buffer).strip(),
                start_time=chunk_start,
                end_time=last_end,
            )
        )

    return chunks


# ---------------------------------------------------------------------------
# Strategy 2: timestamp-gap-aware chunking (default)
# ---------------------------------------------------------------------------

def chunk_by_timestamp_gaps(
    segments: List[TranscriptSegment],
    target_tokens: int | None = None,
    gap_seconds: float | None = None,
    hard_max_multiplier: float = 1.6,
) -> List[TranscriptChunk]:
    """
    Same target size as the naive strategy, but once we're within range of
    the target, we look for the next "natural" break — a gap in speech of at
    least `gap_seconds` between two consecutive segments — and cut there
    instead of mid-sentence. If no gap shows up before we hit
    `hard_max_multiplier` * target words, we cut anyway so a single silent
    video can't produce one giant unbounded chunk.
    """
    target = target_tokens or settings.chunk_target_tokens
    gap_threshold = gap_seconds if gap_seconds is not None else settings.chunk_gap_seconds
    target_words = int(target * WORDS_PER_TOKEN)
    hard_max_words = int(target_words * hard_max_multiplier)

    if not segments:
        return []

    chunks: List[TranscriptChunk] = []
    buffer: List[str] = []
    buffer_words = 0
    chunk_start = segments[0].start
    prev_end = segments[0].start

    for i, seg in enumerate(segments):
        gap = seg.start - prev_end

        # Decide whether to cut BEFORE adding this segment: we're at/above
        # the soft target AND there's a natural pause here, OR we've blown
        # past the hard cap and need to cut regardless.
        at_soft_target = buffer_words >= target_words
        natural_break = gap >= gap_threshold
        past_hard_cap = buffer_words >= hard_max_words

        if buffer and ((at_soft_target and natural_break) or past_hard_cap):
            chunks.append(
                TranscriptChunk(
                    index=len(chunks),
                    text=" ".join(buffer).strip(),
                    start_time=chunk_start,
                    end_time=prev_end,
                )
            )
            buffer = []
            buffer_words = 0
            chunk_start = seg.start

        buffer.append(seg.text.strip())
        buffer_words += len(seg.text.split())
        prev_end = seg.end

    if buffer:
        chunks.append(
            TranscriptChunk(
                index=len(chunks),
                text=" ".join(buffer).strip(),
                start_time=chunk_start,
                end_time=prev_end,
            )
        )

    return chunks


def chunk_transcript(
    segments: List[TranscriptSegment],
    strategy: str = "timestamp_gap",
) -> List[TranscriptChunk]:
    """Single entry point the rest of the app calls."""
    if strategy == "naive":
        return naive_chunk_by_words(segments)
    if strategy == "timestamp_gap":
        return chunk_by_timestamp_gaps(segments)
    raise ValueError(f"Unknown chunking strategy: {strategy}")


def format_timestamp(seconds: float) -> str:
    """Render seconds as MM:SS, or H:MM:SS for anything an hour or longer."""
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"
