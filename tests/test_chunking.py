"""
Offline tests for chunking.py. No network, no LLM calls — pure logic on
synthetic transcript segments.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chunking import (  # noqa: E402
    chunk_by_timestamp_gaps,
    estimate_tokens,
    format_timestamp,
    naive_chunk_by_words,
    needs_chunking,
)
from tests.sample_data import make_segments  # noqa: E402


def test_needs_chunking_short_text_is_false():
    assert needs_chunking("just a few words here", target_tokens=3000) is False


def test_needs_chunking_long_text_is_true():
    long_text = "word " * 10000
    assert needs_chunking(long_text, target_tokens=3000) is True


def test_naive_chunk_covers_full_transcript_with_no_gaps_or_overlaps():
    segments = make_segments(500)
    chunks = naive_chunk_by_words(segments, target_tokens=200)

    assert len(chunks) > 1, "500 segments should produce more than one chunk"

    # every chunk boundary should be contiguous: chunk[i].end_time == chunk[i+1].start_time
    for a, b in zip(chunks, chunks[1:]):
        assert a.end_time == b.start_time, "chunks must be contiguous with no time gaps or overlaps"

    # concatenating all chunk text should reproduce all segment text, in order
    rebuilt = " ".join(c.text for c in chunks)
    original = " ".join(s.text.strip() for s in segments)
    assert rebuilt == original


def test_naive_chunk_respects_target_size_reasonably():
    segments = make_segments(500)
    target = 150
    chunks = naive_chunk_by_words(segments, target_tokens=target)
    target_words = int(target * 0.75)

    # every chunk except possibly the last should be at or near the target
    for c in chunks[:-1]:
        assert c.word_count >= target_words, "non-final chunks should have reached the target size"


def test_timestamp_gap_chunking_prefers_natural_breaks():
    # 3 gaps inserted every 40 segments -> should bias cuts to land near those gaps
    segments = make_segments(150, gap_every=40, gap_size=5.0)
    chunks = chunk_by_timestamp_gaps(segments, target_tokens=200, gap_seconds=2.5)

    assert len(chunks) >= 2

    # contiguity still holds
    for a, b in zip(chunks, chunks[1:]):
        assert a.end_time == b.start_time

    # full coverage, no dropped text
    rebuilt = " ".join(c.text for c in chunks)
    original = " ".join(s.text.strip() for s in segments)
    assert rebuilt == original


def test_timestamp_gap_chunking_has_a_hard_cap_even_with_no_gaps():
    # No gaps at all (gap_every larger than segment count) - must still cut
    # eventually via the hard_max_multiplier safety valve, not run forever.
    segments = make_segments(1000, gap_every=10_000)
    chunks = chunk_by_timestamp_gaps(segments, target_tokens=200, gap_seconds=2.5, hard_max_multiplier=1.5)

    assert len(chunks) > 1, "should still split even with zero natural gaps"
    max_words = max(c.word_count for c in chunks)
    # hard cap is 1.5x the target word count (200 tokens * 0.75 words/token = 150 words)
    assert max_words <= int(200 * 0.75 * 1.5) + 20  # small slack for the segment that tips it over


def test_format_timestamp():
    assert format_timestamp(0) == "0:00"
    assert format_timestamp(65) == "1:05"
    assert format_timestamp(3661) == "1:01:01"


def test_estimate_tokens_scales_with_length():
    short = estimate_tokens("hello world")
    long = estimate_tokens("hello world " * 100)
    assert long > short * 50  # roughly linear, generous bound to avoid flakiness


def test_chunk_by_chapters_normal_and_fallbacks():
    from app.models import TranscriptSegment, VideoChapter
    from app.chunking import chunk_by_chapters, chunk_transcript

    # Setup dummy segments covering 0s to 120s
    segments = [
        TranscriptSegment(text=f"word {i}", start=float(i * 10), duration=5.0)
        for i in range(12)
    ]

    # Test clean fallback if chapters is empty or None
    fallback_chunks = chunk_by_chapters(segments, chapters=None, target_tokens=50)
    assert len(fallback_chunks) > 0

    # Setup 3 chapters
    chapters = [
        VideoChapter(title="Intro", start_time=0.0, end_time=40.0),
        VideoChapter(title="Deep Dive", start_time=40.0, end_time=80.0),
        VideoChapter(title="Conclusion", start_time=80.0, end_time=120.0),
    ]

    # Test chapter chunking under normal conditions (no merge/split)
    chunks = chunk_by_chapters(segments, chapters, target_tokens=100, min_ratio=0.0)
    # Since we have 3 chapters and min_ratio is 0.0, they won't merge
    assert len(chunks) == 3
    assert "[Intro]" in chunks[0].text
    assert "[Deep Dive]" in chunks[1].text
    assert "[Conclusion]" in chunks[2].text


    # Test splitting chapter that is too long
    # Target size very small to trigger splits
    split_chunks = chunk_by_chapters(segments, chapters, target_tokens=2, max_ratio=1.1)
    assert len(split_chunks) > 3
    # Check that at least one has "(Part)" in title
    has_part = any("(Part)" in c.text for c in split_chunks)
    assert has_part

    # Test merging chapters that are too short
    # Target size very large to trigger merges
    merge_chunks = chunk_by_chapters(segments, chapters, target_tokens=1000, min_ratio=0.8)
    assert len(merge_chunks) == 1
    assert "Intro & Deep Dive & Conclusion" in merge_chunks[0].text


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))

