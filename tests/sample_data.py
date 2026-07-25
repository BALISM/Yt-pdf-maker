"""Synthetic transcript data used across tests — no network required."""
from app.models import TranscriptSegment

WORDS = (
    "so today we're going to talk about how neural networks actually learn "
    "from data and why gradient descent is the workhorse behind almost "
    "everything in modern machine learning stick around because this "
    "matters more than you think"
).split()


def make_segments(n: int, words_per_seg: int = 8, gap_every: int = 40, gap_size: float = 4.0):
    """Build n synthetic caption segments. Every `gap_every` segments, insert
    a long pause (gap_size seconds) to simulate a natural topic break, so
    chunk_by_timestamp_gaps has something real to key off."""
    segments = []
    t = 0.0
    for i in range(n):
        text = " ".join(WORDS[(i * words_per_seg) % len(WORDS): (i * words_per_seg) % len(WORDS) + words_per_seg]) or "hello there"
        duration = 2.0
        if i > 0 and i % gap_every == 0:
            t += gap_size
        segments.append(TranscriptSegment(text=text, start=t, duration=duration))
        t += duration
    return segments
