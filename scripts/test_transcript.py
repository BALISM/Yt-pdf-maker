"""
Phase 1 checkpoint script — no FastAPI, no chunking, no Gemini. Just: does
transcript retrieval actually work, in isolation, before wiring anything
else up?

Run it two ways:

  1. Unit-style checks with NO network (URL parsing only):
       python scripts/test_transcript.py --parse-only

  2. Live checks against real videos (needs internet + this script run
     from your own machine, not a locked-down sandbox):
       python scripts/test_transcript.py
       python scripts/test_transcript.py "https://youtu.be/dQw4w9WgXcQ"

  With no URL given, it runs through a small default list covering short
  videos, long videos, and shorts/youtu.be URL formats, so you can get a
  feel for real-world transcript quality across a few different videos
  before building anything else on top of it.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.transcript import (  # noqa: E402
    InvalidYouTubeURL,
    NoTranscriptAvailable,
    TranscriptError,
    extract_video_id,
    get_transcript,
)

# A handful of URL shapes to sanity-check extract_video_id against, with the
# expected video ID. Pure string logic - no network needed.
URL_PARSE_CASES = [
    ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://youtube.com/watch?v=dQw4w9WgXcQ&list=PL123&index=2", "dQw4w9WgXcQ"),
    ("http://m.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://youtu.be/dQw4w9WgXcQ?si=abc123", "dQw4w9WgXcQ"),
    ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://www.youtube.com/live/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("dQw4w9WgXcQ", "dQw4w9WgXcQ"),  # bare ID
]

URL_PARSE_FAILURE_CASES = [
    "https://www.vimeo.com/12345",
    "not a url at all",
    "",
]

# A few real videos worth testing live: short, long, and an alternate URL
# shape. Swap these for whatever you want to try.
DEFAULT_LIVE_VIDEOS = [
    "https://www.youtube.com/watch?v=jNQXAC9IVRw",  # "Me at the zoo" - first YouTube video, very short
    "https://youtu.be/dQw4w9WgXcQ",  # classic short music video, youtu.be shape
]


def run_parse_checks() -> bool:
    print("=== URL parsing checks (no network) ===")
    all_ok = True
    for url, expected in URL_PARSE_CASES:
        try:
            got = extract_video_id(url)
            ok = got == expected
            all_ok &= ok
            print(f"  {'OK ' if ok else 'FAIL'}  {url!r:60s} -> {got!r} (expected {expected!r})")
        except Exception as e:
            all_ok = False
            print(f"  FAIL  {url!r:60s} -> raised {e!r} (expected {expected!r})")

    for url in URL_PARSE_FAILURE_CASES:
        try:
            got = extract_video_id(url)
            all_ok = False
            print(f"  FAIL  {url!r:60s} -> unexpectedly parsed as {got!r}")
        except InvalidYouTubeURL:
            print(f"  OK    {url!r:60s} -> correctly rejected")
        except Exception as e:
            all_ok = False
            print(f"  FAIL  {url!r:60s} -> raised wrong exception type {type(e).__name__}")

    print()
    print("All parse checks passed!" if all_ok else "Some parse checks FAILED.")
    return all_ok


def run_live_checks(urls: list[str]) -> None:
    print("=== Live transcript retrieval checks (needs network) ===")
    for url in urls:
        print(f"\n--> {url}")
        try:
            result = get_transcript(url, allow_whisper_fallback=False)
        except NoTranscriptAvailable as e:
            print(f"    No captions available (this is where Whisper fallback would kick in): {e}")
            continue
        except TranscriptError as e:
            print(f"    Failed: {e}")
            continue

        text = result.full_text
        print(f"    video_id        = {result.video_id}")
        print(f"    language        = {result.language_code} (generated={result.is_generated})")
        print(f"    segments        = {len(result.segments)}")
        print(f"    duration        ~ {result.duration_seconds/60:.1f} min")
        print(f"    total chars     = {len(text)}")
        print(f"    first 200 chars = {text[:200]!r}")


if __name__ == "__main__":
    args = sys.argv[1:]

    if "--parse-only" in args:
        ok = run_parse_checks()
        sys.exit(0 if ok else 1)

    run_parse_checks()
    print()

    urls = [a for a in args if a != "--parse-only"] or DEFAULT_LIVE_VIDEOS
    run_live_checks(urls)
