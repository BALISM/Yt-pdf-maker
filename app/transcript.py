"""
Phase 1 — Get the transcript.

Two paths:
  1. Primary: youtube-transcript-api pulls YouTube's own caption tracks
     (manual or auto-generated). Fast, free, no download needed.
  2. Fallback: if a video has no captions at all, download the audio with
     yt-dlp and transcribe it locally with faster-whisper.

Both paths converge on the same TranscriptResult shape so the rest of the
pipeline (chunking, summarizing) never needs to know which one was used.
"""
from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
)

from app.config import settings
from app.models import TranscriptResult, TranscriptSegment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors — deliberately specific so main.py can turn them into clean HTTP
# responses instead of generic 500s.
# ---------------------------------------------------------------------------

class TranscriptError(Exception):
    """Base class for all transcript-retrieval failures."""


class InvalidYouTubeURL(TranscriptError):
    pass


class NoTranscriptAvailable(TranscriptError):
    """Raised when both captions AND the Whisper fallback fail/are disabled."""


class VideoTooLong(TranscriptError):
    pass


# ---------------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------------

# Covers: youtube.com/watch?v=ID, youtu.be/ID, youtube.com/embed/ID,
# youtube.com/shorts/ID, youtube.com/v/ID, youtube.com/live/ID,
# m.youtube.com and music.youtube.com variants, with or without extra query
# params (t=, si=, list=, ...), with or without scheme/www.
_VIDEO_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{11}$")

_PATH_PATTERNS = [
    re.compile(r"^/(?:embed|shorts|v|live)/([a-zA-Z0-9_-]{11})"),
]


def extract_video_id(url: str) -> str:
    """
    Pull an 11-character YouTube video ID out of any common URL shape.
    Raises InvalidYouTubeURL if nothing recognizable is found.
    """
    if not url or not isinstance(url, str):
        raise InvalidYouTubeURL("Empty or non-string URL")

    url = url.strip()

    # Allow a bare video ID to be passed directly.
    if _VIDEO_ID_RE.match(url):
        return url

    # Ensure it parses as a URL at all (add a scheme if missing so urlparse
    # doesn't dump everything into `.path`).
    parse_target = url if "://" in url else f"https://{url}"
    parsed = urlparse(parse_target)

    host = (parsed.hostname or "").lower().removeprefix("www.").removeprefix("m.")

    if host not in {
        "youtube.com",
        "music.youtube.com",
        "youtube-nocookie.com",
        "youtu.be",
    }:
        raise InvalidYouTubeURL(f"Not a recognized YouTube host: {parsed.hostname!r}")

    # youtu.be/<id>
    if host == "youtu.be":
        candidate = parsed.path.lstrip("/").split("/")[0]
        if _VIDEO_ID_RE.match(candidate):
            return candidate
        raise InvalidYouTubeURL(f"Could not parse video ID from short URL: {url}")

    # youtube.com/watch?v=<id>
    query = parse_qs(parsed.query)
    if "v" in query and _VIDEO_ID_RE.match(query["v"][0]):
        return query["v"][0]

    # youtube.com/embed/<id>, /shorts/<id>, /v/<id>, /live/<id>
    for pattern in _PATH_PATTERNS:
        match = pattern.match(parsed.path)
        if match:
            return match.group(1)

    raise InvalidYouTubeURL(f"Could not extract a video ID from: {url}")


# ---------------------------------------------------------------------------
# Primary path — YouTube captions
# ---------------------------------------------------------------------------

def _fetch_captions(video_id: str) -> TranscriptResult:
    """
    Try to fetch captions for a video. Prefers manually-created transcripts,
    falls back to auto-generated ones, and tries a small set of common
    languages before giving up and taking whatever is available.
    """
    ytt_api = YouTubeTranscriptApi()

    try:
        transcript_list = ytt_api.list(video_id)
    except TranscriptsDisabled as e:
        raise NoTranscriptAvailable("Captions are disabled for this video") from e
    except VideoUnavailable as e:
        raise NoTranscriptAvailable("Video is unavailable (private, deleted, or region-locked)") from e
    except RequestBlocked as e:
        # Very common when this runs on a cloud server (AWS/GCP/Azure IP
        # ranges get blocked by YouTube). Not fixable by retrying - see the
        # "Working around IP bans" section of the youtube-transcript-api
        # README for using a residential proxy (WebshareProxyConfig).
        raise NoTranscriptAvailable(
            "YouTube blocked this server's IP address (common on cloud hosts). "
            "This needs a residential proxy to fix, not a retry - "
            "see youtube_transcript_api.proxies.WebshareProxyConfig."
        ) from e
    except CouldNotRetrieveTranscript as e:
        raise NoTranscriptAvailable(f"Could not retrieve transcript list: {e}") from e

    preferred_languages = ["en", "en-US", "en-GB"]

    transcript = None
    # 1) Manually created transcript in a preferred language
    try:
        transcript = transcript_list.find_manually_created_transcript(preferred_languages)
    except NoTranscriptFound:
        pass
    # 2) Auto-generated transcript in a preferred language
    if transcript is None:
        try:
            transcript = transcript_list.find_generated_transcript(preferred_languages)
        except NoTranscriptFound:
            pass
    # 3) Whatever is available at all, in any language
    if transcript is None:
        try:
            transcript = next(iter(transcript_list))
        except StopIteration:
            raise NoTranscriptAvailable("No transcripts of any language are available")

    fetched = transcript.fetch()

    segments = [
        TranscriptSegment(text=s.text, start=s.start, duration=s.duration)
        for s in fetched
        if s.text and s.text.strip()
    ]

    if not segments:
        raise NoTranscriptAvailable("Transcript was returned but contained no text")

    return TranscriptResult(
        video_id=video_id,
        language_code=transcript.language_code,
        is_generated=transcript.is_generated,
        source="captions",
        segments=segments,
    )


# ---------------------------------------------------------------------------
# Fallback path — download audio + local Whisper transcription
# ---------------------------------------------------------------------------

def _download_audio(video_id: str, dest_dir: Path) -> Path:
    """Download the best available audio track for a video via yt-dlp."""
    import yt_dlp  # imported lazily: heavy dependency, only needed on this path

    out_template = str(dest_dir / f"{video_id}.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }
    url = f"https://www.youtube.com/watch?v={video_id}"
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    mp3_path = dest_dir / f"{video_id}.mp3"
    if not mp3_path.exists():
        # Some formats/postprocessor combos land under a different extension.
        candidates = list(dest_dir.glob(f"{video_id}.*"))
        if not candidates:
            raise NoTranscriptAvailable("yt-dlp did not produce an audio file")
        mp3_path = candidates[0]
    return mp3_path


def _transcribe_with_whisper(audio_path: Path, video_id: str) -> TranscriptResult:
    """Transcribe a local audio file with faster-whisper."""
    from faster_whisper import WhisperModel  # heavy dependency, imported lazily

    model = WhisperModel(
        settings.whisper_model_size,
        device=settings.whisper_device,
        compute_type="int8" if settings.whisper_device == "cpu" else "float16",
    )
    segments_iter, info = model.transcribe(str(audio_path), beam_size=5)

    segments = [
        TranscriptSegment(text=seg.text.strip(), start=seg.start, duration=seg.end - seg.start)
        for seg in segments_iter
        if seg.text and seg.text.strip()
    ]

    if not segments:
        raise NoTranscriptAvailable("Whisper produced an empty transcript")

    return TranscriptResult(
        video_id=video_id,
        language_code=info.language,
        is_generated=True,
        source="whisper",
        segments=segments,
    )


def _fetch_via_whisper(video_id: str) -> TranscriptResult:
    with tempfile.TemporaryDirectory(prefix="ytpdf_audio_") as tmp:
        tmp_dir = Path(tmp)
        logger.info("No captions found for %s — downloading audio for Whisper fallback", video_id)
        try:
            audio_path = _download_audio(video_id, tmp_dir)
        except NoTranscriptAvailable:
            raise
        except Exception as e:
            # yt-dlp raises its own broad exception types (network errors,
            # geo-blocks, age gates, etc). Wrap them so callers only ever
            # have to handle our TranscriptError hierarchy.
            raise NoTranscriptAvailable(f"Audio download failed, so the Whisper fallback could not run: {e}") from e

        try:
            return _transcribe_with_whisper(audio_path, video_id)
        except NoTranscriptAvailable:
            raise
        except Exception as e:
            raise NoTranscriptAvailable(f"Whisper transcription failed: {e}") from e


def _fetch_chapters(url: str) -> List[VideoChapter] | None:
    """Try to extract chapter information from YouTube metadata using yt-dlp."""
    try:
        import yt_dlp
        from app.models import VideoChapter
        ydl_opts = {
            "extract_flat": False,
            "skip_download": True,
            "quiet": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            raw_chapters = info.get("chapters")
            if raw_chapters:
                return [
                    VideoChapter(
                        title=c.get("title", f"Chapter {i+1}"),
                        start_time=float(c.get("start_time", 0.0)),
                        end_time=float(c.get("end_time", 0.0)),
                    )
                    for i, c in enumerate(raw_chapters)
                ]
    except Exception as e:
        logger.warning("Failed to fetch chapters metadata via yt-dlp: %s", e)
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def get_transcript(url: str, allow_whisper_fallback: bool = True) -> TranscriptResult:
    """
    Orchestrator: URL -> video ID -> captions, falling back to Whisper.
    This is the one function the rest of the app should call.
    """
    video_id = extract_video_id(url)

    try:
        result = _fetch_captions(video_id)
    except NoTranscriptAvailable:
        if not allow_whisper_fallback:
            raise
        result = _fetch_via_whisper(video_id)

    if result.duration_seconds > settings.max_video_duration_seconds:
        raise VideoTooLong(
            f"Video is {result.duration_seconds/60:.0f} min long, which exceeds the "
            f"{settings.max_video_duration_seconds/60:.0f} min limit"
        )

    # Fetch and attach chapters metadata
    from app.models import VideoChapter
    result.chapters = _fetch_chapters(url)

    return result


def transcript_to_text(result: TranscriptResult) -> str:
    return result.full_text

